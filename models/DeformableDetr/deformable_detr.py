# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0
# ------------------------------------------------------------------------
# and DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
TadTR model and criterion classes.
"""
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import (accuracy)
from utils.segment_ops import segment_cw_to_t1t2,segment_iou
from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid, get_world_size, is_dist_avail_and_initialized)

from .matcher import build_matcher
from .position_encoding import build_position_encoding
from .custom_loss import sigmoid_focal_loss,softmax_ce_loss
from .transformer import build_deformable_transformer
from models.clip import build_text_encoder
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign
import json
import numpy as np
import os

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def get_norm(norm_type, dim, num_groups=None):
    if norm_type == 'gn':
        assert num_groups is not None, 'num_groups must be specified'
        return nn.GroupNorm(num_groups, dim)
    elif norm_type == 'bn':
        return nn.BatchNorm1d(dim)
    else:
        raise NotImplementedError


class DeformableDETR(nn.Module):
    """ This is the TadTR module that performs temporal action detection """

    def __init__(self, 
                    position_embedding, 
                    transformer, 
                    text_encoder, 
                    logit_scale, 
                    device, 
                    num_classes,
                    args):

        super().__init__()
        self.position_embedding = position_embedding
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.device = device
        self.num_classes = num_classes
        self.args = args

        self.num_queries = args.num_queries
        self.target_type = args.target_type
        self.aux_loss = args.aux_loss

        self.ROIalign_strategy = args.ROIalign_strategy
        self.ROIalign_size = args.ROIalign_size
        
        self.pooling_type = args.pooling_type

        self.with_iterative_refine = args.with_iterative_refine

        self.actionness_loss = args.actionness_loss

        self.enable_posPrior = args.enable_posPrior


        self.distillation_loss = args.distillation_loss

        hidden_dim = transformer.d_model


        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        if self.target_type != "none":
            self.class_embed = nn.Linear(hidden_dim,hidden_dim) # classfication matrix
            self.class_embed.bias.data = torch.ones(hidden_dim) * bias_value
        else:    
            self.class_embed = nn.Linear(hidden_dim, num_classes) # classfication head
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        self.query_embed = nn.Embedding(self.num_queries, hidden_dim*2)

        self.input_proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)


        if self.actionness_loss:
            self.actionness_embed = nn.Linear(hidden_dim,1)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.actionness_embed.bias.data = torch.ones(1) * bias_value



        num_pred = transformer.decoder.num_layers
        if self.with_iterative_refine: # specific parameters for each laryer
            self.class_embed = _get_clones(self.class_embed, num_pred)
            if self.actionness_loss:
                self.actionness_embed = _get_clones(self.actionness_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[1:], -2.0)
            # hack implementation for segment refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else: # shared parameters for each laryer
            nn.init.constant_(
                self.bbox_embed.layers[-1].bias.data[1:], -2.0)
            self.class_embed = nn.ModuleList(
                [self.class_embed for _ in range(num_pred)])
            if self.actionness_loss:
                self.actionness_embed = nn.ModuleList(
                    [self.actionness_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList(
                [self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None


    def get_text_feats(self, cl_names, description_dict, device, target_type):
        def get_prompt(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append("a video of a person doing"+" "+c)
            return temp_prompt
        
        def get_description(cl_names):
            temp_prompt = []
            for c in cl_names:
                temp_prompt.append(description_dict[c]['Elaboration']['Description'][0]) # NOTE: default the idx of description is 0.
            return temp_prompt
        
        if target_type == 'prompt':
            act_prompt = get_prompt(cl_names)
        elif target_type == 'description':
            act_prompt = get_description(cl_names)
        elif target_type == 'name':
            act_prompt = cl_names
        else: 
            raise ValueError("Don't define this text_mode.")
        
        tokens = clip_pkg.tokenize(act_prompt).long().to(device) # input_ids->input_ids:[150,length]
        text_feats = self.text_encoder(tokens).float()

        return text_feats

    def _to_roi_align_format(self, rois, truely_length, scale_factor=1):
        '''Convert RoIs to RoIAlign format.
        Params:
            RoIs: normalized segments coordinates, shape (batch_size, num_segments, 2)
            T: length of the video feature sequence
        '''
        # transform to absolute axis
        B, N = rois.shape[:2]
        rois_center = rois[:, :, 0:1] # [B,N,1]
        rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
        rois_abs = torch.cat(
            (rois_center - rois_size/2, rois_center + rois_size/2), dim=2) * truely_length # [B,N,2]->"start,end"
        # expand the RoIs
        _max = truely_length.repeat(1,N,2)
        _min = torch.zeros_like(_max)
        rois_abs = torch.clamp(rois_abs, min=_min, max=_max)  # (B, N, 2)
        # transfer to 4 dimension coordination
        rois_abs_4d = torch.zeros((B,N,4),dtype=rois_abs.dtype,device=rois_abs.device)
        rois_abs_4d[:,:,0], rois_abs_4d[:,:,2] = rois_abs[:,:,0], rois_abs[:,:,1] # x1,0,x2,0

        # add batch index
        batch_ind = torch.arange(0, B).view((B, 1, 1)).to(rois_abs.device) # [B,1,1]
        batch_ind = batch_ind.repeat(1, N, 1) # [B,N,1]
        rois_abs_4d = torch.cat((batch_ind, rois_abs_4d), dim=2) # [B,N,1+4]->"batch_id,x1,0,x2,0"
        # NOTE: stop gradient here to stablize training
        return rois_abs_4d.view((B*N, 5)).detach()

    def _roi_align(self, rois, origin_feat, mask, ROIalign_size, scale_factor=1):
        B,Q,_ = rois.shape
        B,T,C = origin_feat.shape
        truely_length = T-torch.sum(mask,dim=1) # [B]
        rois_abs_4d = self._to_roi_align_format(rois,truely_length,scale_factor)
        feat = origin_feat.permute(0,2,1) # [B,dim,T]
        feat = feat.reshape(B,C,1,T)
        roi_feat = ROIalign(feat, rois_abs_4d, output_size=(1,ROIalign_size))
        roi_feat = roi_feat.reshape(B,Q,C,-1) # [B,Q,dim,output_width]
        roi_feat = roi_feat.permute(0,1,3,2) # [B,Q,output_width,dim]
        return roi_feat

    # @torch.no_grad()
    def _compute_similarity(self, visual_feats, text_feats):
        '''
        text_feats: [num_classes,dim]
        '''
        if len(visual_feats.shape)==2: # batch_num_instance,dim
            visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = torch.einsum("bd,cd->bc",visual_feats,text_feats)*logit_scale
            return logits
        elif len(visual_feats.shape)==3:# batch,num_queries/snippet_length,dim
            visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)*logit_scale
            return logits
        elif len(visual_feats.shape)==4:# batch,num_queries,snippet_length,dim
            visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
            text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
            logit_scale = self.logit_scale.exp()
            logits = torch.einsum("bqld,cd->bqlc",visual_feats,text_feats)*logit_scale
            return logits
        else:
            raise NotImplementedError


    def _temporal_pooling(self,pooling_type,coordinate,clip_feat,mask,ROIalign_size,text_feats):
        b,t,_ = coordinate.shape
        if pooling_type == "average":
            roi_feat = self._roi_align(rois=coordinate,origin_feat=clip_feat+1e-4,mask=mask,ROIalign_size=ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            # roi_feat = roi_feat.mean(-2) # [B,Q,dim]
            if self.ROIalign_strategy == "before_pred":
                roi_feat = roi_feat.mean(-2) # [B,Q,dim]
                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,num_classes]
            elif self.ROIalign_strategy == "after_pred":
                roi_feat = roi_feat # [B,Q,L,dim]
                ROIalign_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,L,num_classes]
                ROIalign_logits = ROIalign_logits.mean(-2) # [B,Q,num_classes]
            else:
                raise NotImplementedError
        elif pooling_type == "max":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            roi_feat = roi_feat.max(dim=2)[0] # [bs,num_queries,dim]

            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "center1":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            center_idx = int(roi_feat.shape[2] / 2)
            roi_feat = roi_feat[:,:,center_idx,:] 
            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "center2":
            rois = coordinate # [b,n,2]
            rois_center = rois[:, :, 0:1] # [B,N,1]
            # rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
            truely_length = t-torch.sum(mask,dim=1) # [B]
            truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
            center_idx = (rois_center*truely_length).long() # [b,n,1]
            roi_feat = torch.gather(clip_feat + 1e-4, dim=1, index=center_idx.expand(-1, -1, clip_feat.shape[-1]))
            ROIalign_logits = self._compute_similarity(roi_feat,text_feats)
        elif pooling_type == "self_attention":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            attention_weights = F.softmax(torch.matmul(roi_feat, roi_feat.transpose(-2, -1)), dim=-1)
            roi_feat_sa = torch.matmul(attention_weights, roi_feat)
            roi_feat_sa = roi_feat_sa.mean(2)
            ROIalign_logits = self._compute_similarity(roi_feat_sa,text_feats)
        elif pooling_type == "slow_fast":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            fast_feat = roi_feat.mean(dim=2) # [b,q,d]
            step = int(self.ROIalign_size // 4)
            slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
            roi_feat_final = (fast_feat + slow_feat)/2
            ROIalign_logits = self._compute_similarity(roi_feat_final,text_feats)
        elif pooling_type == "sparse":
            roi_feat = self._roi_align(coordinate,clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
            step = int(self.ROIalign_size // 4)
            slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
            ROIalign_logits = self._compute_similarity(slow_feat,text_feats)
        else:
            raise ValueError

        return ROIalign_logits   

    def forward(self, samples: NestedTensor, classes_name, description_dict):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensors: batched images, of shape [batch_size x C x T]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 on padded pixels
            or a tuple of tensors and mask

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-action) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_segments": The normalized segments coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized segment.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
            instance_masks: a list of mask dict, {'label_name':{'mask': T, 'label_id': 1}}
        """
        if not isinstance(samples, NestedTensor):
            if isinstance(samples, (list, tuple)):
                samples = NestedTensor(*samples)
            else:
                samples = nested_tensor_from_tensor_list(samples)  # (b,t,dim)



        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                if self.args.feature_type == "ViFi-CLIP":
                    text_feats = torch.from_numpy(np.load(os.path.join(self.args.feature_path,'text_features_split75_splitID1.npy'))).float().to(self.device)
                elif self.args.feature_type == "CLIP":
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]
                else:
                    raise NotImplementedError


        pos = [self.position_embedding(samples).permute(0,2,1)] # [b,dim,t]
        # origin CLIP features
        clip_feat, mask = samples.tensors, samples.mask  # [b,t,dim], [b,t]
 
        srcs = [self.input_proj[0](clip_feat.permute(0,2,1))] # low-level Conv1d  [b,dim,t]
        masks = [mask]

        query_embeds = self.query_embed.weight # [n,2*dim]

        hs, init_reference, inter_references, memory = self.transformer(srcs, masks, pos, query_embeds)
        # record result
        out = {}
        out['memory'] = memory
        out['hs'] = hs


        outputs_coords = []
        # gather outputs from each decoder layer
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]

            reference = inverse_sigmoid(reference)
            tmp = self.bbox_embed[lvl](hs[lvl]) # [b,nq,2]
            # the l-th layer (l >= 2)
            if reference.shape[-1] == 2:
                tmp += reference
            # the first layer
            else:
                assert reference.shape[-1] == 1
                tmp[..., 0] += reference[..., 0]
            outputs_coord = tmp.sigmoid()
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords)
        out['pred_boxes'] = outputs_coord[-1]


        if self.target_type != "none":
            outputs_embed = torch.stack([self.class_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
            out['class_embs'] = outputs_embed[-1]
            outputs_logit = self._compute_similarity(outputs_embed, text_feats) # [dec_layers, b,num_queries,num_classes]
        else:
            outputs_logit = torch.stack([self.class_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
 
        out['class_logits'] = outputs_logit[-1]
        

        if self.actionness_loss :
            # compute the class-agnostic foreground score
            actionness_logits = torch.stack([self.actionness_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
            out['actionness_logits'] = actionness_logits[-1]
    

        if self.training:
            if self.distillation_loss:
                out['student_emb'] = out['class_embs'] # [b,num_queries,num_classes]

                visual_feats = clip_feat
                roi_feat = self._roi_align(out['pred_boxes'],visual_feats,mask,self.ROIalign_size).squeeze() # [bs,num_queries,ROIalign_size,dim]
                teacher_emb = roi_feat.mean(dim=2) # [b,q,d]
                out['teacher_emb'] = teacher_emb

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_logit, outputs_coord)

        return out


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_segments': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]


class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict,focal_alpha, args, base_losses=['labels', 'boxes']):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            base_losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            gamma: gamma in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.gamma = args.gamma
        self.target_type = args.target_type

        self.actionness_loss = args.actionness_loss 

        self.distillation_loss = args.distillation_loss

        if self.actionness_loss:
            self.base_losses = ['labels','actionness','boxes']
        else:
            self.base_losses = ['labels', 'boxes']

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'class_logits' in outputs
        src_logits = outputs['class_logits'] # [bs,num_queries,num_classes]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_distillation(self,outputs, targets, indices, num_boxes):
        '''
        for distillation the CLIP feat to DETR detector
        '''
        assert 'student_emb' in outputs
        assert 'teacher_emb' in outputs

        # obtain logits
        student_emb = outputs['student_emb']
        teacher_emb = outputs['teacher_emb'] # [B,Q,dim]

        loss = torch.abs(teacher_emb-student_emb).sum(-1)

        losses = {}
        losses['loss_distillation'] = loss.mean()
        return losses

    def loss_actionness(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_actionness': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx


    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'actionness': self.loss_actionness
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """

        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets

        if self.actionness_loss and self.target_type != "none": # if using the actionness loss, adopt actionness logits for set-matching
            assert "actionness_logits" in outputs_without_aux
            # We flatten to compute the cost matrices in a batch
            logits = outputs_without_aux["actionness_logits"]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets]) # [gt_instance_num]
            tgt_bbox = torch.cat([v["segments"] for v in targets]) # [gt_instance_num, 2]
            sizes = [len(v["segments"]) for v in targets]
            
        elif not self.actionness_loss or self.target_type == "none":
            assert "class_logits" in outputs_without_aux
            # We flatten to compute the cost matrices in a batch
            logits = outputs_without_aux["class_logits"]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["semantic_labels"] for v in targets]) # [gt_instance_num]
            tgt_bbox = torch.cat([v["segments"] for v in targets]) # [gt_instance_num, 2]
            sizes = [len(v["segments"]) for v in targets]
        else:
            raise ValueError
        
        indices = self.matcher(logits, outputs_without_aux["pred_boxes"], tgt_ids, tgt_bbox, sizes)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.base_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each innermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.base_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if self.distillation_loss:
            distillation_loss = self.loss_distillation(outputs, targets, indices, num_boxes)
            losses.update(distillation_loss)

        return losses

class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self,args):
        super().__init__()
        self.type = args.postprocess_type
        self.topk = args.postprocess_topk

        self.target_type = args.target_type
        self.proposals_weight_type = args.proposals_weight_type
        self.actionness_loss = args.actionness_loss
        self.prob_type = args.prob_type

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 1] containing the size of each video of the batch
        """
        out_bbox = outputs['pred_boxes'] # [bs,num_queries,2]
        

        assert 'class_logits' in outputs
        class_logits = outputs['class_logits'] #  [bs,num_queries,num_classes] 
        if self.actionness_loss: # [bs,num_queries,1]
            assert 'actionness_logits' in outputs
            actionness_logits = outputs['actionness_logits']
            actionness_prob = actionness_logits.sigmoid()

            if self.prob_type == "softmax":
                if self.proposals_weight_type == "before_softmax":
                    prob = torch.mul(actionness_prob,class_logits).softmax(-1) # [bs,num_queries,num_classes]
                elif self.proposals_weight_type == "after_softmax":
                    prob = torch.mul(actionness_prob,class_logits.softmax(-1)) # [bs,num_queries,num_classes]
            elif self.prob_type == "sigmoid":
                prob = torch.mul(actionness_prob,class_logits.sigmoid()) # [bs,num_queries,num_classes]
            elif self.prob_type == "none_mul":
                prob = class_logits.softmax(-1) # [bs,num_queries,num_classes]
            else:
                raise NotImplementedError
        else:
            if self.prob_type == "softmax":
                prob = class_logits.softmax(-1)
            elif self.prob_type == "sigmoid":
                prob = class_logits.sigmoid()
            else:
                raise NotImplementedError


        B,Q,num_classes = prob.shape
        assert len(prob) == len(target_sizes)


        if self.type == "class_agnostic":
            assert self.topk >= 1, "so small value for class_agnostic type, please check"
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(prob.reshape(B, -1), min(self.topk, Q*num_classes), dim=1) # [bs,num_queries*num_classes] - > [bs,100]
            scores = topk_values
            topk_boxes_idx = torch.div(topk_indexes, num_classes, rounding_mode='trunc') # get the row index of out_logits (b,q,c), i.e., the query idx. [bs,100//num_classes]
            labels = topk_indexes % num_classes # get the col index of out_logits (b,q,c), i.e., the class idx. [bs,100//num_classes]
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = torch.gather(out_boxes, 1, topk_boxes_idx.unsqueeze(-1).repeat(1,1,2)) # [bs,topk,2]
        elif self.type == "class_specific":
            assert self.topk <= 5, "so big value for class_specific type, please check"
            # pick topk classes for each query
            topk_values, topk_indexes = torch.topk(prob, min(self.topk,num_classes), dim=-1) # [bs,num_queries,topk]
            scores, labels = topk_values.flatten(1), topk_indexes.flatten(1) # [bs, num_queries*topk]
            # (bs, nq, 1, 2)
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = out_boxes[:, [torch.div(i, self.topk, rounding_mode='trunc') for i in range(self.topk*out_boxes.shape[1])], :] # [bs,num_queries*topk,2]
            topk_boxes_idx = torch.div(torch.arange(0, self.topk*out_boxes.shape[1], 1, dtype=labels.dtype, device=labels.device), self.topk, rounding_mode='trunc')[None, :].repeat(labels.shape[0], 1)
        elif self.type == "class_one":
            # choose one class that all queries are assigned this category label
            assert self.topk == 1, "so big value for class_one type, please check"
            mean_prob = torch.mean(prob,dim=1) # [bs,num_classes]
            value, idx = torch.topk(mean_prob, self.topk, dim=-1) # [bs,topk=1]
            labels = idx.repeat(1,Q) # [bs,num_queries]
            scores = torch.gather(prob,dim=2,index=idx.unsqueeze(1).repeat(1,Q,1)).squeeze(2) # [bs,num_queries,1]->[bs,num_queries]
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = out_boxes # [bs,num_queries,2]
            topk_boxes_idx = torch.arange(0, out_boxes.shape[1], 1, dtype=labels.dtype, device=labels.device)[None, :].repeat(labels.shape[0], 1)

        else:
            raise ValueError("Don't define this post process type: {self.type}")
        
        # from normalized [0, 1] to absolute [0, length] (second) coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1) # [bs,2]-> "start, end"
        topk_boxes = topk_boxes * scale_fct[:, None, :] # [bs,topk,2] transform fraction to second

        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q } for s, l, b, q in zip(scores, labels, topk_boxes, topk_boxes_idx)] # corresponding to Tad_eval.update()

        return results

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k)
                                    for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x




def build(args, device):
    if args.target_type != "none": # adopt one-hot as target, only used in close_set
        num_classes = int(args.num_classes * args.split / 100)
    else:
        num_classes = args.num_classes

    if args.feature_type == "ViFi-CLIP":
        text_encoder,logit_scale = None, torch.from_numpy(np.load(os.path.join(args.feature_path,'logit_scale.npy'))).float()
    elif args.feature_type == "CLIP":
        text_encoder, logit_scale = build_text_encoder(args,device)
    else:
        raise NotImplementedError
    
    pos_embed = build_position_encoding(args)
    transformer = build_deformable_transformer(args)


    model = DeformableDETR(
        pos_embed,
        transformer,
        text_encoder,
        logit_scale,
        device=device,
        num_classes=num_classes,
        args=args
    )

    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.actionness_loss:
        weight_dict['loss_actionness'] = args.actionness_loss_coef
    if args.distillation_loss:
        weight_dict['loss_distillation'] = args.distillation_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             args=args)
    criterion.to(device)

    postprocessor = PostProcess(args)

    return model, criterion, postprocessor


