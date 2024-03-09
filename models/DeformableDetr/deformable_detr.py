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

from .position_encoding import build_position_encoding
from .transformer import build_deformable_transformer
from models.clip import build_text_encoder
from ..criterion import build_criterion
from ..postprocess import build_postprocess
from ..matcher import build_matcher
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign
import json
import numpy as np
import os

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class ProbObjectnessHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.flatten = nn.Flatten(0,1)
        self.objectness_bn = nn.BatchNorm1d(hidden_dim, affine=False)

    def freeze_prob_model(self):
        self.objectness_bn.eval()
        
    def forward(self, x):
        out=self.flatten(x)
        out=self.objectness_bn(out).unflatten(0, x.shape[:2])
        return out.norm(dim=-1)**2
    


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
        self.salient_loss = args.salient_loss

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

        if self.salient_loss:
            self.salient_head = nn.Sequential(
                nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv1d(hidden_dim, 1, kernel_size=1)
            )


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

    def forward(self, samples: NestedTensor, classes_name, description_dict, targets):
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
        bs, t, dim = clip_feat.shape
 
        srcs = [self.input_proj[0](clip_feat.permute(0,2,1))] # low-level Conv1d  [b,dim,t]
        masks = [mask]

        query_embeds = self.query_embed.weight # [n,2*dim]

        hs, init_reference, inter_references, memory = self.transformer(srcs, masks, pos, query_embeds) # hs: [dec_layer, bs, num_queries, dim], init_reference: [bs, num_queries,1], inter_references: [dec_layer, bs, num_queries, 1], memory: [b,dim,t] 

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

            if self.salient_loss:
                salient_gt = torch.zeros((bs,t),device=self.device) # [bs,t]
                salient_loss_mask = mask.clone() # [bs,t]

                for i, tgt in enumerate(targets):
                    salient_mask = tgt['salient_mask'] # [num_tgt,T]
                    # padding the salient mask
                    num_to_pad = t - salient_mask.shape[1]
                    if num_to_pad > 0:
                        padding = torch.ones((salient_mask.shape[0], num_to_pad), dtype=torch.bool, device=salient_mask.device)
                        salient_mask = torch.cat((salient_mask, padding), dim=1)

                    for salient_mask_j in salient_mask:
                        salient_gt[i,:] = (salient_gt[i,:] + (~salient_mask_j).float()).clamp(0,1)


                out['salient_gt'] = salient_gt
                out['salient_loss_mask'] = salient_loss_mask
            
                salient_logits = self.salient_head(memory).permute(0,2,1) # [b,t,1]
                out['salient_logits'] = salient_logits



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
    
    matcher = build_matcher(args)

 
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor


