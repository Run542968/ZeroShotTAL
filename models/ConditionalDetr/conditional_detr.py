# ------------------------------------------------------------------------
# Conditional DETR model and criterion classes.
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn.functional as F
from torch import nn
import logging

from utils.misc import (NestedTensor, nested_tensor_from_tensor_list, inverse_sigmoid)

from .backbone import build_backbone
from .matcher import build_matcher
from .transformer import build_transformer
from .criterion import build_criterion
from .postprocess import build_postprocess
from models.clip import build_text_encoder
from .refine_decoder import build_refine_decoder
# from .semantic_head import build_semantic_head
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign
import numpy as np

logger = logging.getLogger()


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 text_encoder, 
                 refine_decoder,
                 logit_scale, 
                 device, 
                 num_classes,
                 args):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            text_encoder: text_encoder from CLIP model. See clip.__init__.py
            logit_scale: the logit_scale of CLIP. See clip.__init__.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         Conditional DETR can detect in a single image. For COCO, we recommend 100 queries.
            target_type: use one-hot or text as target. [none,prompt,description]
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            norm_embed: Just for the similarity compute of CLIP visual and text embedding, following the CLIP setting
        """
        super().__init__()
        self.backbone = backbone
        self.transformer = transformer
        self.text_encoder = text_encoder
        self.logit_scale = logit_scale
        self.device = device
        self.num_classes = num_classes
        self.args = args

        self.num_queries = args.num_queries
        self.target_type = args.target_type
        self.aux_loss = args.aux_loss
        self.norm_embed = args.norm_embed
        self.exp_logit_scale = args.exp_logit_scale
       
        self.ROIalign_strategy = args.ROIalign_strategy
        self.ROIalign_size = args.ROIalign_size
        
        self.results_ensemble = args.results_ensemble
        self.ensemble_rate = args.ensemble_rate
        self.ensemble_strategy = args.ensemble_strategy
        self.pooling_type = args.pooling_type

        self.eval_proposal = args.eval_proposal 

        self.actionness_loss = args.actionness_loss
        self.enable_classAgnostic = args.enable_classAgnostic

        self.augment_prompt_type = args.augment_prompt_type
        self.subaction_version = args.subaction_version


        self.enable_refine = args.enable_refine
        self.enable_posPrior = args.enable_posPrior

        self.enable_injection = args.enable_injection

        self.salient_loss = args.salient_loss
        self.salient_loss_type = args.salient_loss_type
        self.salient_oic_delta = args.salient_oic_delta

        self.adapterCLS_loss = args.adapterCLS_loss
        self.adapterCLS_type = args.adapterCLS_type
        self.adapterCLS_conv_weight_type = args.adapterCLS_conv_weight_type


        hidden_dim = transformer.d_model
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        
        if self.enable_refine or self.enable_injection:
            self.refine_decoder = refine_decoder

        if self.enable_posPrior:
            self.query_embed = nn.Embedding(self.num_queries,1)
            self.query_embed.weight.data[:, :1].uniform_(0, 1)
            self.query_embed.weight.data[:, :1] = inverse_sigmoid(self.query_embed.weight.data[:, :1])
            self.query_embed.weight.data[:, :1].requires_grad = False
        else:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        
        self.input_proj = nn.Conv1d(backbone.feat_dim, hidden_dim, kernel_size=1)
        
        if self.target_type != "none":
            self.class_embed = nn.Linear(hidden_dim, hidden_dim)
        else:
            if not self.enable_classAgnostic or not self.eval_proposal:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
                # init prior_prob setting for focal loss
                prior_prob = 0.01
                bias_value = -math.log((1 - prior_prob) / prior_prob)
                self.class_embed.bias.data = torch.ones(num_classes) * bias_value

        if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
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

        if self.adapterCLS_loss:
            if self.adapterCLS_type == "sa":
                self.adapter = nn.MultiheadAttention(embed_dim=hidden_dim,num_heads=4)
            elif self.adapterCLS_type == "conv_avg":
                self.adapter = nn.Conv1d(in_channels=hidden_dim,out_channels=hidden_dim,kernel_size=3,padding=1)
            elif self.adapterCLS_type == "conv_add":
                self.adapter = nn.Conv1d(in_channels=hidden_dim,out_channels=1,kernel_size=3,padding=1)
            elif self.adapterCLS_type == "conv_weight":
                self.adapter = nn.Sequential(
                    nn.Conv1d(in_channels=hidden_dim,out_channels=1,kernel_size=3,padding=1),
                    # nn.LeakyReLU(0.2),
                    # nn.Conv1d(in_channels=hidden_dim,out_channels=1,kernel_size=3,padding=1)
                )
            else:
                raise ValueError

        if self.target_type != "none":
            logger.info(f"The target_type is {self.target_type}, using text embedding as target, on task: {args.task}!")
        else:
            logger.info(f"The target_type is {self.target_type}, using one-hot coding as target, must in close_set!")

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
    
    def augment_text_feats(self,cl_names, description_dict, device, augment_type, subaction_version):
        '''
        v1: only adopt the sub-action 
        v2: class_name + sub-action 
        v3: class_name prompt + sub-action 
        '''
        text_feats = []
        for c in cl_names:
            if subaction_version == "v1":
                subaction_list = []
            elif subaction_version == "v2":
                subaction_list = [c]
            elif subaction_version == "v3":
                subaction_list = ["a video of a person doing"+" "+c]
            else: 
                raise ValueError(f"Don't have this version: {subaction_version}")
            subaction_list = subaction_list + description_dict[c]['Elaboration']['Description'] # [sub-action num, length]
            subaction_tokens = clip_pkg.tokenize(subaction_list).long().to(device)
            subaction_feats = self.text_encoder(subaction_tokens).float() # [sub-action num, dim]
            if augment_type == "average":
                pooled_feats = subaction_feats.mean(dim=0,keepdim=True) # [1,dim]
                text_feats.append(pooled_feats)
            elif augment_type == "attention": 
                text_feats.append(subaction_feats) # [sub-action num, dim]
            else:
                raise NotImplementedError
        
        if augment_type == "average":
            text_feats = torch.cat(text_feats,dim=0) # [num_classes,dim]
            return text_feats
        elif augment_type == "attention":
            nested_text_feats = nested_tensor_from_tensor_list(text_feats) # tensors: [num_classes,unify length,dim] mask: [num_classes,unify length]
            return nested_text_feats
        else:
            raise NotImplementedError

    def _text_feats(self,text_feats, augment_prompt_type, semantic_text_head):
        if augment_prompt_type == "attention":
            text_feats_tensors, text_feats_mask = text_feats.decompose()
            text_feats = semantic_text_head(text_feats_tensors,src_key_padding_mask=text_feats_mask)[-1] # [layers, num_classes,padding length,dim]->[num_classes,padding length,dim]
            # text_feats = text_feats.sum(dim=1)/torch.sum(~text_feats_mask,dim=1,keepdim=True) # [num_classes,dim]
            text_feats = text_feats[:,0,:] # [num_classes,dim] get the firt position token, i.e., class_name token
        else:
            text_feats = text_feats # [num_classes,dim]

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

    def _get_roi_prediction_v1(self,samples,coord,text_feats,ROIalign_size):
        '''
        _get_roi_prediction_v1: ROIalign before compute similarity, this strategy is easy to avoid the padded zero value

        samples: NestedTensor
        coord: [B,Q,2]
        '''
        origin_feat, mask = samples.decompose() # [B,T,dim] [B,T]
        roi_feat = self._roi_align(rois=coord,origin_feat=origin_feat,mask=mask,ROIalign_size=ROIalign_size).mean(-2) # [B,Q,dim]

        roi_logits = self._compute_similarity(roi_feat,text_feats) # [b,Q,num_classes]
        return roi_logits
    
    def _get_roi_prediction_v2(self,snippet_logits,mask,coord,ROIalign_size):
        '''
        _get_roi_prediction_v2: ROIalign before compute similarity, this strategy is more reasonable to utilize the CLIP semantic

        snippet_logits: [B,T,num_classes]
        coord: [B,Q,2]
        '''

        roi_logits = self._roi_align(rois=coord,origin_feat=snippet_logits,ROIalign_size=ROIalign_size,mask=mask).mean(-2) # [B,Q,num_classes]

        return roi_logits
    
    @torch.no_grad()
    def _compute_similarity(self, visual_feats, text_feats):
        '''
        text_feats: [num_classes,dim]
        '''
        if len(visual_feats.shape)==2: # batch_num_instance,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bd,cd->bc",logit_scale*visual_feats,text_feats)
            else:
                logits = torch.einsum("bd,cd->bc",visual_feats,text_feats)
            return logits
        elif len(visual_feats.shape)==3:# batch,num_queries/snippet_length,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqd,cd->bqc",logit_scale*visual_feats,text_feats)
            else:
                logits = torch.einsum("bqd,cd->bqc",visual_feats,text_feats)
            return logits
        elif len(visual_feats.shape)==4:# batch,num_queries,snippet_length,dim
            if self.norm_embed:
                visual_feats = visual_feats / visual_feats.norm(dim=-1,keepdim=True)
                text_feats = text_feats / text_feats.norm(dim=-1,keepdim=True)
                if self.exp_logit_scale:
                    logit_scale = self.logit_scale.exp()
                else:
                    logit_scale = self.logit_scale
                logits = torch.einsum("bqld,cd->bqlc",logit_scale*visual_feats,text_feats)
            else:
                logits = torch.einsum("bqld,cd->bqlc",visual_feats,text_feats)
            return logits
        
        else:
            raise NotImplementedError

    def _adapter_forward(self, roi_feat, roi_pos=None):
        '''
        roi_feat: [b,q,l,dim]
        '''
        def with_pos_embed(tensor, pos=None):
            return tensor if pos is None else tensor + pos


        b,q,l,d = roi_feat.shape
        if self.adapterCLS_type == "sa":
            roi_feat = roi_feat.permute(2,0,1,3)
            roi_feat = roi_feat.reshape(l,b*q,d)
            roi_pos = roi_pos.permute(2,0,1,3)
            roi_pos = roi_pos.reshape(l,b*q,d)
            query = key = with_pos_embed(roi_feat,roi_pos)
            value = roi_feat
            adapted_roi_feat = self.adapter(query=query,
                                            key=key,
                                            value=value
                                            )[0] # [l,b*q,dim]
            adapted_roi_feat = adapted_roi_feat.mean(0) # [b*q,dim]
            adapted_roi_feat = adapted_roi_feat.reshape(b,q,d) # [b,q,dim]
            roi_feat_weight = None
        elif self.adapterCLS_type == "conv_avg":
            roi_feat = roi_feat.reshape(b*q,l,d) # [b*q,l,d]
            roi_feat = roi_feat.permute(0,2,1) # [b*q,dim,l]
            adapted_roi_feat = self.adapter(roi_feat).permute(0,2,1) # [b*q,l,dim]
            adapted_roi_feat = adapted_roi_feat.reshape(b,q,l,d)
            adapted_roi_feat = adapted_roi_feat.mean(2) # [b,q,dim]

            roi_feat_weight= None
        elif self.adapterCLS_type == "conv_add":
            roi_feat_re = roi_feat.reshape(b*q,l,d) # [b*q,l,d]
            roi_feat_weight = self.adapter(roi_feat_re.permute(0,2,1)).permute(0,2,1) # [b*q,l]
            roi_feat_weight = roi_feat_weight.reshape(b,q,l,1)
            roi_feat_weight = roi_feat_weight.sigmoid()
            adapted_roi_feat = (roi_feat * roi_feat_weight).sum(2) # [b,q,dim]
        elif self.adapterCLS_type == "conv_weight":
            roi_feat_re = roi_feat.reshape(b*q,l,d) # [b*q,l,d]
            roi_feat_weight = self.adapter(roi_feat_re.permute(0,2,1)).permute(0,2,1) # [b*q,l]
            roi_feat_weight = roi_feat_weight.reshape(b,q,l,1)
            if self.adapterCLS_conv_weight_type == "kl":
                roi_feat_weight_ = (roi_feat_weight+1e-4).softmax(dim=2)
            elif self.adapterCLS_conv_weight_type == "l1":
                roi_feat_weight_ = (roi_feat_weight+1e-4).sigmoid()
            else:
                raise ValueError
            adapted_roi_feat = (roi_feat * roi_feat_weight_).sum(2) # [b,q,dim]
        else:
            raise ValueError
        
        return adapted_roi_feat, roi_feat_weight
       

    def forward(self, samples: NestedTensor, classes_name, description_dict, targets, epoch):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched video, of shape [batch_size x T x C]
               - samples.mask: a binary mask of shape [batch_size x T], containing 1 (i.e., true) on padded snippet
            classes_name: the class name of involved category
            description_dict: the dict of description file
            targets: the targets that contain gt

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x num_classes]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center, width). These values are normalized in [0, 1],
                               relative to the size of each individual video (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)

        # origin CLIP features
        clip_feat, mask = samples.decompose()
        bs,t,dim = clip_feat.shape

        # backbone for temporal modeling
        feature_list, pos = self.backbone(samples) # list of [b,t,c], list of [b,t,c]

        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                if self.augment_prompt_type == "single":
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]
                elif self.augment_prompt_type == "average":
                    text_feats = self.augment_text_feats(classes_name, description_dict, self.device, "average", self.subaction_version) # [N classes,dim]
                elif self.augment_prompt_type == "attention":
                    text_feats = self.augment_text_feats(classes_name, description_dict, self.device, "attention", self.subaction_version) # nested_text_feats, tensors: [num_classes,unify length,dim] mask: [num_classes,unify length]
                else:
                    raise NotImplementedError
                
        # feed into model
        src, mask = feature_list[-1].decompose()
        assert mask is not None
        src = self.input_proj(src.permute(0,2,1)).permute(0,2,1)
        
        if self.enable_injection:
            memory, hs, reference = self.transformer(src, mask, self.query_embed.weight, pos[-1], clip_feat, self.bbox_embed, self.refine_decoder) # [enc_layers, b,t,c], [dec_layers,b,num_queries,c], [b,num_queries,1]
        else:
            memory, hs, reference = self.transformer(src, mask, self.query_embed.weight, pos[-1]) # [enc_layers, b,t,c], [dec_layers,b,num_queries,c], [b,num_queries,1]

        # record result
        out = {}
        out['memory'] = memory
        out['hs'] = hs

        
        # generate the salient gt
        if self.salient_loss:
            if self.training: # only generate gt in training phase
                salient_gt = torch.zeros((bs,t),device=self.device) # [bs,t]
                if self.salient_loss_type == "oic":
                    salient_loss_mask = torch.ones_like(mask) # [bs,t]
                else:
                    salient_loss_mask = mask.clone() # [bs,t]
                cam = self._compute_similarity(clip_feat,text_feats) # [b,t,num_classes]
                cam_softmax = cam.softmax(dim=-1)
                
                for i, tgt in enumerate(targets):
                    cam_i = cam_softmax[i] # [t,num_classes]
                    semantic_labels = tgt['semantic_labels'] # [num_tgt]
                    salient_mask = tgt['salient_mask'] # [num_tgt,T]

                    # padding the salient mask
                    num_to_pad = t - salient_mask.shape[1]
                    if num_to_pad > 0:
                        padding = torch.ones((salient_mask.shape[0], num_to_pad), dtype=torch.bool, device=salient_mask.device)
                        salient_mask = torch.cat((salient_mask, padding), dim=1)

                    for salient_mask_j,semantic_label in zip(salient_mask,semantic_labels):
                        un_salient_mask = ~salient_mask_j.unsqueeze(dim=1) # [T,1]
                        masked_cam = cam_i*un_salient_mask # [t,num_classes]
                        masked_cam_class = masked_cam[:,semantic_label] # [T]
                        masked_cam_class_Tsoftmax = masked_cam_class.softmax(dim=0)
                        max_idx = masked_cam_class_Tsoftmax.max(dim=0)[1]
                        if self.salient_loss_type == "all" or self.salient_loss_type == "oic":
                            salient_gt[i,:] = (salient_gt[i,:] + (~salient_mask_j).float()).clamp(0,1)
                        else:
                            salient_gt[i,max_idx] = 1

                        if self.salient_loss_type == "point":
                            salient_loss_mask[i,:] = salient_loss_mask[i,:] | (~salient_mask_j)
                            salient_loss_mask[i,max_idx] = False
                        elif self.salient_loss_type == "oic":
                            false_indices = torch.where(salient_mask_j == False)[0]
                            start_index = false_indices[0]
                            end_index = false_indices[-1]
                            false_len = end_index - start_index + 1
                            expansion = int((end_index - start_index + 1) * self.salient_oic_delta)
                            inflated_start = max(0,start_index - expansion)
                            inflated_end = min(len(salient_mask_j)-1, end_index + expansion)

                            expanded_tensor = torch.clone(salient_mask_j)
                            expanded_tensor[inflated_start:inflated_end + 1] = False
                            salient_loss_mask[i,:] = salient_loss_mask[i,:] & expanded_tensor

                out['salient_gt'] = salient_gt
                out['salient_loss_mask'] = salient_loss_mask
            
            salient_logits = self.salient_head(memory[-1].permute(0,2,1)).permute(0,2,1) # [b,t,1]
            out['salient_logits'] = salient_logits
        

        # refine encoder
        if self.enable_refine:
            with torch.no_grad():
                reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.
                tmp = self.bbox_embed(hs[-1]) # [b,num_queries,2], tmp is the predicted offset value.
                tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
                outputs_coord = tmp.sigmoid() # [b,num_queries,2]
                roi_pos = self._roi_align(outputs_coord,pos[-1],mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                roi_feat = self._roi_align(outputs_coord,clip_feat,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
    
            b,q,l,d = roi_feat.shape
            refine_hs = self.refine_decoder(hs[-1],clip_feat,roi_feat,
                                    video_feat_key_padding_mask=mask,
                                    video_pos=pos[-1],
                                    roi_pos=roi_pos)

            refine_hs = hs[-1] + refine_hs
            reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.
            tmp = self.bbox_embed(refine_hs) # [b,num_queries,2], tmp is the predicted offset value.
            tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
            outputs_coord = tmp.sigmoid() # [b,num_queries,2]
            out['pred_boxes'] = outputs_coord

            if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
                # compute the class-agnostic foreground score
                actionness_logits = self.actionness_embed(refine_hs) # [b,num_queries,2]
                out['actionness_logits'] = actionness_logits

        else:
            reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.
            outputs_coords = []
            for lvl in range(hs.shape[0]):
                tmp = self.bbox_embed(hs[lvl]) # [b,num_queries,2], tmp is the predicted offset value.
                tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
                outputs_coord = tmp.sigmoid() # [b,num_queries,2]
                outputs_coords.append(outputs_coord)
            outputs_coord = torch.stack(outputs_coords) # [dec_layers,b,num_queries,2]
            out['pred_boxes'] = outputs_coord[-1]

            if self.actionness_loss or self.eval_proposal or self.enable_classAgnostic:
                # compute the class-agnostic foreground score
                actionness_logits = self.actionness_embed(hs)[-1] # [dec_layers,b,num_queries,1]->[b,num_queries,2]
                out['actionness_logits'] = actionness_logits
        
        # adopt adapterCLS_loss
        if self.adapterCLS_loss:
            with torch.no_grad():
                adapter_roi_pos = self._roi_align(out['pred_boxes'],pos[-1],mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                adapter_roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]

            adapted_roi_feat,roi_feat_weight = self._adapter_forward(adapter_roi_feat, adapter_roi_pos) # [b,q,dim], [b,q,l,1]
            adapted_roi_feat_logits = self._compute_similarity(adapted_roi_feat,text_feats) # [b,q,c]
            out['adapted_roi_feat_logits'] = adapted_roi_feat_logits
            if self.adapterCLS_type == "conv_weight":
                out['roi_feat_weight'] = roi_feat_weight # [b,q,l,1]
                out['roi_feat_logits'] = self._compute_similarity(adapter_roi_feat,text_feats) # [b,q,l,c]


        if self.target_type != "none":
           
            if not self.eval_proposal and not self.enable_classAgnostic:
                class_emb = self.class_embed(hs)[-1] # [dec_layers,b,num_queries,dim]->[b,num_queries,dim]
                class_logits = self._compute_similarity(class_emb,text_feats) # [b,num_queries,num_classes]
                out['class_logits'] = class_logits


            ## compute the classification accuate of CLIP
            # prepare instance coordination
            # gt_roi_feat = [] 
            # gt_labels = []
            # for i, t in enumerate(targets):
            #     if len(t['segments']) > 0 :
            #         gt_coordinations = t['segments'].unsqueeze(0) # [1,num_instance,2]->"center,width"
            #         visual_feat_i = clip_feat[i].unsqueeze(0) # [1,T,dim]
            #         mask_i = mask[i].unsqueeze(0) # [1,T]
            #         roi_feat = self._roi_align(gt_coordinations,visual_feat_i,mask_i,self.ROIalign_size).squeeze(dim=0) # [1,num_instance,ROIalign_size,dim]->[num_instance,ROIalign_size,dim]
            #         gt_roi_feat.append(roi_feat)
            #         gt_lbl = t['semantic_labels'] # [num]
            #         gt_labels.append(gt_lbl)
            # if len(gt_labels) > 0:
            #     gt_roi_feat = torch.cat(gt_roi_feat,dim=0) # [batch_instance_num,ROIalign_size,dim]
            #     gt_roi_feat = gt_roi_feat.mean(dim=1) # [batch_instance_num,dim]
            #     gt_labels = torch.cat(gt_labels,dim=0) # [batch_instance_num]

            #     gt_logits = self._compute_similarity(gt_roi_feat,text_feats) # [batch_instance_num,num_classes]
                
            #     out['gt_labels'] = gt_labels
            #     out['gt_logits'] = gt_logits
            ## compute the classification accuate of CLIP


            # obtain the ROIalign logits
            if not self.training: # only in inference stage

                if self.results_ensemble and not self.enable_classAgnostic:
                    fixed_text_feats = self.get_text_feats(classes_name, description_dict, self.device, "prompt") # [N classes,dim]
                    
                    if self.ROIalign_strategy == "before_pred":
                        ROIalign_logits = self._get_roi_prediction_v1(samples,out['pred_boxes'],fixed_text_feats,self.ROIalign_size)
                    else:
                        visual_feats = clip_feat + 1e-8 # avoid the NaN [B,T,dim]
                        snippet_logits = self._compute_similarity(visual_feats,fixed_text_feats) # [b,T,num_classes]
                        ROIalign_logits = self._get_roi_prediction_v2(snippet_logits,mask,out['pred_boxes'],self.ROIalign_size) # this operation must cooperate with segmenatation_loss, [b,num_queries,num_classes]
                    
                    semantic_logits = out['class_logits']

                    if self.ensemble_strategy == "arithmetic":
                        prob = self.ensemble_rate*semantic_logits + (1-self.ensemble_rate)*ROIalign_logits
                    elif self.ensemble_strategy == "geomethric":
                        prob = torch.mul(semantic_logits.pow(self.ensemble_rate),ROIalign_logits.pow(1-self.ensemble_rate))
                    else:
                        NotImplementedError
                    out['class_logits'] = prob

                elif self.enable_classAgnostic:
                    fixed_text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]

                    if self.pooling_type == "average" and not self.adapterCLS_loss:
                        if self.ROIalign_strategy == "before_pred":
                            ROIalign_logits = self._get_roi_prediction_v1(samples,out['pred_boxes'],fixed_text_feats,self.ROIalign_size)
                        else:
                            visual_feats = clip_feat + 1e-8 # avoid the NaN [B,T,dim]
                            snippet_logits = self._compute_similarity(visual_feats,fixed_text_feats) # [b,T,num_classes]
                            ROIalign_logits = self._get_roi_prediction_v2(snippet_logits,mask,out['pred_boxes'],self.ROIalign_size) # this operation must cooperate with segmenatation_loss, [b,num_queries,num_classes]
                    elif self.pooling_type == "max" and not self.adapterCLS_loss:
                        roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                        roi_feat = roi_feat.max(dim=2)[0] # [bs,num_queries,dim]

                        ROIalign_logits = self._compute_similarity(roi_feat,fixed_text_feats)
                    elif self.pooling_type == "center1" and not self.adapterCLS_loss:
                        roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                        center_idx = int(roi_feat.shape[2] / 2)
                        roi_feat = roi_feat[:,:,center_idx,:] 
                        ROIalign_logits = self._compute_similarity(roi_feat,fixed_text_feats)
                    elif self.pooling_type == "center2" and not self.adapterCLS_loss:
                        rois = out['pred_boxes'] # [b,n,2]
                        rois_center = rois[:, :, 0:1] # [B,N,1]
                        # rois_size = rois[:, :, 1:2] * scale_factor # [B,N,1]
                        truely_length = t-torch.sum(mask,dim=1) # [B]
                        truely_length = truely_length.reshape(-1,1,1) # [B,1,1]
                        center_idx = (rois_center*truely_length).long() # [b,n,1]
                        roi_feat = torch.gather(clip_feat + 1e-4, dim=1, index=center_idx.expand(-1, -1, clip_feat.shape[-1]))
                        ROIalign_logits = self._compute_similarity(roi_feat,fixed_text_feats)
                    elif self.pooling_type == "self_attention" and not self.adapterCLS_loss:
                        roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                        attention_weights = F.softmax(torch.matmul(roi_feat, roi_feat.transpose(-2, -1)), dim=-1)
                        roi_feat_sa = torch.matmul(attention_weights, roi_feat)
                        roi_feat_sa = roi_feat_sa.mean(2)
                        ROIalign_logits = self._compute_similarity(roi_feat_sa,fixed_text_feats)
                    elif self.pooling_type == "slow_fast" and not self.adapterCLS_loss:
                        roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                        fast_feat = roi_feat.mean(dim=2) # [b,q,d]
                        step = int(self.ROIalign_size // 4)
                        slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
                        roi_feat_final = (fast_feat + slow_feat)/2
                        ROIalign_logits = self._compute_similarity(roi_feat_final,fixed_text_feats)
                    elif self.pooling_type == "sparse" and not self.adapterCLS_loss:
                        roi_feat = self._roi_align(out['pred_boxes'],clip_feat + 1e-4,mask,self.ROIalign_size) # [bs,num_queries,ROIalign_size,dim]
                        step = int(self.ROIalign_size // 4)
                        slow_feat = roi_feat[:,:,::step,:].mean(dim=2) # [b,q,d]
                        ROIalign_logits = self._compute_similarity(slow_feat,fixed_text_feats)
                    elif self.adapterCLS_loss:
                        ROIalign_logits = out['adapted_roi_feat_logits']
                    else:
                        raise ValueError
                    out['class_logits'] = ROIalign_logits 


        else:
            detector_logits = self.class_embed(hs) # [dec_layers,b,num_queries,num_classes]
            out['pred_logits'] = detector_logits[-1]

        # aux_loss
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(detector_logits, outputs_coord)
        
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]





def build(args, device):
    if args.target_type != "none": # adopt one-hot as target, only used in close_set
        num_classes = int(args.num_classes * args.split / 100)
    else:
        num_classes = args.num_classes

    text_encoder, logit_scale = build_text_encoder(args,device)
    backbone = build_backbone(args)
    transformer = build_transformer(args)
    # semantic_vhead,semantic_thead = build_semantic_head(args)
    if args.enable_refine or args.enable_injection:
        refine_decoder = build_refine_decoder(args)
    else:
        refine_decoder = None

    model = ConditionalDETR(
        backbone,
        transformer,
        text_encoder,
        refine_decoder,
        logit_scale,
        device=device,
        num_classes=num_classes,
        args=args
    )
    matcher = build_matcher(args)

    weight_dict = {'loss_ce': args.cls_loss_coef, 'loss_bbox': args.bbox_loss_coef}
    weight_dict['loss_giou'] = args.giou_loss_coef
    
    if args.actionness_loss or args.eval_proposal or args.enable_classAgnostic:
        weight_dict['loss_actionness'] = args.actionness_loss_coef
    if args.rank_loss and args.enable_relaxGT:
        weight_dict['loss_rank'] = args.rank_loss_coef
        print(f"Warning!!!!! Please tuning the num_queries when you adopt rank loss!!!!!!!!!!!")
    if args.salient_loss:
        weight_dict['loss_salient'] = args.salient_loss_coef
    if args.adapterCLS_loss:
        weight_dict['loss_adapterCLS'] = args.adapterCLS_loss_coef


    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    criterion = build_criterion(args, num_classes, matcher=matcher, weight_dict=weight_dict)
    criterion.to(device)

    postprocessor = build_postprocess(args)

    return model, criterion, postprocessor
