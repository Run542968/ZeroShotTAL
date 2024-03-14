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
from .transformer import build_transformer
from ..criterion import build_criterion
from ..postprocess import build_postprocess
from ..matcher import build_matcher
from ..refine_decoder import build_refine_decoder
from models.clip import build_text_encoder
from models.clip import clip as clip_pkg
import torchvision.ops.roi_align as ROIalign
import numpy as np
import os
import copy
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
    

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class ConditionalDETR(nn.Module):
    """ This is the Conditional DETR module that performs object detection """
    def __init__(self, 
                 backbone, 
                 transformer, 
                 text_encoder, 
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

        self.ROIalign_strategy = args.ROIalign_strategy
        self.ROIalign_size = args.ROIalign_size
        
        self.pooling_type = args.pooling_type

        self.with_iterative_refine = args.with_iterative_refine

        self.actionness_loss = args.actionness_loss

        self.enable_posPrior = args.enable_posPrior


        self.distillation_loss = args.distillation_loss
        self.salient_loss = args.salient_loss

        self.enable_ensemble = args.enable_ensemble
        self.ensemble_rate = args.ensemble_rate
        self.ensemble_strategy = args.ensemble_strategy

        self.enable_element = args.enable_element
        self.num_element = args.num_element
        self.element_rate = args.element_rate

        self.compact_loss = args.compact_loss

        self.enable_sdfusion = args.enable_sdfusion
        self.sdfuison_rate = args.sdfuison_rate
        self.fusion_type = args.fusion_type

        self.enable_bg = args.enable_bg

        hidden_dim = transformer.d_model



        if self.enable_posPrior:
            self.query_embed = nn.Embedding(self.num_queries,1)
            self.query_embed.weight.data[:, :1].uniform_(0, 1)
            self.query_embed.weight.data[:, :1] = inverse_sigmoid(self.query_embed.weight.data[:, :1])
            self.query_embed.weight.data[:, :1].requires_grad = False
        else:
            self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        # self.input_proj = nn.Conv1d(backbone.feat_dim, hidden_dim, kernel_size=1)
        self.input_proj = nn.ModuleList([ # following TadTR
            nn.Sequential(
                nn.Conv1d(backbone.feat_dim, hidden_dim, kernel_size=1),
                nn.GroupNorm(32, hidden_dim),
            )])
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)



        self.bbox_embed = MLP(hidden_dim, hidden_dim, 2, 3)
        # init bbox_mebed
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        


        if self.target_type != "none":
            self.class_embed = nn.Linear(hidden_dim, hidden_dim)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(hidden_dim) * bias_value
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.class_embed.bias.data = torch.ones(num_classes) * bias_value


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

        # learnable element query
        if self.enable_element:
            self.dynamic_element = nn.Embedding(self.num_element, hidden_dim)
            self.dynamic_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2)

        # compacting feature after hs
        if self.compact_loss:
            self.compact_proj = nn.Linear(hidden_dim, hidden_dim)
            # init prior_prob setting for focal loss
            prior_prob = 0.01
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.compact_proj.bias.data = torch.ones(hidden_dim) * bias_value
            
            self.compact_head = ProbObjectnessHead(hidden_dim)

        if self.enable_sdfusion:
            self.fusion_attn = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=0.2)

        if self.enable_bg:
            self.bg_embedding = nn.Embedding(1,hidden_dim)

        # following TadTR
        num_pred = transformer.decoder.num_layers
        if self.with_iterative_refine: # specific parameters for each laryer
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(
                self.bbox_embed[0].layers[-1].bias.data[1:], -2.0)
            if self.actionness_loss:
                self.actionness_embed = _get_clones(self.actionness_embed, num_pred)
            if self.compact_loss:
                self.compact_proj = _get_clones(self.compact_proj, num_pred)
                self.compact_head =  _get_clones(self.compact_head, num_pred)

        else: # shared parameters for each laryer
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[1:], -2.0)
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            if self.actionness_loss:
                self.actionness_embed = nn.ModuleList([self.actionness_embed for _ in range(num_pred)])
            if self.compact_loss:
                self.compact_proj = nn.ModuleList([self.compact_proj for _ in range(num_pred)])
                self.compact_head = nn.ModuleList([self.compact_head for _ in range(num_pred)])


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
            samples = nested_tensor_from_tensor_list(samples) # [b,t,dim]
 
        # origin CLIP features
        clip_feat, mask = samples.decompose()
        bs,t,dim = clip_feat.shape

        # backbone for temporal modeling
        feature_list, pos_list = self.backbone(samples) # list of [b,t,c], list of [b,t,c]
        feature, pos = feature_list[-1], pos_list[-1]

        # prepare text target
        if self.target_type != "none":
            with torch.no_grad():
                if self.args.feature_type == "ViFi-CLIP":
                    text_feats = torch.from_numpy(np.load(os.path.join(self.args.feature_path,'text_features_split75_splitID1.npy'))).float().to(self.device)
                elif self.args.feature_type == "CLIP":
                    text_feats = self.get_text_feats(classes_name, description_dict, self.device, self.target_type) # [N classes,dim]
                else:
                    raise NotImplementedError
            if self.enable_bg:
                text_feats = torch.cat([text_feats,self.bg_embedding.weight],dim=0) # [num_classes+1,dim]
                
        # feed into model
        src, mask = feature.decompose()
        assert mask is not None
        src = self.input_proj[0](src.permute(0,2,1)).permute(0,2,1)
        
        memory, hs, reference = self.transformer(src, mask, self.query_embed.weight, pos) # return: [enc_layers, b,t,c], [dec_layers,b,num_queries,c], [b,num_queries,1]

        # record result
        out = {}
        out['memory'] = memory
        out['hs'] = hs


        outputs_coords = []
        reference_before_sigmoid = inverse_sigmoid(reference) # [b,num_queries,1], Reference point is the predicted center point.

        for lvl in range(hs.shape[0]):
            tmp = self.bbox_embed[lvl](hs[lvl]) # [b,num_queries,2], tmp is the predicted offset value.
            tmp[..., :1] += reference_before_sigmoid # [b,num_queries,2], only the center coordination add reference point
            outputs_coord = tmp.sigmoid() # [b,num_queries,2]
            outputs_coords.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coords) # [dec_layers,b,num_queries,2]
        out['pred_boxes'] = outputs_coord[-1]


        if self.target_type != "none":
            outputs_embed = torch.stack([self.class_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
            out['class_embs'] = outputs_embed[-1]

            if self.enable_element: # fuse the dynamic element into query embedding
                l,b,n,dim = outputs_embed.shape
                dynamics = self.dynamic_element.weight # [k, dim]
                dynamics = dynamics.unsqueeze(0).repeat(l*b,1,1).permute(1,0,2) # [l*b,k,dim]->[k,l*b,dim]
                query_embed = outputs_embed.reshape(l*b,n,dim).permute(1,0,2) # [l*b,n,dim]->[n,l*b,dim]
                tgt = self.dynamic_attn(query=query_embed, key=dynamics, value=dynamics)[0] # [n,l*b,dim]
                outputs_embed = (query_embed + self.element_rate*tgt).permute(1,0,2).reshape(l,b,n,dim) # [n,l*b,dim]->[l*b,n,dim]->[l,b,n,dim]
            if self.enable_sdfusion: # fuse the clip feat into query embedding via cross-attention
                roi_feat = self._roi_align(out['pred_boxes'],clip_feat,mask,self.ROIalign_size).squeeze() # [bs,num_queries,ROIalign_size,dim]

                query_feat = outputs_embed[-1].permute(1,0,2) # [num_queries,b,dim]
                segment_feat = roi_feat.permute(2,1,0,3) # [RoIsize,num_queries,b,dim]
                RoIsize,n,b,dim = segment_feat.shape
                segment_feat = segment_feat.reshape(RoIsize,n*b,dim) # [RoIsize,n*b,dim]
                query_feat = query_feat.reshape(1,n*b,dim) # [1,n*b,dim]

                if self.fusion_type=="parameter":
                    tgt = self.fusion_attn(query=query_feat,
                                                key=segment_feat,
                                                value=segment_feat)[0] # [1,n*b,dim]
                    tgt = tgt.reshape(n,b,dim).permute(1,0,2) # [b,n,dim]
                elif self.fusion_type=="non_parameter":
                    attention_scores = torch.einsum("lbd,rbd->lbr",query_feat,segment_feat)
                    attention_scores = attention_scores / math.sqrt(dim)
                    attention_probs = F.softmax(attention_scores, dim = -1)
                    tgt = torch.einsum("lbr,rbd->lbd",attention_probs, segment_feat) # [1,n*b,dim]
                    tgt = tgt.reshape(n,b,dim).permute(1,0,2) # [b,n,dim]
                elif self.fusion_type=="average":
                    tgt = roi_feat.mean(2) # [bs,num_queries,dim]
                else:
                    raise ValueError
                outputs_embed[-1] = self.sdfuison_rate*outputs_embed[-1] + tgt
            outputs_logit = self._compute_similarity(outputs_embed, text_feats) # [dec_layers, b,num_queries,num_classes]
        else:
            outputs_logit = torch.stack([self.class_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
        out['class_logits'] = outputs_logit[-1]


        if self.actionness_loss :
            # compute the class-agnostic foreground score
            if self.compact_loss:
                # feature constrain before input it to actionness_embed head
                compact_feature = torch.stack([self.compact_proj[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,dim]
                compact_value = torch.stack([self.compact_head[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries]
                out['compact_value'] = compact_value[-1]
                actionness_logits = torch.stack([self.actionness_embed[lvl](compact_feature[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,1]
            else:
                actionness_logits = torch.stack([self.actionness_embed[lvl](hs[lvl]) for lvl in range(hs.shape[0])]) # [dec_layers,b,num_queries,1]
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
            
                salient_logits = self.salient_head(memory[-1].permute(0,2,1)).permute(0,2,1) # [b,t,1]
                out['salient_logits'] = salient_logits
        else:
            if self.enable_ensemble:
                ROIalign_logits = self._temporal_pooling(self.pooling_type, out['pred_boxes'], clip_feat, mask, self.ROIalign_size, text_feats) # [b,num_queries,num_classes]
                
                if self.ensemble_strategy == "arithmetic":
                    losgits = self.ensemble_rate*out['class_logits'] + (1-self.ensemble_rate)*ROIalign_logits
                elif self.ensemble_strategy == "geomethric":
                    losgits = torch.mul(out['class_logits'].pow(self.ensemble_rate),ROIalign_logits.pow(1-self.ensemble_rate))
                else:
                    NotImplementedError
                
                out['class_logits'] = losgits
            else:
                pass


        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(
                outputs_logit, outputs_coord)

        return out

    @torch.jit.unused
    def _set_aux_loss(self, class_logits, outputs_coord, actionness_logits=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if actionness_logits!=None:
            return [{'class_logits': a, 'pred_boxes': b, 'actionness_logits': c} for a, b, c in zip(class_logits[:-1], outputs_coord[:-1], actionness_logits[:-1])]
        else:
            return [{'class_logits': a, 'pred_boxes': b} for a, b in zip(class_logits[:-1], outputs_coord[:-1])]





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
    
    backbone = build_backbone(args)
    transformer = build_transformer(args)

    model = ConditionalDETR(
        backbone,
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
    if args.compact_loss:
        weight_dict['loss_compact'] = args.compact_loss_coef
    if args.enable_bg:
        weight_dict['loss_ce_bg'] = args.cls_loss_coef


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
