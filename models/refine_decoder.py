import math
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class RefineDecoderV2(nn.Module):
    def __init__(self, refine_layer, num_layers) -> None:
        super().__init__()
        self.layers = num_layers
        self.layers = _get_clones(refine_layer, num_layers)

    def forward(self, query_feat, video_feat, roi_segment_feat,
                video_feat_key_padding_mask: Optional[Tensor] = None,
                video_pos: Optional[Tensor] = None,
                roi_pos: Optional[Tensor] = None):
        
        output = query_feat
        for layer in self.layers:
            output = layer(query_feat, video_feat, roi_segment_feat, video_feat_key_padding_mask, video_pos, roi_pos)
        
        return output

class RefineDecoderV2_layer(nn.Module):

    def __init__(self, nheads=4, d_model=256, args=None):
        super().__init__()
        self.d_model = d_model
        self.cross_attn_local = nn.MultiheadAttention(d_model,nheads)
        self.refine_drop_saResidual = args.refine_drop_saResidual
        self.refine_drop_sa = args.refine_drop_sa
        self.refine_fusion_type = args.refine_fusion_type
        self.refine_cat_type = args.refine_cat_type
        if "concat" in self.refine_cat_type:
            self.proj_head = nn.Linear(2*d_model,d_model)
            self.self_attn = nn.MultiheadAttention(2*d_model,nheads)
        else:
            self.self_attn = nn.MultiheadAttention(d_model,nheads)


    def forward(self, query_feat, video_feat, roi_segment_feat,
                video_feat_key_padding_mask: Optional[Tensor] = None,
                video_pos: Optional[Tensor] = None,
                roi_pos: Optional[Tensor] = None):
        '''
            query_feat: [b,n,c]
            roi_segment_feat: [b,n,l,c]
            video_feat: [b,t,c]
            video_feat_key_padding_mask: equal the mask [b,t]
            video_pos: [b,t,c]
            roi_pos: [b,n,l,c]
        '''
        if self.refine_fusion_type == "ca":
            # cross-attetion in query_embed and sement feat
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            l,n,b,dim = segment_feat.shape
            segment_feat = segment_feat.reshape(l,n*b,dim) # [l,n*b,dim]
            query_feat_seg = query_feat.reshape(1,n*b,dim) # [1,n*b,dim]

            tgt1 = self.cross_attn_local(query=query_feat_seg,
                                        key=segment_feat,
                                        value=segment_feat)[0] # [1,n*b,dim]
            tgt1 = tgt1.reshape(n,b,dim)
        elif self.refine_fusion_type == "mean":
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            tgt1 = segment_feat.mean(0) # [n,b,dim]
        elif self.refine_fusion_type == "max":
            query_feat = query_feat.permute(1,0,2) # [num_queries,b,dim]
            segment_feat = roi_segment_feat.permute(2,1,0,3) # [l,num_queries,b,dim]
            tgt1 = segment_feat.max(0)[0] # [n,b,dim]
        else:
            raise NotImplementedError



        if self.refine_drop_sa:
            query_feat = tgt1
        else:
            if "concat" in self.refine_cat_type:
                query_feat = torch.cat([query_feat, tgt1], dim=-1) # [n,b,2*dim]
            elif self.refine_cat_type == "sum":
                query_feat = query_feat + tgt1
            else:
                raise ValueError

            # self-attetnion between different query
            tgt2 = self.self_attn(query=query_feat,
                                key=query_feat,
                                value=query_feat)[0]
            if self.refine_drop_saResidual:
                if self.refine_cat_type == "concat1":
                    n,b,dim = tgt2.shape
                    query_feat = tgt2[:,:,0:self.d_model]
                elif self.refine_cat_type == "concat2":
                    query_feat = self.proj_head(query_feat) # [n,b,2*dim]->[n,b,dim]
                elif self.refine_cat_type == "sum":
                    query_feat = tgt2
            else:
                query_feat = query_feat + tgt2

        query_feat = query_feat.permute(1,0,2) # [b,n,dim]
        return query_feat
    
def build_refine_decoder(args):
    refine_layer = RefineDecoderV2_layer(nheads=args.nheads,
                           d_model=args.hidden_dim,
                           args=args)
    return RefineDecoderV2(refine_layer=refine_layer,
                           num_layers=args.refine_layer_num)