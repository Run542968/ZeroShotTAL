#!/usr/bin/python3
# -*- encoding: utf-8 -*-


"""
TemporalConv1D modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
import torch.nn.init as torch_init
from typing import Dict, List

from utils.misc import NestedTensor

from .position_encoding import build_position_encoding

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        # torch_init.xavier_uniform_(m.weight)
        # import pdb
        # pdb.set_trace()
        torch_init.kaiming_uniform_(m.weight)
        if type(m.bias)!=type(None):
            m.bias.data.fill_(0)


class Backbone(nn.Module):
    """Con1D Local Relationship Modeling."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,  num_layers: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Conv1d(i, o, kernel_size=3, padding=1) for i, o in zip([input_dim] + h, h + [output_dim])])
        self.apply(weights_init)
    
    def forward(self, tensor_list: NestedTensor):
        out_tensor_list = []
        tensors, mask = tensor_list.decompose()
        x = tensors.permute(0,2,1) # [b,c,t]
        for i, layer in enumerate(self.layers):
            x = F.leaky_relu(layer(x),0.2)
            
            out_tensor_list.append(NestedTensor(x.permute(0,2,1),mask))
        # out = x.permute(0,2,1) # [b,t,c]
        # out_tensor = NestedTensor(out,mask)
        return out_tensor_list

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding, args):
        super().__init__(backbone, position_embedding)
        self.args = args
    def forward(self, tensor_list: NestedTensor):
        if self.args.enable_backbone:
            out_tensor_list = self[0](tensor_list) # list: [b,t,c]
            out: List[NestedTensor] = []
            pos = []
            for nest in out_tensor_list:
                out.append(nest)
                pos.append(self[1](nest).to(nest.tensors.dtype))
        else:
            out: List[NestedTensor] = []
            pos = []
            out.append(tensor_list)
            pos.append(self[1](tensor_list).to(tensor_list.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = Backbone(args.hidden_dim,args.hidden_dim,args.hidden_dim,args.backbone_layers)
    model = Joiner(backbone, position_embedding, args)
    model.feat_dim = backbone.output_dim
    return model
