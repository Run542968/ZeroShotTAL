# ------------------------------------------------------------------------
# TadTR: End-to-end Temporal Action Detection with Transformer
# Copyright (c) 2021. Xiaolong Liu.
# ------------------------------------------------------------------------

'''build models'''

from .deformable_detr import build

def build_model(args, device):
    return build(args, device)
