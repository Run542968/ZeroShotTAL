import argparse
import yaml
import os

parser = argparse.ArgumentParser(description='ZSTAL')

# model basic
parser.add_argument('--cfg_path', type=str, help="the path of yaml file, there are some default configs for specific dataset")
parser.add_argument('--model_name', type=str, help="the model name to save logging")
parser.add_argument('--seed', type=int, default=3552, help='random seed (default: 1)')
parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--task', type=str, default="zero_shot", choices=('zero_shot', 'close_set'), help='[zero_shot,close_set]')
parser.add_argument('--target_type', type=str, default="prompt", choices=('none', 'prompt', 'description', 'name'), help="[none,prompt,description,name]") # NOTE: 'none' means use one-hot target that just for close_set


# dataset
## basic
parser.add_argument('--num_workers', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=16) 
parser.add_argument('--dataset_name', type=str, default="Thumos14", choices=('Thumos14', 'ActivityNet13'), help='[Thumos14,ActivityNet13]]')
parser.add_argument('--num_classes', type=int, default=20, help='total num_classes of dataset')
parser.add_argument('--split', type=int, default=75)
parser.add_argument('--split_id', type=int, default=0, help="the id of split file")
## path
parser.add_argument('--anno_file_path', type=str, default="./data/Thumos14/Thumos14_annotations.json")
parser.add_argument('--feature_info_path', type=str, default="./data/Thumos14/CLIP/Thumos14_info.json")
parser.add_argument('--description_file_path', type=str, default="./data/Thumos14/Thumos14_description.json")
parser.add_argument('--feature_path', type=str, default="/mnt/Datasets/Thumos14/CLIP_feature")
## feature info
parser.add_argument('--feature_type', default="CLIP", choices=('CLIP', 'ViFi-CLIP'), help='[CLIP,ViFi-CLIP]') 
parser.add_argument('--feature_stride', type=int, default=16, help="16 consecutive frames to form a snippet")
## for ActivityNet13
parser.add_argument('--rescale_length', type=int, default=0, help="300 for ActivityNet13, 0 denotes using origin length")
## for Thumos14
parser.add_argument('--slice_size', type=int, default=128, help="the length of slice")
parser.add_argument('--slice_overlap', type=float, default=0.75, help="the overlap of slice")
parser.add_argument('--inference_slice_overlap', type=float, default=0.25, help="the overlap of slice")



# model
parser.add_argument('--hidden_dim', type=int, default=512, help="the feat_dim on feature")
parser.add_argument('--detr_architecture', type=str, default="ConditionalDetr", choices=('ConditionalDetr', 'DeformableDetr'), help="the architecture of detr that is used")
## Backbone
parser.add_argument('--enable_backbone', action='store_true', default=False, help="whether to use TemporalConv1D as backbone")
parser.add_argument('--position_embedding', type=str, default='sine', choices=('sine', 'learned'),help="Type of positional embedding to use on top of the video features")
parser.add_argument('--backbone_layers', type=int, default=1, help="the number of the Conv1D in backbone")
## Transformer
parser.add_argument('--dropout', type=float, default=0.1, help="Dropout applied in the transformer")
parser.add_argument('--nheads', type=int, default=8, help="Number of attention heads inside the transformer's attentions")
parser.add_argument('--dim_feedforward', type=int, default=2048, help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--enc_layers', type=int, default=6, help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', type=int, default=6, help="Number of decoding layers in the transformer")
## CLIP
parser.add_argument('--CLIP_model_name', type=str, default='ViT-B/16', help="The version of different pretrain CLIP")
## Conditional DETR
parser.add_argument('--num_queries', type=int, default=15, help="Number of query slots")
parser.add_argument('--ROIalign_strategy', default="before_pred", choices=("before_pred","after_pred"), help="when to perform ROIalign, pred means compute visual-text similarity")
parser.add_argument('--ROIalign_size', type=int, default=16, help="The length of ROIalign ouput size")
parser.add_argument('--with_iterative_refine', action='store_true', default=False, help="whether to use different parameter in different decoder layer")
## Deformable DETR
parser.add_argument('--enc_n_points', type=int, default=4, help="Number of sampled points per head for deformable attention in the encoder")
parser.add_argument('--dec_n_points', type=int, default=4, help="Number of sampled points per head for deformable attention in the decoder")


## posPrior
parser.add_argument('--enable_posPrior', action='store_true', default=False)



# Loss
parser.add_argument('--aux_loss', action='store_true', default=False, help="Enable auxiliary decoding losses (loss at each layer)")
parser.add_argument('--cls_loss_coef', type=float, default=2)
parser.add_argument('--bbox_loss_coef', type=float, default=5)
parser.add_argument('--giou_loss_coef', type=float, default=2)
parser.add_argument('--focal_alpha', type=float, default=0.25)
parser.add_argument('--gamma', type=float, default=2)

parser.add_argument('--actionness_loss', action='store_true', default=False)
parser.add_argument('--actionness_loss_coef', type=float, default=2)

parser.add_argument('--distillation_loss', action='store_true', default=False)
parser.add_argument('--distillation_loss_coef', type=float, default=1)

parser.add_argument('--salient_loss', action='store_true', default=False)
parser.add_argument('--salient_loss_coef', type=float, default=1)
parser.add_argument('--salient_loss_impl', type=str, default="BCE", choices=('BCE','CE'))




# Matcher
parser.add_argument('--set_cost_class', type=float, default=2, help="Class coefficient in the matching cost")
parser.add_argument('--set_cost_bbox', type=float, default=5, help="L1 box coefficient in the matching cost")
parser.add_argument('--set_cost_giou', type=float, default=2, help="giou box coefficient in the matching cost")

# Optimizer
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr_backbone', type=float, default=1e-5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_drop', type=int, default=40, help="the step begin to drop lr")
parser.add_argument('--clip_max_norm', type=float, default=0.1, help="gradient clipping max norm")

# PostProcess
parser.add_argument('--postprocess_type', type=str, default='class_agnostic', choices=('class_agnostic', 'class_specific', 'class_one'),help="Type of post process") # NOTE: need to change the topk simultaneously
parser.add_argument('--postprocess_topk', type=int, default=100, help="The number proposals for post process, note that: class_agnostic: 100, class_specific: 1")

# Inference
parser.add_argument('--save_result', action='store_true', default=False, help="Whether to save the prediction result")
parser.add_argument('--test_interval', type=int, default=1, help="The interval to inference, -1 denotes not using this")
parser.add_argument('--proposals_weight_type', default="after_softmax", choices=("before_softmax","after_softmax"), help="the way to perform multiple between detector scores and ROIalign proposals")
parser.add_argument('--prob_type', type=str, default="softmax", choices=("softmax","sigmoid", "none_mul"), help="the strategy to get normalized probability")
parser.add_argument('--inference_entire', action='store_true', default=False, help="Whether to test entire video instead of slide window")
parser.add_argument('--filter_threshold', type=float, default=0, help="the threshold to filter negative proposals")
parser.add_argument('--enable_ensemble', action='store_true', default=False, help="Whether to ensemble the result")
parser.add_argument('--ensemble_rate', type=float, default=0.5, help="the balance coefficient between semantic head and CLIP capability")
parser.add_argument('--ensemble_strategy', type=str, default="arithmetic", choices=("arithmetic","geomethric"), help="the strategy to ensemble")
parser.add_argument('--pooling_type', type=str, default="average", choices=("average","max", "center1", "center2","self_attention","slow_fast","sparse"), help="the strategy to get normalized probability")



def merge_cfg_from_file(args,cfg_path):
    '''
    the config from yaml file is the latest one
    '''
    assert os.path.exists(cfg_path), f'cfg_path: {cfg_path} is invalid'
    cfg_from_file = yaml.load(open(cfg_path), yaml.FullLoader) # dict()
    args_dic = vars(args)
    args_dic.update(cfg_from_file)
    args = argparse.Namespace(**args_dic)

    return args

if __name__=="__main__":
    args = parser.parse_args()
    cfg_path = args.cfg_path

    import os
    cfg_path = "./config/Thumos14_CLIP.yaml"
    assert os.path.exists(cfg_path), 'cfg_path is invalid'
    cfg_from_file = yaml.load(open(cfg_path), yaml.FullLoader)


    print(cfg_from_file)
    print(type(cfg_from_file))
    print(args)
    args_dic = vars(args)
    args_dic.update(cfg_from_file)
    print(args_dic)
