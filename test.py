import os.path as osp
import torch
import logging
from eval import tad_eval
from sklearn.metrics import accuracy_score
import time


logger = logging.getLogger()


@torch.no_grad()
def test(model, 
         criterion, 
         postprocessor, 
         data_loader, 
         dataset_name, 
         epoch,
         device, 
         args,
         verbose=None):
    model.eval()
    criterion.eval()
    postprocessor.eval()

    iou_range = [0.3, 0.4, 0.5, 0.6, 0.7] if dataset_name == 'Thumos14' else [num/100 for num in range(50, 100, 5)]
    logger.info(f'iou range {iou_range} in dataset {dataset_name}.')
    if verbose != None:
        print(f'iou range {iou_range} in dataset {dataset_name}.')

    # action_evaluator = None
    action_evaluator = getattr(tad_eval,dataset_name+"Evaluator")(dataset_name=dataset_name, epoch=epoch, dataset=data_loader.dataset, iou_range=iou_range,
                                    nms_mode=['raw'],
                                    filter_threshold=args.filter_threshold
                                    )

    count = 0
    # res_dict = {}
    for samples, targets in data_loader:
        samples = samples.to(device)

        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(samples, classes, description_dict, targets)


        count = count + len(targets)
        logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)})")
        if verbose != None:
            print(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)})")

        # logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_value:{loss_value}, loss_dict_scaled:{loss_dict_scaled}")
        # logger.info(f"Inference Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_dict_unscaled:{loss_dict_unscaled}")
        
        # post process
        video_duration = torch.FloatTensor([t["video_duration"] for t in targets]).to(device)
        results = postprocessor(outputs, video_duration)
        res = {target['video_name']: output for target, output in zip(targets, results)}
        # res_dict.update(res)

        if action_evaluator is not None:
            action_evaluator.update(res)


    # epoch_loss_dict.update({k: v/count for k, v in epoch_loss_dict.items()})
    # logger.info(f"Inference Epoch: {epoch}, epoch_loss_dict:{epoch_loss_dict}")

    # accumulate predictions from all videos
    if action_evaluator is not None:
        action_evaluator.synchronize_between_processes()
        action_evaluator.accumulate()
        # dump detections
        if args.save_result:
            save_path = osp.join('./results/',args.dataset_name,args.model_name,'detection_{}_'+str(epoch)+'_{}.json')
            action_evaluator.dump_detection(save_path)
        action_evaluator.summarize()

    # summary the all stats for logger
    stats = {}
    stats['iou_range'] = iou_range
    stats['epoch'] = epoch
    if action_evaluator is not None: # fusion the nms_mode to the name of key
        for k, v in action_evaluator.stats.items():
            for vk, vv in v.items():
                stats[vk + '_' + k] = vv

        mAP_values = ' '.join([f'{k}: {100*v:.2f}'.format(k, v) for k, v in stats.items() if k.startswith('mAP')])
        logger.info(mAP_values)

        stats['stats_summary'] = action_evaluator.stats_summary

    
    return stats

if __name__ == "__main__":
    import options
    from options import merge_cfg_from_file
    from utils.util import setup_seed
    import dataset
    from torch.utils.data import DataLoader
    from utils.misc import collate_fn
    from models.ConditionalDetr import build_model
    import os

    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)
    seed=args.seed
    setup_seed(seed)

    # load dataset
    print(f"Loading the dataset...")
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    # load model
    print(f"Building the model...")
    model, criterion, postprocessor = build_model(args,device)
    ckpt_path = os.path.join("./ckpt",args.dataset_name,"best_"+args.model_name+".pkl")
    model.load_state_dict(torch.load(ckpt_path))
    model.to(device)

    print(f"Starting the inference...")
    test_stats = test(model,criterion,postprocessor,val_loader,args.dataset_name,-1,device,args,verbose=True)
    print('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_stats['iou_range'][i],test_stats['per_iou_ap_raw'][i]*100) for i in range(len(test_stats['iou_range']))]))
    print('Intermediate mAP Avg ALL: {}'.format(test_stats['mAP_raw']*100))
    print('Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@25: {}, AR@40: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(test_stats['AR@1_raw']*100, test_stats['AR@5_raw']*100, test_stats['AR@10_raw']*100, test_stats['AR@25_raw']*100, test_stats['AR@40_raw']*100, test_stats['AR@50_raw']*100, test_stats['AR@100_raw']*100, test_stats['AUC_raw']*100))
    

# 
# CUDA_VISIBLE_DEVICES=5 python test.py --model_name "xxx" --cfg_path "./config/Thumos14_CLIP_zs_75_8frame.yaml" --batch_size 16 --lr 1e-4 --epochs 100 --postprocess_type "class_agnostic" --postprocess_topk 100 --num_queries 40 --enc_layers 2 --dec_layers 5 --enable_classAgnostic --actionness_loss_coef 3 --enable_refine --refine_drop_saResidual --split_id 1
