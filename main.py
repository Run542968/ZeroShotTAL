#!/usr/bin/python3
# -*- encoding: utf-8 -*-
"""
@File :  main.py
@Time :  2023/09/12 15:37:33
@Author :  Jia-Run Du
@Version :  1.0
@Contact :  dujr6@mail2.sysu.edu.cn
@License :  Copyright (c) ISEE Lab
@Desc :  main file
"""


from __future__ import print_function
import os
import torch
from torch.utils.data import DataLoader
from utils.misc import collate_fn
import random
import options
from options import merge_cfg_from_file
import numpy as np
from tqdm import tqdm
from utils.util import get_logger, setup_seed, write_to_csv
import dataset
from models.ConditionalDetr import build_model as build_ConditionalDetr
from models.DeformableDetr import build_model as build_DeformableDetr
from train import train
from test import test
from tqdm import tqdm
import shutil


# Computing the parameters of the model
def count_parameters(model):
    total_params = 0
    trainable_params = 0

    for param in model.parameters():
        temp = param.numel()
        total_params += temp

        if param.requires_grad:
            trainable_params += temp
    print(f"Total parameters: {total_params/1000000} M, Trainable parameters: {trainable_params/1000000} M")
    return total_params, trainable_params 

# torch.set_default_tensor_type('torch.cuda.FloatTensor')
def check_directory(args):
    # contruct ckpt directory
    if not os.path.exists(os.path.join('./ckpt/',args.dataset_name)):
        os.makedirs(os.path.join('./ckpt/',args.dataset_name))

    # contruct logger
    if not os.path.exists(os.path.join('./logs/',args.dataset_name)):
        os.makedirs(os.path.join('./logs/',args.dataset_name))
    if os.path.exists(os.path.join('./logs/', args.dataset_name, args.model_name+'.log')):
        os.remove(os.path.join('./logs/', args.dataset_name, args.model_name+'.log'))
    logger = get_logger(os.path.join('./logs/', args.dataset_name, args.model_name+'.log'))

    # contruct results directory
    if not os.path.exists(os.path.join('./results/',args.dataset_name,args.model_name)):
        os.makedirs(os.path.join('./results/',args.dataset_name,args.model_name))
    if os.path.exists(os.path.join('./results/',args.dataset_name,args.model_name)): 
        shutil.rmtree(os.path.join('./results/',args.dataset_name,args.model_name))

    if not os.path.exists(os.path.join('./results/excel',args.dataset_name)):
        os.makedirs(os.path.join('./results/excel',args.dataset_name))
    if os.path.exists(os.path.join('./results/excel',args.dataset_name,args.model_name + "_results.csv")): 
        os.remove(os.path.join('./results/excel',args.dataset_name,args.model_name + "_results.csv"))

    return logger

if __name__ == '__main__':
    args = options.parser.parse_args()
    if args.cfg_path is not None:
        args = merge_cfg_from_file(args,args.cfg_path) # NOTE that the config comes from yaml file is the latest one.

    device = torch.device(args.device)
    seed=args.seed
    setup_seed(seed)
    logger = check_directory(args)
    logger.info('=============seed: {}, pid: {}============='.format(seed,os.getpid()))
    logger.info(args)


    # load dataset
    train_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='train', mode='train', args=args)
    val_dataset = getattr(dataset,args.dataset_name+"Dataset")(subset='inference', mode='inference', args=args)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True, shuffle=False, drop_last=False)

    # load model
    build_func = {
        'ConditionalDetr': build_ConditionalDetr,
        'DeformableDetr': build_DeformableDetr
    }
    model, criterion, postprocessor = build_func[args.detr_architecture](args, device)
    model.to(device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'number of params: {n_parameters/1000000} M')

    param_dicts = [
        # the parameters in transformaer
        {
            "params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]
         },
        # the parameters in backbone
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        }
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)


    best_stats = {}
    with torch.autograd.set_detect_anomaly(True):
        for epoch in tqdm(range(args.epochs)):
            epoch_loss_dict_scaled = train(model=model, criterion=criterion, data_loader=train_loader, optimizer=optimizer, device=device, epoch=epoch, max_norm=args.clip_max_norm)
            lr_scheduler.step()
            torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'last_' + args.model_name + '.pkl'))
            
            if epoch % args.test_interval == 0 and args.test_interval != -1:
                test_stats = test(model=model,criterion=criterion,postprocessor=postprocessor,data_loader=val_loader,dataset_name=args.dataset_name,epoch=epoch,device=device,args=args)
                logger.info('||'.join(['Intermediate map @ {} = {:.3f} '.format(test_stats['iou_range'][i],test_stats['per_iou_ap_raw'][i]*100) for i in range(len(test_stats['iou_range']))]))
                logger.info('Intermediate mAP Avg ALL: {}'.format(test_stats['mAP_raw']*100))
                logger.info('Intermediate AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(test_stats['AR@1_raw']*100, test_stats['AR@5_raw']*100, test_stats['AR@10_raw']*100, test_stats['AR@50_raw']*100,test_stats['AR@100_raw']*100,test_stats['AUC_raw']*100))
                write_to_csv(os.path.join('./results/excel',args.dataset_name,args.model_name), test_stats, epoch)

                # update best
                if test_stats['mAP_raw'] > best_stats.get('mAP_raw',0.0):
                    best_stats = test_stats
                    logger.info('new best metric {:.4f}@epoch{}'.format(best_stats['mAP_raw']*100, epoch))
                    torch.save(model.state_dict(), os.path.join('./ckpt/',args.dataset_name,'best_' + args.model_name + '.pkl'))

                logger.info('Current best metric from {:.4f}@epoch{}'.format(best_stats['mAP_raw']*100, best_stats['epoch']))

    
    iou = best_stats['iou_range']
    max_map = best_stats['per_iou_ap_raw']
    max_Avg = best_stats['mAP_raw']
    best_epoch = best_stats['epoch']
    logger.info('||'.join(['MAX map @ {} = {:.3f} '.format(iou[i],max_map[i]*100) for i in range(len(iou))]))
    logger.info('MAX mAP Avg ALL: {} in Epoch: {}'.format(max_Avg*100,best_epoch))
    # logger.info('MAX AR@1: {}, AR@5: {}, AR@10: {}, AR@50: {}, AR@100: {}, AUC: {}'.format(best_stats['AR@1_raw']*100, best_stats['AR@5_raw']*100, best_stats['AR@10_raw']*100, best_stats['AR@50_raw']*100,best_stats['AR@100_raw']*100,best_stats['AUC_raw']*100))
    logger.info('MAX AR@10: {}, AR@25: {}, AR@40: {}, AUC: {}'.format(best_stats['AR@10_raw']*100, best_stats['AR@25_raw']*100, best_stats['AR@40_raw']*100, best_stats['AUC_raw']*100))
                

