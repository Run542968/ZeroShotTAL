# TadTR: End-to-end Temporal Action Detection with Transformer

import json
import os.path as osp
import os
import pandas as pd
import numpy as np
import logging
import concurrent.futures
import sys
from .eval_detection import compute_average_precision_detection,average_recall_vs_avg_nr_proposals
from scipy.interpolate import interp1d
from utils.misc import all_gather
from utils.segment_ops import soft_nms, temporal_nms
import logging

logger = logging.getLogger()


def eval_ap(iou, cls, gt, predition):
    ap = compute_average_precision_detection(gt, predition, iou)
    sys.stdout.flush()
    return cls, ap


def apply_nms(dets_arr, nms_thr=0.4, use_soft_nms=False):
    # the last column are class ids
    unique_classes = np.unique(dets_arr[:, 3])
    output_dets = []
    for cls in unique_classes:
        this_cls_dets = dets_arr[dets_arr[:,3] == cls]
        if not use_soft_nms:
            this_cls_dets_kept = temporal_nms(this_cls_dets, nms_thr)
        else:
            classes = this_cls_dets[:, [3]]
            this_cls_dets_kept = soft_nms(this_cls_dets, 0.8, 0, 0, 100)
            this_cls_dets_kept = np.concatenate((this_cls_dets_kept, classes), -1)
        output_dets.append(this_cls_dets_kept)
    output_dets = np.concatenate(output_dets, axis=0)
    sort_idx = output_dets[:, 2].argsort()[::-1]
    output_dets = output_dets[sort_idx, :]
    return output_dets


class TADEvaluator(object):
    def __init__(self,dataset_name, subset, epoch, iou_range, classes, 
                 nms_mode=['raw'], 
                 num_workers = 8,
                 filter_threshold = 0
                 ):
        '''
        dataset_name:  thumos14, activitynet
        subset: val or test
        anno_file: the path to load anno file
        video_dict: the dataset dict created in video_dataset.py
        iou_range: [0.3:0.7:0.1] for thumos14; [0.5:0.95:0.05] for anet and hacs.
        task_setting: what setting the model for, ['close_set','zero_shot']
        split: how to split the classes in zero-shot setting
        filter_threshold: the threshold to filter negative proposals
        '''
        self.dataset_name = dataset_name
        self.subset = subset
        self.iou_range = iou_range
        self.classes = classes
        self.num_classes = len(self.classes)
        self.nms_mode = nms_mode
        self.epoch = epoch
        self.filter_threshold = filter_threshold


        # obtain gt dataframe
        all_gt = self.import_gt()

        all_gt = pd.DataFrame(all_gt, columns=["video-id", "cls","t-start", "t-end"])
        self.video_ids = all_gt['video-id'].unique().tolist()
        logger.info(f"Eval: {self.num_classes} classes that needed to evaluate in subset:{self.subset}, the classes_name are:{self.classes}")
        logger.info(f'Eval: {len(all_gt)} ground truth instances from {len(self.video_ids)} videos in subset {self.subset} of {self.dataset_name}')

        # per class ground truth
        gt_by_cls = []
        for cls in range(self.num_classes):
            gt_by_cls.append(all_gt[all_gt.cls == cls].reset_index(drop=True).drop(labels=['cls'],axis=1))
        self.gt_by_cls = gt_by_cls

        self.all_pred = {k: [] for k in self.nms_mode}
        self.all_gt = all_gt
        self.num_workers = num_workers
        self.stats = {k: dict() for k in self.nms_mode}
    

    def format_arr(self, arr, format='{:.2f}'):
        line = ' '.join([format.format(x) for x in arr])
        return line

    def synchronize_between_processes(self):
        for nms_mode in self.nms_mode:
            logger.info(f"Eval: the num of pred proposal:{len(self.all_pred[nms_mode])} in nms_mode {nms_mode}, the num of pred video: {len({x[0] for x in self.all_pred[nms_mode]})}") 
        self.all_pred = merge_distributed(self.all_pred)


    def compute_map(self, nms_mode):
        '''Compute mean average precision'''

        gt_by_cls, pred_by_cls = self.gt_by_cls, self.pred_by_cls[nms_mode]

        iou_range = self.iou_range
        num_classes = self.num_classes
        ap_values = np.zeros((num_classes, len(iou_range)))

        with concurrent.futures.ProcessPoolExecutor(min(self.num_workers, 8)) as p:
            futures = []
            for cls in range(len(pred_by_cls)):
                if len(gt_by_cls[cls]) == 0:
                    logger.info('no gt for class {}'.format(self.classes[cls]))
                if len(pred_by_cls[cls]) == 0:
                    logger.info('no prediction for class {}'.format(self.classes[cls]))
                futures.append(p.submit(eval_ap, iou_range, cls, gt_by_cls[cls], pred_by_cls[cls]))
            for f in concurrent.futures.as_completed(futures):
                x = f.result()
                ap_values[x[0], :] = x[1]

        per_iou_ap = ap_values.mean(axis=0)
        per_cls_ap = ap_values.mean(axis=1)
        mAP = per_cls_ap.mean()
       
        self.stats[nms_mode]['mAP'] = mAP
        self.stats[nms_mode]['ap_values'] = ap_values
        self.stats[nms_mode]['per_iou_ap'] = per_iou_ap
        self.stats[nms_mode]['per_cls_ap'] = per_cls_ap
        return per_iou_ap

    def compute_ar(self, nms_mode):
        # ground_truth = pd.read_csv('datasets/thumos14_test_groundtruth.csv')

        # Computes average recall vs average number of proposals.
        # average_recall, average_nr_proposals = average_recall_vs_nr_proposals(
        #     self.all_pred[nms_mode], self.all_gt)

        # f = interp1d(average_nr_proposals,
        #             average_recall,
        #             axis=0,
        #             fill_value='extrapolate')

        # return {
        #     'AR@1': f(1),
        #     'AR@50': f(50),
        #     'AR@100': f(100),
        #     'AR@200': f(200),
        #     'AR@500': f(500)
        # }

        if self.dataset_name == "Thumos14":
            tiou_thresholds = np.linspace(0.5, 1, 11)
        elif self.dataset_name == "ActivityNet13":
            tiou_thresholds = np.linspace(0.5, 0.95, 10)
        else:
            raise NotImplementedError
        
        recall, avg_recall, proposals_per_video = average_recall_vs_avg_nr_proposals(
            self.all_pred[nms_mode], self.all_gt,
            max_avg_nr_proposals=100,
            tiou_thresholds=tiou_thresholds
            )
        
        area_under_curve = np.trapz(avg_recall, proposals_per_video)


        logger.info(f'[RESULTS] Performance on {self.dataset_name} proposal task.')
        logger.info('\tArea Under the AR vs AN curve: {}%'.format(100.*float(area_under_curve)/proposals_per_video[-1]))

        return {
            'AR@1': np.mean(recall[:,0]),
            'AR@5': np.mean(recall[:,4]),
            'AR@10': np.mean(recall[:,9]),
            'AR@25': np.mean(recall[:,24]),
            'AR@40': np.mean(recall[:,39]),
            'AR@50': np.mean(recall[:,49]),
            'AR@100': np.mean(recall[:,-1]),
            'AUC': float(area_under_curve)/proposals_per_video[-1]
        }


    def dump_to_json(self, dets, save_path):
        result_dict = {}
        videos = dets['video-id'].unique()
        for video in videos:
            this_detections = dets[dets['video-id'] == video]
            det_list = []
            for idx, row in this_detections.iterrows():
                det_list.append(
                    {'segment': [float(row['t-start']), float(row['t-end'])], 'label': self.classes[int(row['cls'])], 'score': float(row['score'])}
                )
            
            video_id = video[2:] if video.startswith('v_') else video
            result_dict[video_id] = det_list

        # the standard detection format for ActivityNet
        output_dict={
            "results": result_dict,
            "external_data":{}}
        if save_path:
            dirname = osp.dirname(save_path)
            if not osp.exists(dirname):
                os.makedirs(dirname)
            with open(save_path, 'w') as f:
                json.dump(output_dict, f)
        # return output_dict

    def dump_detection(self, save_path=None):
        for nms_mode in self.nms_mode:
            logger.info(f'Eval: dump detection result in JSON format to {save_path.format(self.subset, nms_mode)}')
            self.dump_to_json(self.all_pred[nms_mode], save_path.format(self.subset, nms_mode))


def merge_distributed(all_pred):
    '''gather outputs from different nodes at distributed mode'''
    all_pred_gathered = all_gather(all_pred)
    
    merged_all_pred = {k: [] for k in all_pred}
    for p in all_pred_gathered:
        for k in p:
            merged_all_pred[k] += p[k]

    return merged_all_pred


class ActivityNet13Evaluator(TADEvaluator):
    def __init__(self, dataset_name, epoch, dataset, iou_range,
                 nms_mode=['raw'], 
                 num_workers = 8,
                 filter_threshold = 0
                 ):
        '''
        dataset_name:  thumos14, activitynet
        subset: val or test
        anno_file: the path to load anno file
        video_dict: the dataset dict created in video_dataset.py
        iou_range: [0.3:0.7:0.1] for thumos14; [0.5:0.95:0.05] for anet and hacs.
        task_setting: what setting the model for, ['close_set','zero_shot']
        split: how to split the classes in zero-shot setting
        '''

        self.valid_anno_dict = dataset.valid_anno_dict
        super(ActivityNet13Evaluator,self).__init__(dataset_name, dataset.subset, epoch, iou_range, dataset.classes,
                                                nms_mode,
                                                num_workers,
                                                filter_threshold)

    def summarize(self):
        '''Compute mAP and collect stats
           also comupte AR 
        '''

        # 0.5 0.75 0.95 avg
        display_iou_thr_inds = [0, 5, 9]
    
        for nms_mode in self.nms_mode:
            logger.info(f"Eval: mode={nms_mode} {len(self.all_pred[nms_mode])} predictions from {len(self.all_pred[nms_mode]['video-id'].unique())} videos")

        header = ' '.join('%.2f' % self.iou_range[i] for i in display_iou_thr_inds) + ' avg'  # 0 5 9
        lines = []
        for nms_mode in self.nms_mode:
            per_iou_ap = self.compute_map(nms_mode)
            line = ' '.join(['%.2f' % (100*per_iou_ap[i]) for i in display_iou_thr_inds]) + ' %.2f' % (100*per_iou_ap.mean()) + ' {} epoch{}'.format(nms_mode, self.epoch)
            lines.append(line)
        msg = header
        for l in lines:
            msg += '\n' + l
        logger.info('\n' + msg)

        for nms_mode in self.nms_mode:
            self.stats[nms_mode]['AP50'] = self.stats[nms_mode]['per_iou_ap'][0]
        self.stats_summary = msg

        # compute AR
        for nms_mode in self.nms_mode:
            ar_dict = self.compute_ar(nms_mode)
            self.stats[nms_mode].update(ar_dict)

    def import_gt(self):
        # obtain gt dataframe
        all_gt = []
        for viddo_name, value in self.valid_anno_dict.items():
            annotations = value['annotations']
            all_gt += [[viddo_name, self.classes.index(x['label']), x['segment'][0], x['segment'][1]] for x in annotations]
        return all_gt

    def update(self, pred, assign_cls_labels=False):
        '''
        pred: a dict of predictions for each video. For each video, the predictions are in a dict with these fields: scores, labels, segments, query_ids
        assign_cls_labels: whether to follow the tradition to use external video label or manually assign class labels to the detections. This is optional when the predictions are class-agnostic.
        '''
        pred_numpy = {k: {kk: vv.detach().cpu().numpy() for kk, vv in v.items()} for k,v in pred.items()}
        for k, v in pred_numpy.items():
            this_dets = [
                [v['segments'][i, 0], v['segments'][i, 1], v['scores'][i], v['labels'][i]] for i in range(len(v['scores']))
                ]
            video_id = k

            # ignore videos that are not in ground truth set
            if video_id not in self.video_ids:
                continue
            
            this_dets = np.array(this_dets)   # [num_proposals,4]->"start, end, score, label"
            
            for nms_mode in self.nms_mode:
                input_dets = np.copy(this_dets)

                if nms_mode == 'raw':
                    sort_idx = input_dets[:, 2].argsort()[::-1]
                    dets = input_dets[sort_idx, :]
                    # # only keep top 200 detections per video
                    # dets = dets[:200, :]

                    if self.filter_threshold > 0: # filter the low confidence proposals
                        try:
                            fix_idx = np.where(dets[:,2] < self.filter_threshold)[0][0]
                        except:
                            fix_idx = len(dets)
                        dets = dets[:fix_idx]
                        
                    # On ActivityNet, follow the tradition to use external video label
                    if assign_cls_labels:
                            raise NotImplementedError
                    self.all_pred[nms_mode] += [[video_id] + det for det in dets.tolist()] 
                else:
                    raise NotImplementedError
     
    def accumulate(self):
        '''accumulate detections in all videos'''
        for nms_mode in self.nms_mode:
            self.all_pred[nms_mode] = pd.DataFrame(self.all_pred[nms_mode], columns=["video-id", "t-start", "t-end", "score", "cls"])
        
        self.pred_by_cls = {}
        for nms_mode in self.nms_mode:
            self.pred_by_cls[nms_mode] = [self.all_pred[nms_mode][self.all_pred[nms_mode].cls == cls].reset_index(drop=True).drop(labels=['cls'],axis=1) for cls in range(self.num_classes)]



class Thumos14Evaluator(TADEvaluator):
    def __init__(self, dataset_name, epoch, dataset, iou_range,
                 nms_mode=['raw'], 
                 num_workers = 8,
                 filter_threshold = 0
                 ):
        '''
        dataset_name:  thumos14, activitynet
        subset: val or test
        anno_file: the path to load anno file
        video_dict: the dataset dict created in video_dataset.py
        iou_range: [0.3:0.7:0.1] for thumos14; [0.5:0.95:0.05] for anet and hacs.
        task_setting: what setting the model for, ['close_set','zero_shot']
        split: how to split the classes in zero-shot setting
        '''

        self.src_valid_anno_dict = dataset.src_valid_anno_dict # The difference of ActivityNet13
        self.valid_anno_dict = dataset.valid_anno_dict
        self.slice_overlap = dataset.slice_overlap
        self.inference_entire = dataset.inference_entire
        super(Thumos14Evaluator,self).__init__(dataset_name, dataset.subset, epoch, iou_range, dataset.classes,
                                                nms_mode, 
                                                num_workers,
                                                filter_threshold)

    def summarize(self):
        '''Compute mAP and collect stats'''
        # 0.3~0.7 avg
        display_iou_thr_inds = [0, 1, 2, 3, 4]

        for nms_mode in self.nms_mode:
            logger.info(f"Eval: mode={nms_mode} {len(self.all_pred[nms_mode])} predictions from {len(self.all_pred[nms_mode]['video-id'].unique())} videos")

        header = ' '.join('%.2f' % self.iou_range[i] for i in display_iou_thr_inds) + ' avg'  # 0 5 9
        lines = []
        for nms_mode in self.nms_mode:
            per_iou_ap = self.compute_map(nms_mode)
            line = ' '.join(['%.2f' % (100*per_iou_ap[i]) for i in display_iou_thr_inds]) + ' %.2f' % (100*per_iou_ap.mean()) + ' {} epoch{}'.format(nms_mode, self.epoch)
            lines.append(line)
        msg = header
        for l in lines:
            msg += '\n' + l
        logger.info('\n' + msg)

        for nms_mode in self.nms_mode:
            self.stats[nms_mode]['AP50'] = self.stats[nms_mode]['per_iou_ap'][2]
        self.stats_summary = msg
        
        # compute AR
        for nms_mode in self.nms_mode:
            ar_dict = self.compute_ar(nms_mode)
            self.stats[nms_mode].update(ar_dict)

    def import_gt(self):
        # obtain gt dataframe
        all_gt = []
        for viddo_name, value in self.src_valid_anno_dict.items():
            annotations = value['annotations']
            all_gt += [[viddo_name, self.classes.index(x['label']), x['segment'][0], x['segment'][1]] for x in annotations]
        return all_gt

    def update(self, pred, assign_cls_labels=False):
        '''
        pred: a dict of predictions for each video. For each video, the predictions are in a dict with these fields: scores, labels, segments, query_ids
        assign_cls_labels: whether to follow the tradition to use external video label or manually assign class labels to the detections. This is optional when the predictions are class-agnostic.
        '''
        pred_numpy = {k: {kk: vv.detach().cpu().numpy() for kk, vv in v.items()} for k,v in pred.items()}
        if self.inference_entire:
            for k, v in pred_numpy.items():
                this_dets = [
                    [v['segments'][i, 0], 
                        v['segments'][i, 1], 
                        v['scores'][i],
                        v['labels'][i]]
                    for i in range(len(v['scores']))]
                video_id = k

                # ignore videos that are not in ground truth set
                if video_id not in self.video_ids:
                    continue
                
                this_dets = np.array(this_dets)   # [num_proposals,4]->"start, end, score, label"
                
                for nms_mode in self.nms_mode:
                    input_dets = np.copy(this_dets)

                    if nms_mode == 'raw':
                        sort_idx = input_dets[:, 2].argsort()[::-1]
                        dets = input_dets[sort_idx, :]
                        # # only keep top 200 detections per video
                        # dets = dets[:200, :]

                        if self.filter_threshold > 0: # filter the low confidence proposals
                            try:
                                fix_idx = np.where(dets[:,2] < self.filter_threshold)[0][0]
                            except:
                                fix_idx = len(dets)
                            dets = dets[:fix_idx]
                            
                        # On ActivityNet, follow the tradition to use external video label
                        if assign_cls_labels:
                                raise NotImplementedError
                        self.all_pred[nms_mode] += [[video_id] + det for det in dets.tolist()]  # "video-id, start, end, score, label"
                    else:
                        raise NotImplementedError
        else:
            pred_numpy = {k: {kk: vv.detach().cpu().numpy() for kk, vv in v.items()} for k,v in pred.items()}
            for k, v in pred_numpy.items():
                window_start = self.valid_anno_dict[k]['time_offset']
                video_id = self.valid_anno_dict[k]['src_video_name']
                this_dets = [
                    [v['segments'][i, 0] + window_start, 
                        v['segments'][i, 1] + window_start, 
                        v['scores'][i],
                        v['labels'][i]]
                    for i in range(len(v['scores']))]
            
                # ignore videos that are not in ground truth set
                if video_id not in self.video_ids:
                    continue
                
                this_dets = np.array(this_dets)   # [num_proposals,4]->"start, end, score, label"
                
                for nms_mode in self.nms_mode:
                    input_dets = np.copy(this_dets)

                    if nms_mode == 'raw':
                        sort_idx = input_dets[:, 2].argsort()[::-1]
                        dets = input_dets[sort_idx, :]
                        # # only keep top 200 detections per video
                        # dets = dets[:200, :]

                        if self.filter_threshold > 0: # filter the low confidence proposals
                            try:
                                fix_idx = np.where(dets[:,2] < self.filter_threshold)[0][0]
                            except:
                                fix_idx = len(dets)
                            dets = dets[:fix_idx]
                            
                        # On ActivityNet, follow the tradition to use external video label
                        if assign_cls_labels:
                                raise NotImplementedError
                        self.all_pred[nms_mode] += [[video_id, k] + det for det in dets.tolist()]  # "video-id, slice-id, start, end, score, label"
                    else:
                        raise NotImplementedError

    def cross_window_fusion(self):
        '''
        merge detections in the overlapped regions of adjacent windows. Only used for THUMOS14
        '''
        all_pred = []

        video_ids = self.all_pred['raw']['video-id'].unique()
        vid = video_ids[0]

        for vid in video_ids:
            this_dets = self.all_pred['raw'][self.all_pred['raw']['video-id'] == vid]
            slice_ids = this_dets['slice-id'].unique().tolist()
            if len(slice_ids) > 1:
                slice_sorted = sorted(slice_ids, key=lambda k: int(k.split('_')[-2]))
               
                overlap_region_time_list = []
                for i in range(0, len(slice_sorted) - 1):
                    slice_name = slice_sorted[i]
                    feature_fps = self.valid_anno_dict[slice_name]['feature_fps']
                    time_base = 0  # self.video_dict[slice_name]['time_base']
                    # parse the temporal coordinate from name
                    cur_slice = [int(x) for x in slice_sorted[i].split('_')[-2:]]
                    next_slice = [int(x) for x in slice_sorted[i+1].split('_')[-2:]]
                    overlap_region_time = [next_slice[0], cur_slice[1]]
                    # add time offset of each window/slice
                    overlap_region_time = [time_base + overlap_region_time[iii] / feature_fps for iii in range(2)]
                    overlap_region_time_list.append(overlap_region_time)
                
                mask_union = None
                processed_dets = []
                for overlap_region_time in overlap_region_time_list:
                    inters = np.minimum(this_dets['t-end'], overlap_region_time[1]) - np.maximum(this_dets['t-start'], overlap_region_time[0])
                    # we only perform NMS to the overlapped regions
                    mask = inters > 0
                    overlap_dets = this_dets[mask]
                    overlap_dets_arr = overlap_dets[['t-start', 't-end', 'score', 'cls']].values
                    if len(overlap_dets) > 0:
                        kept_dets_arr = apply_nms(np.concatenate((overlap_dets_arr, np.arange(len(overlap_dets_arr))[:, None]), axis=1))
                        processed_dets.append(overlap_dets.iloc[kept_dets_arr[:, -1].astype('int64')])
                    
                    if mask_union is not None:
                        mask_union = mask_union | mask
                    else:
                        mask_union = mask
                # instances not in overlapped region
                processed_dets.append(this_dets[~mask_union])
                all_pred += processed_dets
            else:
                all_pred.append(this_dets)

        all_pred = pd.concat(all_pred)
        self.all_pred['raw'] = all_pred

    def accumulate(self):
        '''accumulate detections in all videos'''
        if self.inference_entire:
            for nms_mode in self.nms_mode:
                self.all_pred[nms_mode] = pd.DataFrame(self.all_pred[nms_mode], columns=["video-id", "t-start", "t-end", "score", "cls"])
        else:
            for nms_mode in self.nms_mode:
                self.all_pred[nms_mode] = pd.DataFrame(self.all_pred[nms_mode], columns=["video-id", "slice-id", "t-start", "t-end", "score", "cls"])
            
        self.pred_by_cls = {}
        for nms_mode in self.nms_mode:
            if nms_mode == 'raw' and self.slice_overlap > 0 and not self.inference_entire:
                self.cross_window_fusion()
            self.pred_by_cls[nms_mode] = [self.all_pred[nms_mode][self.all_pred[nms_mode].cls == cls].reset_index(drop=True).drop(labels=['cls'],axis=1) for cls in range(self.num_classes)]



if __name__ == '__main__':
    pass


