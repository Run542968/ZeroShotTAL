from utils.misc import (accuracy)
from utils.segment_ops import segment_cw_to_t1t2,segment_iou
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example. [bs,num_queries,num_classes]
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs. [bs,num_queries,num_classes]
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid() # [bs,num_queries,num_classes]
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets) # [bs,num_queries,num_classes]
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes 

class SetCriterion(nn.Module):
    """ This class computes the loss for Conditional DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict,focal_alpha, args, base_losses=['labels', 'boxes']):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            base_losses: list of all the losses to be applied. See get_loss for list of available losses.
            focal_alpha: alpha in Focal Loss
            gamma: gamma in Focal Loss
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.focal_alpha = focal_alpha
        self.gamma = args.gamma
        self.target_type = args.target_type

        self.actionness_loss = args.actionness_loss 

        self.distillation_loss = args.distillation_loss
        self.salient_loss = args.salient_loss
        self.salient_loss_impl = args.salient_loss_impl

        self.compact_loss = args.compact_loss
        self.min_obj=-512*math.log(0.9)


        if self.actionness_loss:
            self.base_losses = ['labels','actionness','boxes']
        else:
            self.base_losses = ['labels', 'boxes']

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'class_logits' in outputs
        src_logits = outputs['class_logits'] # [bs,num_queries,num_classes]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["semantic_labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "segments" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        src_boxes = outputs['pred_boxes'][idx] # [batch_matched_queries,2]
        target_boxes = torch.cat([t['segments'][i] for t, (_, i) in zip(targets, indices)], dim=0) # [batch_target,2]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(segment_iou(
            segment_cw_to_t1t2(src_boxes).clamp(min=0,max=1), 
            segment_cw_to_t1t2(target_boxes))) # the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_distillation(self,outputs, targets, indices, num_boxes):
        '''
        for distillation the CLIP feat to DETR detector
        '''
        assert 'student_emb' in outputs
        assert 'teacher_emb' in outputs

        # obtain logits
        student_emb = outputs['student_emb']
        teacher_emb = outputs['teacher_emb'] # [B,Q,dim]

        loss = torch.abs(teacher_emb-student_emb).sum(-1)

        losses = {}
        losses['loss_distillation'] = loss.mean()
        return losses

    def loss_actionness(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'actionness_logits' in outputs
        src_logits = outputs['actionness_logits'] # [bs,num_queries,1]

        idx = self._get_src_permutation_idx(indices) # (batch_idx,src_idx)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]) # [batch_target_class_id]
        target_classes = torch.full(src_logits.shape[:2], src_logits.shape[2],
                                    dtype=torch.int64, device=src_logits.device) # [bs,num_queries]
        target_classes[idx] = target_classes_o # [bs,num_queries]

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device) # [bs,num_queries,num_classes+1]
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs,num_queries,num_classes]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=self.gamma) * src_logits.shape[1]
        losses = {'loss_actionness': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0] # [batch_matched_queries,num_classes]
        return losses

    def loss_salient(self, outputs, targets, indices, num_boxes, log=True):
        """
            Rank loss, for fine-grained boundary perception
            NOTE: If the num_queries is so small, can not cover all gt, an error will appear here
        """

        assert 'salient_logits' in outputs
        assert 'salient_loss_mask' in outputs
        assert 'salient_gt' in outputs
        salient_logits = outputs['salient_logits'] # [bs,t,1]
        salient_logits = salient_logits.squeeze(dim=2) # [bs,t]
        mask = outputs['salient_loss_mask'] # [bs,t]
        salient_gt = outputs['salient_gt'] # [bs,t]

        if self.salient_loss_impl == "BCE":
            prob = salient_logits.sigmoid() # [bs,t]
            ce_loss = F.binary_cross_entropy_with_logits(salient_logits, salient_gt, reduction="none")
            p_t = prob * salient_gt + (1 - prob) * (1 - salient_gt) # [bs,t]
            loss = ce_loss * ((1 - p_t) ** self.gamma)

            if self.focal_alpha >= 0:
                alpha_t = self.focal_alpha * salient_gt + (1 - self.focal_alpha) * (1 - salient_gt)
                loss = alpha_t * loss

            un_mask = ~mask
            loss_salient = loss*un_mask

            loss_salient = loss_salient.mean(1).sum() / num_boxes 

        elif self.salient_loss_impl == "CE":

            salient_gt = salient_gt / (torch.sum(salient_gt, dim=1, keepdim=True) + 1e-4) # [b,t]

            loss_salient = -(salient_gt * F.log_softmax(salient_logits, dim=-1)) # [b,t]
            
            un_mask = ~mask
            loss_salient = loss_salient*un_mask
            loss_salient = loss_salient.sum(dim=1).mean()
        else:
            raise ValueError
 
        losses = {'loss_salient': loss_salient}

        return losses

    def loss_compact(self, outputs, targets, indices, num_boxes):
        assert "compact_value" in outputs
        idx = self._get_src_permutation_idx(indices)
        pred_obj = outputs["compact_value"][idx]
        return  {'loss_compact': torch.clamp(pred_obj, min=self.min_obj).sum()/ num_boxes}

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'boxes': self.loss_boxes,
            'actionness': self.loss_actionness
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets

        if self.actionness_loss and self.target_type != "none": # if using the actionness loss, adopt actionness logits for set-matching
            assert "actionness_logits" in outputs_without_aux
            # We flatten to compute the cost matrices in a batch
            logits = outputs_without_aux["actionness_logits"]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["labels"] for v in targets]) # [gt_instance_num]
            tgt_bbox = torch.cat([v["segments"] for v in targets]) # [gt_instance_num, 2]
            sizes = [len(v["segments"]) for v in targets]
            
        elif not self.actionness_loss or self.target_type == "none":
            assert "class_logits" in outputs_without_aux
            # We flatten to compute the cost matrices in a batch
            logits = outputs_without_aux["class_logits"]

            # Also concat the target labels and boxes
            tgt_ids = torch.cat([v["semantic_labels"] for v in targets]) # [gt_instance_num]
            tgt_bbox = torch.cat([v["segments"] for v in targets]) # [gt_instance_num, 2]
            sizes = [len(v["segments"]) for v in targets]
        else:
            raise ValueError
        
        indices = self.matcher(logits, outputs_without_aux["pred_boxes"], tgt_ids, tgt_bbox, sizes)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.base_losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each innermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.base_losses:
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)


        if self.distillation_loss:
            distillation_loss = self.loss_distillation(outputs, targets, indices, num_boxes)
            losses.update(distillation_loss)
        
        if self.salient_loss:
            salient_loss = self.loss_salient(outputs, targets, indices, num_boxes)
            losses.update(salient_loss)

        if self.compact_loss:
            compact_loss = self.loss_compact(outputs, targets, indices, num_boxes)
            losses.update(compact_loss)


        return losses

def build_criterion(args,num_classes,matcher,weight_dict):
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             focal_alpha=args.focal_alpha,
                             args=args)
    return criterion