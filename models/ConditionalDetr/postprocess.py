from utils.segment_ops import segment_cw_to_t1t2
import torch
import torch.nn as nn


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    def __init__(self,args):
        super().__init__()
        self.type = args.postprocess_type
        self.topk = args.postprocess_topk

        self.target_type = args.target_type
        self.proposals_weight_type = args.proposals_weight_type
        self.actionness_loss = args.actionness_loss
        self.compact_loss = args.compact_loss
        self.temperature = 1
        self.prob_type = args.prob_type

    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 1] containing the size of each video of the batch
        """
        out_bbox = outputs['pred_boxes'] # [bs,num_queries,2]
        assert 'class_logits' in outputs
        class_logits = outputs['class_logits'] #  [bs,num_queries,num_classes] 

        if self.actionness_loss: # [bs,num_queries,1]
            assert 'actionness_logits' in outputs
            actionness_logits = outputs['actionness_logits']
            actionness_prob = actionness_logits.sigmoid()

            if self.prob_type == "softmax":
                if self.proposals_weight_type == "before_softmax":
                    prob = torch.mul(actionness_prob,class_logits).softmax(-1) # [bs,num_queries,num_classes]
                elif self.proposals_weight_type == "after_softmax":
                    prob = torch.mul(actionness_prob,class_logits.softmax(-1)) # [bs,num_queries,num_classes]
            elif self.prob_type == "sigmoid":
                prob = torch.mul(actionness_prob,class_logits.sigmoid()) # [bs,num_queries,num_classes]
            elif self.prob_type == "none_mul":
                prob = class_logits.softmax(-1) # [bs,num_queries,num_classes]
            else:
                raise NotImplementedError
        elif self.compact_loss:
            assert 'compact_logits' in outputs
            compact_logits = outputs['compact_logits']
            obj_prob = torch.exp(-self.temperature*compact_logits).unsqueeze(-1)
            prob = obj_prob*class_logits.sigmoid()

        else:
            if self.prob_type == "softmax":
                prob = class_logits.softmax(-1)
            elif self.prob_type == "sigmoid":
                prob = class_logits.sigmoid()
            else:
                raise NotImplementedError


        B,Q,num_classes = prob.shape
        assert len(prob) == len(target_sizes)


        if self.type == "class_agnostic":
            assert self.topk >= 1, "so small value for class_agnostic type, please check"
            # sort across different instances, pick top 100 at most
            topk_values, topk_indexes = torch.topk(prob.reshape(B, -1), min(self.topk, Q*num_classes), dim=1) # [bs,num_queries*num_classes] - > [bs,100]
            scores = topk_values
            topk_boxes_idx = torch.div(topk_indexes, num_classes, rounding_mode='trunc') # get the row index of out_logits (b,q,c), i.e., the query idx. [bs,100//num_classes]
            labels = topk_indexes % num_classes # get the col index of out_logits (b,q,c), i.e., the class idx. [bs,100//num_classes]
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = torch.gather(out_boxes, 1, topk_boxes_idx.unsqueeze(-1).repeat(1,1,2)) # [bs,topk,2]
        elif self.type == "class_specific":
            assert self.topk <= 5, "so big value for class_specific type, please check"
            # pick topk classes for each query
            topk_values, topk_indexes = torch.topk(prob, min(self.topk,num_classes), dim=-1) # [bs,num_queries,topk]
            scores, labels = topk_values.flatten(1), topk_indexes.flatten(1) # [bs, num_queries*topk]
            # (bs, nq, 1, 2)
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = out_boxes[:, [torch.div(i, self.topk, rounding_mode='trunc') for i in range(self.topk*out_boxes.shape[1])], :] # [bs,num_queries*topk,2]
            topk_boxes_idx = torch.div(torch.arange(0, self.topk*out_boxes.shape[1], 1, dtype=labels.dtype, device=labels.device), self.topk, rounding_mode='trunc')[None, :].repeat(labels.shape[0], 1)
        elif self.type == "class_one":
            # choose one class that all queries are assigned this category label
            assert self.topk == 1, "so big value for class_one type, please check"
            mean_prob = torch.mean(prob,dim=1) # [bs,num_classes]
            value, idx = torch.topk(mean_prob, self.topk, dim=-1) # [bs,topk=1]
            labels = idx.repeat(1,Q) # [bs,num_queries]
            scores = torch.gather(prob,dim=2,index=idx.unsqueeze(1).repeat(1,Q,1)).squeeze(2) # [bs,num_queries,1]->[bs,num_queries]
            out_boxes = segment_cw_to_t1t2(out_bbox).clamp(min=0,max=1) #  [bs,num_queries,2], the clamp is to deal with the case "center"-"width/2" < 0 and "center"+"width/2" < 1
            topk_boxes = out_boxes # [bs,num_queries,2]
            topk_boxes_idx = torch.arange(0, out_boxes.shape[1], 1, dtype=labels.dtype, device=labels.device)[None, :].repeat(labels.shape[0], 1)

        else:
            raise ValueError("Don't define this post process type: {self.type}")
        
        # from normalized [0, 1] to absolute [0, length] (second) coordinates
        vid_length = target_sizes
        scale_fct = torch.stack([vid_length, vid_length], dim=1) # [bs,2]-> "start, end"
        topk_boxes = topk_boxes * scale_fct[:, None, :] # [bs,topk,2] transform fraction to second

        results = [{'scores': s, 'labels': l, 'segments': b, 'query_ids': q } for s, l, b, q in zip(scores, labels, topk_boxes, topk_boxes_idx)] # corresponding to Tad_eval.update()

        return results

def build_postprocess(args):
    return PostProcess(args)