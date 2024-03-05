import torch
import math
from typing import Iterable
import logging
import sys
from sklearn.metrics import accuracy_score

logger = logging.getLogger()

def train(model: torch.nn.Module, 
          criterion: torch.nn.Module,
          data_loader: Iterable, 
          optimizer: torch.optim.Optimizer,
          device: torch.device, 
          epoch: int, 
          max_norm: float = 0):
    model.train()
    criterion.train()


    epoch_loss_dict_scaled = {}
    epoch_loss_dict_unscaled = {}
    count = 0


    for samples, targets in data_loader:
        samples = samples.to(device)
        targets = [{k: v.to(device) if k in ['segments', 'labels', 'salient_mask', 'semantic_labels'] else v for k, v in t.items()} for t in targets]
        classes = data_loader.dataset.classes
        description_dict = data_loader.dataset.description_dict

        outputs = model(samples, classes, description_dict, targets)

        loss_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict) # the weight_dict controls which loss is applied

        loss_dict_unscaled = {f'{k}_unscaled': v.item() for k, v in loss_dict.items()} # logging all losses thet are computed (note that some of these are not allpied for backward)
        loss_dict_scaled = {k: v.item() * weight_dict[k] for k, v in loss_dict.items() if k in weight_dict}
        loss_value = sum(loss_dict_scaled.values())

        # update the epoch_loss
        epoch_loss_dict_unscaled.update({k: epoch_loss_dict_unscaled.get(k,0.0) + v for k, v in loss_dict_unscaled.items()})
        epoch_loss_dict_scaled.update({k: epoch_loss_dict_scaled.get(k,0.0) + v for k, v in loss_dict_scaled.items()})
        count = count + len(targets)
        logger.info(f"Train Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_value:{loss_value}, loss_dict_scaled:{loss_dict_scaled}")
        logger.info(f"Train Epoch: {epoch} ({count}/{len(data_loader)*len(targets)}), loss_dict_unscaled:{loss_dict_unscaled}")

        if not math.isfinite(loss_value):
            logger.info("Loss is {}, stopping training".format(loss_value))
            logger.info(loss_dict_scaled)
            raise ValueError("Loss is {}, stopping training".format(loss_value))

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()


    epoch_loss_dict_scaled.update({k: v/count for k, v in epoch_loss_dict_scaled.items()})
    epoch_loss_dict_unscaled.update({k: v/count for k, v in epoch_loss_dict_unscaled.items()})
    logger.info(f"Train Epoch: {epoch}, epoch_loss_dict_scaled:{epoch_loss_dict_scaled}")
    logger.info(f"Train Epoch: {epoch}, epoch_loss_dict_unscaled:{epoch_loss_dict_unscaled}")


    return epoch_loss_dict_scaled

