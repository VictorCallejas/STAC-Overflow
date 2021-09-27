
import torch
import torch.nn as nn 

import numpy as np 

from segmentation_models_pytorch.losses.soft_bce import SoftBCEWithLogitsLoss
from segmentation_models_pytorch.losses import DiceLoss, LovaszLoss, FocalLoss

Y_NAN_VALUE = 255

def jaccard_coeff(preds, true):

    preds = nn.Sigmoid()(preds)
    preds = (preds > 0.5) * 1

    valid_pixel_mask = true.ne(255)
    true = true.masked_select(valid_pixel_mask)
    preds = preds.masked_select(valid_pixel_mask)

    intersection = np.logical_and(true, preds)
    union = np.logical_or(true, preds)
    return intersection.sum() / union.sum()

def BCE1_DICE(preds, true):
    #f(x) = BCE + 1 — DICE
    bce = SoftBCEWithLogitsLoss(ignore_index = Y_NAN_VALUE,smooth_factor = 0.01)
    dice = DiceLoss('binary', log_loss=True, from_logits=True, smooth=0.01, ignore_index=Y_NAN_VALUE, eps=1e-07)
    return 0.1 * bce(preds,true) + 0.9 * dice(preds,true)