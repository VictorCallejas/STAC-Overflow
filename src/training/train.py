import segmentation_models_pytorch as smp

import torch

from utils.losses import *

from training.utils import train_epoch, valid_epoch

from tqdm import tqdm 

import optuna
import neptune.new as neptune 

from models.swin_unet import SwinTransformerSys
from models.models import CTM

def train(train_dataloader, val_dataloader, run, cfg, trial = None):

    device = torch.device(cfg.device)
    ''' 
    model = SwinTransformerSys(img_size=512, patch_size=4, in_chans=len(cfg.channels), num_classes=1,
            embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.0,
            norm_layer=nn.LayerNorm, ape=True, patch_norm=True,
            use_checkpoint=False, final_upsample="expand_first")
    
    model = getattr(smp,cfg.model_name)(
        encoder_name=cfg.encoder_name,       
        encoder_weights="imagenet", 
        in_channels=len(cfg.channels),                  
        classes=1,
    )
    '''
    model = CTM(cfg)    
    print(model)
    model = model.to(device)

    optimizer = getattr(torch.optim,cfg.optimizer)(
                model.parameters(),
                lr = cfg.lr
            )

    #criterion = LovaszLoss(mode='binary', ignore_index=255)
    #criterion = SoftBCEWithLogitsLoss(ignore_index=255)
    criterion = BCE1_DICE
    #criterion = FocalLoss(ignore_index=255)

    fp16 = cfg.fp16
    scaler = torch.cuda.amp.GradScaler()

    steps_to_accumulate = cfg.steps_to_accumulate

    val_watershed = cfg.val_watershed
    val_plot = cfg.val_plot

    path = cfg.save_path + cfg.save_name

    for epoch in tqdm(range(0, cfg.epochs), total = cfg.epochs,desc='EPOCH',leave=False):
        t_loss, t_jac = train_epoch(model, train_dataloader, optimizer, device, criterion, run, scaler, steps_to_accumulate, fp16)
        v_loss, v_jac = valid_epoch(model, val_dataloader, device, criterion, run, scaler, fp16, val_plot, val_watershed)

        torch.save({
            'model':model.state_dict(),
            'model_name':cfg.model_name,
            'encoder_name':cfg.encoder_name,
            'optimizer':optimizer.state_dict(),
            'data':{
                'channels':train_dataloader.dataset.channels,
                'norm':{
                    'mean':train_dataloader.dataset.t_mean,
                    'std':train_dataloader.dataset.t_std
                }
            }
        },
        path
        )
        #run['model_pt'].upload(path)

        study_metric = (v_jac + t_jac) / 2
        if trial != None:

            trial.report(study_metric.item(), epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                run.stop()
                raise optuna.TrialPruned()

    run.stop()
    return study_metric