import segmentation_models_pytorch as smp

import torch

from utils.losses import *

from training.utils import train_epoch, valid_epoch, save_ckpt

from tqdm import tqdm 

import optuna
import neptune.new as neptune 

from models.swin_unet import SwinTransformerSys
from models.models import Net

import torchvision



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
        encoder_depth=3,
        decoder_channels=(64, 32, 16),
        activation=None,
        decoder_attention_type=None,
        decoder_use_batchnorm=True,
        in_channels=len(cfg.channels),                  
        classes=1,
    )
    '''
    
    model = Net(cfg)
    #model = JA()
    print(model)
    model = model.to(device)

    optimizer = getattr(torch.optim,cfg.optimizer)(
                model.parameters(),
                lr = cfg.lr
            )

    #criterion = LovaszLoss(mode='binary', ignore_index=255)
    #criterion = SoftBCEWithLogitsLoss(ignore_index=255)
    #criterion = BCE1_DICE
    #criterion = FocalLoss(ignore_index=255)
    criterion = DiceLoss('binary', log_loss=True, from_logits=True, smooth=0.05, ignore_index=Y_NAN_VALUE, eps=1e-07)

    fp16 = cfg.fp16
    scaler = torch.cuda.amp.GradScaler()

    steps_to_accumulate = cfg.steps_to_accumulate

    val_watershed = cfg.val_watershed
    val_plot = cfg.val_plot

    path = cfg.save_path + cfg.save_name

    swa_model = torch.optim.swa_utils.AveragedModel(model)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[80, 160], gamma=1.0)
    swa_scheduler = torch.optim.swa_utils.SWALR(optimizer, anneal_strategy="linear", anneal_epochs=cfg.swa_epochs, swa_lr=cfg.swa_lr)

    for epoch in tqdm(range(0, cfg.epochs + cfg.swa_epochs), total = cfg.epochs + cfg.swa_epochs,desc='EPOCH',leave=False):
        t_loss, t_jac = train_epoch(model, train_dataloader, optimizer, device, criterion, run, scaler, steps_to_accumulate, fp16)
        run["train/loss"].log(t_loss)
        run["train/jaccard"].log(t_jac)

        v_loss, v_jac = valid_epoch(model, val_dataloader, device, criterion, run, scaler, fp16, val_plot, val_watershed)
        run["dev/loss"].log(v_loss)
        run["dev/jaccard"].log(v_jac)

        print('TRAIN: ',t_loss, t_jac, 'VAL: ',v_loss, v_jac)

        for param_group in optimizer.param_groups:
            run["lr"].log(param_group['lr'])

        if (epoch >= cfg.epochs) or (t_jac > 0.97):
            swa_model.update_parameters(model)
            swa_scheduler.step()

            torch.optim.swa_utils.update_bn(train_dataloader, swa_model, device=device)
            v_loss, v_jac = valid_epoch(swa_model, train_dataloader, device, criterion, run, scaler, fp16, val_plot, val_watershed,swa='swa/')
            run["swa/train/loss"].log(t_loss)
            run["swa/train/jaccard"].log(t_jac)

            #torch.optim.swa_utils.update_bn(val_dataloader, swa_model, device=device)
            v_loss, v_jac = valid_epoch(swa_model, val_dataloader, device, criterion, run, scaler, fp16, val_plot, val_watershed,swa='swa/')
            run["swa/dev/loss"].log(v_loss)
            run["swa/dev/jaccard"].log(v_jac)

            print('SWA TRAIN: ',t_loss, t_jac, 'SWA VAL: ',v_loss, v_jac)

            save_ckpt(swa_model, optimizer, train_dataloader, cfg, path)
        else:
            scheduler.step()

        #run['model_pt'].upload(path)
        '''
        study_metric = (v_jac + t_jac) / 2
        if trial != None:

            trial.report(study_metric.item(), epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                run.stop()
                raise optuna.TrialPruned()

    run.stop()
    return study_metric
    '''
    return