import torch
import torch.nn as nn 

from utils.plot import display_preds
from utils.post import post_watershed

from utils.losses import jaccard_coeff

from tqdm import tqdm


def train_epoch(model, dataloader, optimizer, device, criterion, run, scaler, steps_to_accumulate = 1, fp16 = False):

    model.train()
    optimizer.zero_grad(set_to_none=True)

    labels, preds = torch.tensor([]), torch.tensor([])

    for step, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='TRAINING',leave=False):

        x, targets = batch
        x = x.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=fp16):
            b_preds = model(x).squeeze(1)
            loss = criterion(b_preds, targets)
        scaler.scale(loss).backward()

        #run["train/batch_loss"].log(loss.item())

        if (step + 1) % steps_to_accumulate == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(),2.0)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        preds = torch.cat([preds, b_preds.detach().cpu()], dim = 0)
        labels = torch.cat([labels, targets.detach().cpu()], dim = 0)   

    epoch_loss = criterion(preds,labels)
    jac = jaccard_coeff(preds,labels)
    
    return epoch_loss.item(), jac.item()



def valid_epoch(model, dataloader, device, criterion, run, scaler, fp16 = False, plot = True, watershed = False):

    model.eval()

    labels, preds = torch.tensor([]), torch.tensor([])

    for _, batch in tqdm(enumerate(dataloader),total=len(dataloader),desc='VALIDATION',leave=False):

        x, targets = batch
        x = x.to(device,non_blocking=True)
        targets = targets.to(device,non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=fp16):
            with torch.no_grad(): 
                b_preds = model(x).squeeze(1)
                loss = criterion(b_preds, targets)
        
                scaler.scale(loss)
                
                #run["dev/batch_loss"].log(loss.item())

                preds = torch.cat([preds, b_preds.detach().cpu()], dim = 0)
                labels = torch.cat([labels, targets.detach().cpu()], dim = 0)
                 

    epoch_loss = criterion(preds,labels)
    jac = jaccard_coeff(preds,labels)
    
    '''
    if watershed:
        preds_ws = post_watershed(preds)
        jac_w = jaccard_coeff(preds_ws,labels)
        run["dev/jaccard_ws"].log(jac_w)
    '''

    if plot:
        fig = display_preds(dataloader.dataset ,preds,labels,3)
        run['validation/plt'].upload(fig)

    return epoch_loss.item(), jac.item()


def save_ckpt(model, optimizer, dataloader, cfg, path):
    torch.save({
            'model':model.state_dict(),
            'model_name':cfg.model_name,
            'encoder_name':cfg.encoder_name,
            'optimizer_name':cfg.optimizer,
            'optimizer':optimizer.state_dict(),
            'data':{
                'channels':dataloader.dataset.channels,
                'norm':{
                    'mean':dataloader.dataset.t_mean,
                    'std':dataloader.dataset.t_std
                }
            }
        },
        path
        )