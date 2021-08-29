import torch 
import numpy as np 
import random 

from datetime import datetime 
import os 

import neptune.new as neptune

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True


def init_experiment():
    
    run = neptune.init(
        project='victorcallejas/Flood',
        api_token='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlNDRlNTJiNC00OTQwLTQxYjgtYWZiNS02OWQ0MDcwZmU5N2YifQ=='
    )
    return run

def dict_from_module(module):
    context = {}
    for setting in dir(module):
        # you can write your filter here
        if setting[0].isalpha():
            context[setting] = getattr(module, setting)

    return context

def config_run_folder(cfg):

    now = datetime.now()

    cfg.save_path = cfg.save_path + now.strftime("%Y_%m_%d_%H_%M_%S/")

    try:
        os.makedirs(cfg.save_path)
    except:
        print('Folder already exists')
        exit()

    print('Run folder created: ', cfg.save_path)