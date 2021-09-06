import torch
import torch.nn as nn

import torchvision.transforms as T

import rasterio

from torch.utils.data import Dataset

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

from data.aug import TRAIN_AUGMENTATIONS, TEST_AUGMENTATIONS
from data.utils import masked_mean, masked_std

from tqdm import tqdm

import numpy as np 

from pysheds.grid import Grid

FEATURES_PATH = '/train_features/'
EXT = '.tif'

DTYPE = torch.float32
MEM_DTYPE = np.float32

X_NAN_VALUE = -255
Y_NAN_VALUE = 255



class IterChip(Dataset):

    def __init__(self, chip_ids, cfg, augment):
        super().__init__()

        print('creating dataset')

        self.chip_ids = chip_ids
        self.DATA_PATH = cfg.data_path
        self.FEATURES_PATH = FEATURES_PATH

        self.augment = augment
        self.TRAIN_AUGMENTATIONS = TRAIN_AUGMENTATIONS

        self.TEST_AUGMENTATIONS = TEST_AUGMENTATIONS

        self.aug_to_tensor = A.Compose([
                    ToTensorV2()
                ])
        
        self.channels = cfg.channels
        self.num_channels = len(self.channels)

        self.x = []
        self.mask = []
        self.y = []

        print('reading chips')
        for id_ in tqdm(self.chip_ids):

            x = np.array([]).reshape(0,512,512)

            ### X CHANNELS
            vv, vh = None, None
            # VV
            if 'vv' in self.channels:
                path = self.DATA_PATH + self.FEATURES_PATH + id_ + '_vv' + EXT
                with rasterio.open(path) as img:
                    vv = np.expand_dims(img.read(1),axis=0)

                x = np.concatenate([x, vv], axis=0)

            # VH
            if 'vh' in self.channels: 
                path = self.DATA_PATH + self.FEATURES_PATH + id_ + '_vh' + EXT
                with rasterio.open(path) as img:
                    vh = np.expand_dims(img.read(1),axis=0)
                x = np.concatenate([x, vh], axis=0)


            # NASADEM 12 dims
            if 'nasadem' in self.channels:
                path = self.DATA_PATH + '/nasadem/' + id_ + EXT
                with rasterio.open(path) as img:
                    nasadem = np.expand_dims(img.read(1),axis=0)
                x = np.concatenate([x, nasadem], axis=0)

                grid = Grid.from_raster(path, data_name='dem')
                
                # Resolve flats
                grid.resolve_flats('dem', out_name='inflated_dem')
                x = np.concatenate([x,np.expand_dims(grid.inflated_dem,axis=0)], axis=0)

                # Flow direction on hot encoded
                dirmap = (1, 2, 3, 4, 5, 6, 7, 8)
                grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
                one_hot = np.transpose(np.eye(9)[grid.dir],(2,0,1))
                x = np.concatenate([x,one_hot], axis=0)
                
                # Accumulation
                grid.accumulation(data='dir', dirmap=dirmap, out_name='acc')
                x = np.concatenate([x,np.expand_dims(grid.acc,axis=0)], axis=0)
            
            # ABS
            if 'abs' in self.channels:
                if True:#vv == None:
                    path = self.DATA_PATH + self.FEATURES_PATH + id_ + '_vv' + EXT
                    with rasterio.open(path) as img:
                        vv = np.expand_dims(img.read(1),axis=0)
                if True:#vh == None:
                    path = self.DATA_PATH + self.FEATURES_PATH + id_ + '_vh' + EXT
                    with rasterio.open(path) as img:
                        vh = np.expand_dims(img.read(1),axis=0)
                va = np.abs(vv - vh)
                x = np.concatenate([x, va], axis=0)

            # CHANGE
            p_channels = ['change', 'extent','seasonality','occurrence','recurrence','transitions']
            for c in p_channels:
                if c in self.channels:
                    path = self.DATA_PATH + '/jrc_' + c + '/' + id_ + EXT
                    with rasterio.open(path) as img:
                        c_jrc = np.expand_dims(img.read(1),axis=0)
                    x = np.concatenate([x, c_jrc], axis=0)

            ### MASK
            path = self.DATA_PATH + self.FEATURES_PATH + id_ + '_vv'  + EXT
            with rasterio.open(path) as img:
                mask = img.dataset_mask() / 255
            self.mask.append(mask)

            if 'mask' in self.channels:
                x = np.concatenate([x, np.expand_dims(mask,axis=0)], axis=0)
            self.x.append(x)

            ### LABEL
            path = self.DATA_PATH + '/labels/' + id_  + EXT
            with rasterio.open(path) as img:
                y = img.read(1)
            self.y.append(y)

        self.x = np.transpose(np.array(self.x,dtype=MEM_DTYPE),[0,2,3,1])
        self.mask = np.array(self.mask,dtype=MEM_DTYPE)
        self.y = np.array(self.y,dtype=MEM_DTYPE)

        print('calculating norms - mean and std')
        #self.t_mean = masked_mean(self.x,self.mask, self.num_channels)
        #self.t_std = masked_std(self.x,self.mask, self.num_channels)
        '''
        print('NORMALIZE: ', self.t_mean, self.t_std)
        print('normalizing...')
        
        for dim, _p in enumerate(zip(self.t_mean, self.t_std)):
            _mean, _std = _p
            self.x[:,dim] = (self.x[:,dim] - _mean) / _std
        '''
        print('DATASET CREATED', self.x.shape)


    def __len__(self):
        return len(self.chip_ids)

    def __getitem__(self, index, test_augment=False):

        x = self.x[index]
        y = self.y[index]

        # Train Augments
        if self.augment:

            aug = A.Compose(
                self.TRAIN_AUGMENTATIONS
            )

            transformed = aug(image=x,mask=y)

            x, y = transformed['image'], transformed['mask']


        # Test Augments
        if test_augment:
            for augmentation in self.TEST_AUGMENTATIONS:

                aug = A.Compose([
                    augmentation,
                ])

        # CHANGE NAN VALUE
        #x[0][y == Y_NAN_VALUE] = X_NAN_VALUE
        #x[1][y == Y_NAN_VALUE] = X_NAN_VALUE

        #x = torch.tensor(x).to(DTYPE)
        #y = torch.tensor(y).to(DTYPE)

        transformed = self.aug_to_tensor(image=x, mask=y)
        x, y =  transformed['image'], transformed['mask']
        return x, y
        '''
        return {
            #'id':self.chip_ids[index],
            #'idx':index,
            'x':x,
            #'mask':mask,
            'label':y
        }
        '''

    def converge_aug_inference(self, preds):
        
        y = np.transpose(preds.unsqueeze(1).numpy(), [0, 2, 3, 1])

        p = np.expand_dims(y[0], axis=0)
        y = y[1:]

        for augmentation, y_ in zip(self.TEST_AUGMENTATIONS, y):

            aug = A.Compose([
                augmentation
            ])

            transformed = aug(image=y_)

            p = np.concatenate((p, np.expand_dims(transformed['image'], axis=0)))
        
        p = np.transpose(p, [0, 3, 1, 2]).squeeze(1)
        p = np.mean(p,axis=0)
        
        p = torch.tensor(p).to(DTYPE).unsqueeze(0)

        return p