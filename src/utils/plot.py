import rasterio 

import matplotlib.pyplot as plt
import numpy as np

import torch

import torchvision.transforms as T

DATA_PATH = '../../data/raw/'
EXT = '.tif'

# Helper functions for visualizing Sentinel-1 images
def scale_img(matrix):

    min_values = np.array([-23, -28, 0.2])
    max_values = np.array([0, -5, 1])

    w, h, d = matrix.shape
    matrix = np.reshape(matrix, [w * h, d]).astype(np.float64)

    matrix = (matrix - min_values[None, :]) / (
        max_values[None, :] - min_values[None, :]
    )
    matrix = np.reshape(matrix, [w, h, d])

    return matrix.clip(0, 1)

def display_chip(chip_id,metadata):

    f, ax = plt.subplots(1, 2, figsize=(9, 9))

    path = DATA_PATH + '/train_features/' + chip_id + '_vv' + EXT
    with rasterio.open(path) as img:
        vv = img.read(1)
    
    path = DATA_PATH + '/train_features/' + chip_id + '_vh' + EXT
    with rasterio.open(path) as img:
        vh = img.read(1)

    path = DATA_PATH + '/labels/' + chip_id + EXT
    with rasterio.open(path) as img:
        y = img.read(1)

    s1_img = np.stack((vv, vh), axis=-1)

    img = np.zeros((512, 512, 3), dtype=np.float32)
    img[:, :, :2] = s1_img.copy()
    img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]

    s1_img = scale_img(img)

    ax[0].imshow(s1_img)
    ax[0].set_title("S1 Chip", fontsize=14)

    label = np.ma.masked_where((y == 0) | (y == 255), y)

    ax[1].imshow(s1_img)
    ax[1].imshow(label, cmap="cool", alpha=1)
    ax[1].set_title("S1 Chip with Label", fontsize=14)

    plt.tight_layout(pad=5)
    plt.show()
    plt.close()

def display_preds(x_,pred_,true_,num=5):
    '''
    mean = x_.t_mean
    std = x_.t_std

    invTrans = T.Compose([
            T.Normalize(
                mean= [-m/s for m, s in zip(mean, std)],
                std= [1/s for s in std]
            )
        ])
    '''
    f, ax = plt.subplots(num, 3, figsize=(9, 9))

    for i in range(0,num):

        #x = invTrans(x_.__getitem__(i)[0])
        x = x_.__getitem__(i)[0]
        pred = pred_[i]
        true = true_[i]

        y = true.numpy()

        # X
        x = x.numpy()
        s1_img = np.transpose(x, [1, 2, 0])[:, :, :2]

        img = np.zeros((512, 512, 3), dtype=np.float32)
        img[:, :, :2] = s1_img.copy()
        img[:, :, 2] = s1_img[:, :, 0] / s1_img[:, :, 1]

        s1_img = scale_img(img)

        ax[i][0].imshow(s1_img)
        ax[i][0].set_title("S1 Chip "+str(i), fontsize=14)

        # Pred
        pred = torch.nn.Sigmoid()(pred)
        pred = (pred > 0.5) * 1
        pred = pred.numpy()
        pred = np.ma.masked_where((pred == 0) | (y == 255), pred)

        ax[i][1].imshow(s1_img)
        ax[i][1].imshow(pred, cmap="cool", alpha=1)
        ax[i][1].set_title("Prediction", fontsize=14)

        # Label
        label = np.ma.masked_where((y == 0) | (y == 255), y)

        ax[i][2].imshow(s1_img)
        ax[i][2].imshow(label, cmap="cool", alpha=1)
        ax[i][2].set_title("Ground Truth", fontsize=14)

    return f