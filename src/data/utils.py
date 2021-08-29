import numpy as np 

# CALCULATE DATASET NORMALIZATION
# RESULTS 
# VV -10.4163 4.0595093
# VH -17.3540 4.3767033

def masked_std(x, mask, num_channels):
    x = np.transpose(x,[3,0,1,2]).reshape(num_channels,-1)
    mask = np.repeat(np.expand_dims(mask > 0, axis = 0), num_channels,axis=0).reshape(num_channels,-1)
    return np.std(x,axis=1,where=mask)

def masked_mean(x, mask, num_channels):
    x = np.transpose(x,[3,0,1,2]).reshape(num_channels,-1)
    mask = np.repeat(np.expand_dims(mask > 0, axis = 0), num_channels,axis=0).reshape(num_channels,-1)
    return np.mean(x,axis=1,where=mask)