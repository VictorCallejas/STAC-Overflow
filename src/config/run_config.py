
# RUN
SEED = 42
fast_run = False

# DATASET
data_path = '../data/raw/'

#channels = ['vv','vh','abs', 'mask', 'change', 'extent','seasonality','occurrence','recurrence','transitions','nasadem']
channels = ['vv','vh', 'change', 'extent','seasonality','occurrence','recurrence','transitions','nasadem']

val_size = 0.33

batch_size = 8
num_workers = 0

# MODEL
save_path = '//HAL9000v2/nfs/artifacts/flood/'
save_name = 'model.pt'

model_name = 'Unet'
encoder_name = 'timm-resnest101e'


# TRAIN
device = 'cuda'
fp16 = False

optimizer = 'AdamW'

lr = 2e-4
steps_to_accumulate = 1

train_watershed = False#
val_watershed = False

train_plot = False#
val_plot = True

epochs = 400