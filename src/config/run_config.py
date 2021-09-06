
# RUN
SEED = 42
fast_run = False

# DATASET
data_path = '../data/raw/'

#channels = ['vv','vh','abs', 'mask', 'change', 'extent','seasonality','occurrence','recurrence','transitions','nasadem']
channels = ['vv','vh', 'change', 'extent','seasonality','occurrence','recurrence','transitions','nasadem']

val_size = 0.33

batch_size = 8
num_workers = 3

# MODEL
save_path = '../artifacts/'
save_name = 'model.pt'

model_name = 'Unet'
encoder_name = 'resnet18'


# TRAIN
device = 'cuda'
fp16 = False

optimizer = 'SGD'

lr = 5e-1
swa_lr = 5e-2
steps_to_accumulate = 1

epochs = 500
swa_epochs = 7

train_watershed = False#
val_watershed = False

train_plot = False#
val_plot = True

