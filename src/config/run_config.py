
# RUN
SEED = 42
fast_run = False

# DATASET
data_path = '../data/raw/'

#channels = ['vv','vh','abs', 'mask', 'change', 'extent','seasonality','occurrence','recurrence','transitions','nasadem']
channels = ['vv','vh']

val_size = 0.0

batch_size = 10
num_workers = 3

# MODEL
save_path = '../artifacts/'
save_name = 'model.pt'

model_name = 'UnetPlusPlus'
encoder_name = 'resnet18'


# TRAIN
device = 'cuda'
fp16 = False

optimizer = 'AdamW'

lr = 5e-5
swa_lr = 1e-4
steps_to_accumulate = 1

epochs = 1800
swa_epochs = 5

train_watershed = False#
val_watershed = False

train_plot = False#
val_plot = True

