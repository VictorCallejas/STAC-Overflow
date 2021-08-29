import pandas as pd 

from data.dataset import IterChip

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


def get_dataloaders(cfg):

    metadata = pd.read_csv( cfg.data_path + 'metadata.csv', parse_dates=["scene_start"])

    if cfg.fast_run:
        train, val = train_test_split(metadata.chip_id[:20], test_size=cfg.val_size, random_state=cfg.SEED)
    else:
        train, val = train_test_split(metadata.chip_id, test_size=cfg.val_size, random_state=cfg.SEED)

    train_dataset = IterChip(train, cfg, True)
    val_dataset = IterChip(val, cfg, False)

    train_dataloader = DataLoader(
        dataset=train_dataset,
        sampler=RandomSampler(train_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        sampler=SequentialSampler(val_dataset),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True
    )

    return train_dataloader, val_dataloader