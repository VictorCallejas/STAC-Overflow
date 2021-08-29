
import albumentations as A

TEST_AUGMENTATIONS = [
    A.Transpose(p=1),
    A.VerticalFlip(p=1),
    A.HorizontalFlip(p=1),
]

TRAIN_AUGMENTATIONS = [
    A.OneOf([
        A.Transpose(p=1),
        A.VerticalFlip(p=1),
        A.HorizontalFlip(p=1),
        A.RandomRotate90(p=1),
    ])
]