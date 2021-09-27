import sys
from pathlib import Path

from loguru import logger
import numpy as np
import typer
from tifffile import imwrite, imread
from tqdm import tqdm

import segmentation_models_pytorch as smp

ROOT_DIRECTORY = Path("/codeexecution")
SUBMISSION_DIRECTORY = ROOT_DIRECTORY / "submission"
DATA_DIRECTORY = ROOT_DIRECTORY / "data"
INPUT_IMAGES_DIRECTORY = DATA_DIRECTORY / "test_features"


def make_predictions(chip_id: str, model, device):
    """
    Given an image ID, read in the appropriate files and predict a mask of all ones or zeros
    """

    x = np.array([]).reshape(0,512,512)

    vv = np.expand_dims(imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vv.tif"),axis=0)
    x = np.concatenate([x, vv], axis=0)

    vh = np.expand_dims(imread(INPUT_IMAGES_DIRECTORY / f"{chip_id}_vh.tif"),axis=0)
    x = np.concatenate([x, vh], axis=0)

    # AUG test inference 8 dihedral
    x = np.expand_dims(x,axis=0)
    input_ = np.array([]).reshape(0,2,512,512)

    input_ = np.concatenate([input_, x], axis=0) # identity
    input_ = np.concatenate([input_,numpy.rot90(x, k=1, axes=(2, 3))])# 90
    input_ = np.concatenate([input_,numpy.rot90(x, k=2, axes=(2, 3))])# 180
    input_ = np.concatenate([input_,numpy.rot90(x, k=3, axes=(2, 3))])# 270
    input_ = np.concatenate([input_,numpy.rot90(numpy.transpose(x, axes=(2, 3)),k=2, axes=(2, 3))])# T
    input_ = np.concatenate([input_,numpy.transpose(x, axes=(2, 3))])# T2
    input_ = np.concatenate([input_,numpy.flip(x, axes=2)])# HF
    input_ = np.concatenate([input_,numpy.flip(x, axes=3)])# VF

    input_ = torch.tensor(input_)
    input_ = input_.to(device)

    with torch.no_grad():
        preds = model(x).squeeze(1).detach().cpu()

    preds = nn.Sigmoid()(preds)
    preds = torch.median(preds, dim=0, keepdim=True)
    preds = (preds > 0.5) * 1
        
    return preds.numpy()


def get_expected_chip_ids():
    """
    Use the input directory to see which images are expected in the submission
    """
    paths = INPUT_IMAGES_DIRECTORY.glob("*.tif")
    # images are named something like abc12.tif, we only want the abc12 part
    ids = list(sorted(set(path.stem.split("_")[0] for path in paths)))
    return ids


def main():
    """
    for each input file, make a corresponding output file using the `make_predictions` function
    """
    chip_ids = get_expected_chip_ids()

    checkpoint = torch.load('../artifacts/2021_09_25_06_16_08/model.pt')
    model = getattr(smp,checkpoint['model_name'])(
        encoder_name='resnet18',        
        encoder_depth=5,
        #decoder_channels=(64, 32, 16),
        activation=None,
        decoder_attention_type=None,
        decoder_use_batchnorm=True,
        in_channels=2,                  
        classes=1,
    )
    device = torch.device('cuda')
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    logger.info("Model loaded")

    if not chip_ids:
        typer.echo("No input images found!")
        raise typer.Exit(code=1)
    logger.info(f"found {len(chip_ids)} expected image ids; generating predictions for each ...")
    for chip_id in tqdm(chip_ids, miniters=25, file=sys.stdout, leave=True):
        # figure out where this prediction data should go
        output_path = SUBMISSION_DIRECTORY / f"{chip_id}.tif"
        # make our predictions! (you should edit `make_predictions` to do something useful)
        output_data = make_predictions(chip_id, model, device)
        imwrite(output_path, output_data, dtype=np.uint8)
    logger.success(f"... done")


if __name__ == "__main__":
    typer.run(main)