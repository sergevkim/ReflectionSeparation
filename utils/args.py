import argparse
from pathlib import Path

import torch


def train_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet", type=str, help="model type, default: unet")
    parser.add_argument("--batch-size", default=16, type=int, help="batch_size, default: 16")
    parser.add_argument("--n-epochs", default=10, type=int, help="number of epochs, default: 10")
    parser.add_argument("--version", default=8, type=int, help="version of the model, default: 8")

    parser.add_argument("--subject-limit", default=5400, type=int, help="max number of subject images, default: 5400")
    parser.add_argument("--astigma-limit", default=2700, type=int, help="number of epochs, default: 2700")
    parser.add_argument("--multi_reflection", default=8, type=int, help="multi reflection, default: 8")

    parser.add_argument("--disable-cuda", action='store_true', help="disable CUDA")
    parser.add_argument("--save-model", action='store_true', help="save model")
    parser.add_argument("--verbose", action='store_true', help="verbose")
    parser.add_argument("--from-checkpoint", action='store_true', help="from checkpoint")

    parser.add_argument(
        "--subject-images-path",
        default="{}/data/subject_images".format(Path.cwd()),
        type=str,
        help="subject images path, default: ./data/subject_images")
    parser.add_argument(
        "--astigma-images-path",
        default="{}/data/astigma_images".format(Path.cwd()),
        type=str,
        help="astigma images path, default: ./data/astigma_images")
    parser.add_argument(
        "--checkpoints-path",
        default="{}/checkpoints".format(Path.cwd()),
        type=str,
        help="weigths path, default: ./checkpoints")
    psrser.add_argument(
        "--logs-path",
        default="{}/logs",
        type=str,
        help="logs path, default: ./logs"
    )

    return parser.parse_args()


def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{}/weights/unet_v5_e0.hdf5".format(Path.cwd()),
                        type=str, help="model")
    parser.add_argument("--input", default="{}/data/basket/serge.jpg".format(Path.cwd()),
                        type=str, help="image to handle")
    parser.add_argument("--output", default="{}/data/basket/output.jpg".format(Path.cwd()),
                        type=str, help="output")
    parser.add_argument("--basket-dir", default="{}/data/basket/".format(Path.cwd()),
                        type=str, help="basket dir")

    return parser.parse_args()


def prepare_data_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--where", default="{}/data".format(Path.cwd()), type=str, help="where we want to place a dir with data")
    parser.add_argument("--subject-images-dir-name", default="subject_images", type=str, help="a dir for subject images")
    parser.add_argument("--astigma-images-dir-name", default="astigma_images", type=str, help="a dir for astigma images")
    parser.add_argument("--tar-name", default="indoorCVPR_09.tar", type=str, help="tar name that was downloaded before (README.md)")
    parser.add_argument("--url", default="https://www.hel-looks.com/archive/#20190810_13", type=str, help="url with astigma data")

    return parser.parse_args()
