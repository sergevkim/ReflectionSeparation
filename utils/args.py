import argparse
from pathlib import Path


def train_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet", type=str, help="model type")
    parser.add_argument("--logs", default="./runs/0")  # TODO where must we use it?
    parser.add_argument("--batch_size", default=2, type=int, help="batch_size")
    parser.add_argument("--n_epochs", default=10, type=int, help="number of epochs")
    parser.add_argument("--disable_cuda", action='store_true', help="disable CUDA")
    parser.add_argument("--save_model", action='store_true', help="save_model")
    parser.add_argument("--verbose", action='store_true', help="verbose")

    return parser.parse_args()


def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", default="{}/data/basket/pexa.jpg".format(Path.cwd()), type=str, help="image to handle")

    return parser.parse_args()


def prepare_data_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--where", default="{}/data".format(Path.cwd()), type=str, help="where we want to place a dir with data")
    parser.add_argument("--subject_images_dir_name", default="subject_images", type=str, help="a dir for subject images")
    parser.add_argument("--astigma_images_dir_name", default="astigma_images", type=str, help="a dir for astigma images")
    parser.add_argument("--tar_name", default="indoorCVPR_09.tar", type=str, help="tar name that was downloaded before (README.md)")
    parser.add_argument("--url", default="https://www.hel-looks.com/archive/#20190810_13", type=str, help="url with astigma data")

    return parser.parse_args()

