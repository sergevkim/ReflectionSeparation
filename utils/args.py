import argparse
from pathlib import Path


def train_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet", type=str, help="model type, default: unet")
    parser.add_argument("--batch-size", default=2, type=int, help="batch_size, default: 2")
    parser.add_argument("--n-epochs", default=10, type=int, help="number of epochs, default: 10")

    parser.add_argument("--disable-cuda", action='store_true', help="disable CUDA")
    parser.add_argument("--save-model", action='store_true', help="save model")
    parser.add_argument("--verbose", action='store_true', help="verbose")

    parser.add_argument("--subject-images-path", default="{}/data/subject_images".format(Path.cwd()), type=str, help="subject images path")
    parser.add_argument("--astigma-images-path", default="{}/data/astigma_images".format(Path.cwd()), type=str, help="astigma images path")
    parser.add_argument("--weights-path", default="{}/weights".format(Path.cwd()), type=str, help="weigths path")
    parser.add_argument("--logs-path", default="{}/logs".format(Path.cwd()), type=str, help="logs path")

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

