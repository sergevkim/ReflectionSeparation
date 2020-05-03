import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

import cv2

from models.torch.UNet import UNet
from models.torch.ResNet import ResNet
from models.draft import DummyModel
from models.draft2 import DummyModel2

import utils
from utils.args import test_parse_args
from utils.data import DummyDataset, filter_filenames, all_transform

from constants import SUBJECT_SIZE, ASTIGMA_SIZE, MULTI_REFLECTION


def test(args, model, image, device):
    model.eval()
    output = model(image)


def main():
    np.random.seed(9)

    args = test_parse_args()

    #model = DummyModel2().to(torch.device('cpu'))

    '''
    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'resnet':
        model = ResNet().to(device)
    else:
        print("args.model must be unet or resnet")
        return 0
    '''

    ckpt = torch.load(args.model, map_location=torch.device('cpu'))
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])

    def viewImage(image, name_of_window):
        cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
        cv2.imshow(name_of_window, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    image = cv2.imread(args.input)
    rgb_image = [np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2, 0, 1))]

    print(image.shape, rgb_image[0].shape)
    output = model(torch.Tensor(rgb_image))['transmission']
    print(output)
    ready_img_2 = np.transpose(output[0].detach().cpu().numpy(), (1, 2, 0))
    cv2.imwrite(args.output, ready_img_2)
    print(mse_loss(output, torch.Tensor(rgb_image)))


if __name__ == "__main__":
    main()

