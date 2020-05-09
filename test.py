import argparse
from pathlib import Path
import time

import numpy as np
#from sklearn.model_selection import train_test_split

import torch
from torch.nn.functional import mse_loss
#from torch.utils.data import DataLoader

import cv2

from models.UNet import UNet
from models.ResNet import ResNet

import utils
from utils.args import test_parse_args
#from utils.data import DummyDataset, filter_filenames, all_transform


def test(args, model, image, device):
    model.eval()
    output = model(image)


def process(model, img):    #TODO add it as method to a model class
    img = img.transpose((2, 0, 1))[None, ...].astype(np.float32) / 255.0
    with torch.no_grad():
        x = torch.tensor(img)
        out = model(x)["transmission"]
        #out = self.model(x)["reflection"]
        out = out.data.numpy()

    #print(torch.nn.functional.mse_loss(torch.Tensor(img), torch.Tensor(out)).item())

    out = (255.0 * out[0, ...]).clip(0, 255).astype(np.uint8)
    return out.transpose((1, 2, 0))


def main():
    np.random.seed(9)

    args = test_parse_args()

    ckpt = torch.load(args.model, map_location=torch.device('cpu'))
    model = ckpt['model']
    model.load_state_dict(ckpt['model_state_dict'])

    basket_filenames = [str(p) for p in Path(args.basket_dir).glob("*.jpg")]
    basket_out_filenames = [str(p)[:-4] + "_out" + ".jpg" for p in Path(args.basket_dir).glob("*.jpg")]

    for i, filename in enumerate(basket_filenames):
        if filename[-7:] == "out.jpg":
            continue
        print(i, filename)
        image = cv2.imread(filename)
        out = process(model, image)
        cv2.imwrite(basket_out_filenames[i], out)

    '''
    rgb_image = [np.transpose(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), (2, 0, 1))]

    print(image.shape, rgb_image[0].shape)
    output = model(torch.Tensor(rgb_image))['transmission']
    print(output)
    ready_img_2 = np.transpose(output[0].detach().cpu().numpy(), (1, 2, 0))
    cv2.imwrite(args.output, ready_img_2)
    print(mse_loss(output, torch.Tensor(rgb_image)))
    '''

if __name__ == "__main__":
    main()

