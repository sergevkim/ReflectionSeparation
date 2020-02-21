import argparse
import pathlib

import numpy as np

import tensorflow as tf
from tensorflow.data import Dataset

import torch
from torch.utils.data import DataLoader

import data
import models
from utils.args import train_parse_args


if __name__ == "__main__":
    template = "torch: {},\ntensorflow: {}"
    print(template.format(torch.version, tf.version))

    args = train_parse_args()

    template = "logs = {},\nbatch_size = {},\nn_epochs = {}"
    print(template.format(args.logs, args.batch_size, args.n_epochs))

    device = torch.device('cuda')
