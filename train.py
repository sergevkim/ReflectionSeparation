import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from models.torch.UNet import UNet
from models.torch.ResNet import ResNet

import utils
from utils.args import train_parse_args
from utils.data import DummyDataset, filter_filenames, all_transform

from constants import SUBJECT_SIZE, ASTIGMA_SIZE, MULTI_REFLECTION, SUBJECT_TRAIN_SIZE, ASTIGMA_TRAIN_SIZE


def train(args, model, device, train_loader_a, train_loader_b, optimizer, epoch):
    time_start = time.time()
    model.train()

    for batch_index, (a, b) in enumerate(zip(train_loader_a, train_loader_b)):
        batch = all_transform(a, b, device) # loss, metrics, loss_trans.item() - calc for backward pass

        #loss, metrics_dict = model.forward_and_compute_all(batch, device=device)

        output = model.forward(batch['synthetic'])
        loss_transmission = mse_loss(output['transmission'], batch['alpha_transmitted'])
        loss_reflection = mse_loss(output['reflection'], batch['reflected'])

        metrics_dict = {'mse_transmission': loss_transmission.item(),
                        'mse_reflection': loss_reflection.item()}

        loss = loss_transmission + loss_reflection

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.verbose:
            print(epoch, batch_index, metrics_dict)

        if args.save_model:
            torch.save(model.state_dict(), './weights/last.hdf5') #TODO paths for checkpoints
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        './weights/{}_v5_{}_{}.hdf5'.format(args.model, epoch, batch_index))

    print("The training epoch ended in {} seconds".format(time.time() - time_start))


def main():
    np.random.seed(9)

    args = train_parse_args()

    print("====ARGS====")
    for arg in vars(args):
        print("{}: {}".format(arg, vars(args)[arg]))
    print("============")

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is on")
    else:
        device = torch.device('cpu')
        print("CUDA is off")

    if args.model == 'unet':
        model = UNet().to(device)
    elif args.model == 'resnet':
        model = ResNet().to(device) #TODO ResNet #TODO config file for resnet
    else:
        print("args.model must be unet or resnet")
        return 0

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

    subject_filenames = filter_filenames(paths=[str(p) for p in Path("./data/subject_images").glob("*.jpg")], limit=SUBJECT_SIZE)
    astigma_filenames = filter_filenames(paths=[str(p) for p in Path("./data/astigma_images").glob("*.jpg")], limit=ASTIGMA_SIZE)

    subject_filenames = np.array(MULTI_REFLECTION * subject_filenames)
    astigma_filenames = np.array(2 * MULTI_REFLECTION * astigma_filenames)

    print("There are {} subject and {} astigma files".format(len(subject_filenames), len(astigma_filenames)))

    subject_filenames_train, subject_filenames_test = train_test_split(subject_filenames, test_size=0.25, shuffle=True)
    astigma_filenames_train, astigma_filenames_test = train_test_split(astigma_filenames, test_size=0.25, shuffle=True)

    #TODO validation dataset
    train_loader_a = DataLoader(DummyDataset(subject_filenames_train),
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True)
    train_loader_b = DataLoader(DummyDataset(astigma_filenames_train),
                                batch_size=args.batch_size,
                                shuffle=True,
                                drop_last=True)
    test_loader_a = DataLoader(DummyDataset(subject_filenames_test),
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True)
    test_loader_b = DataLoader(DummyDataset(astigma_filenames_test),
                               batch_size=args.batch_size,
                               shuffle=True,
                               drop_last=True)

    for epoch in range(args.n_epochs):
        print(epoch)
        train(args, model, device, train_loader_a, train_loader_b, optimizer, epoch)


if __name__ == "__main__":
    main()

