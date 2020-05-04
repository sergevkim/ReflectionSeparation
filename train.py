import argparse
import time

import numpy as np
import torch

from models.UNet import UNet
from models.ResNet import ResNet

from utils.args import train_parse_args, handle_args
from utils.data import DummyDataset, make_dataloaders, filter_filenames, all_transform


def train(args, model, train_loader_subject, train_loader_astigma, device, optimizer, epoch):
    time_start = time.time()
    model.train()

    train_loader_full = zip(train_loader_subject, train_loader_astigma)

    for batch_index, (subject, astigma) in enumerate(train_loader_full):
        batch = all_transform(subject, astigma, device)
        losses = model.compute_losses(batch)
        loss = losses['full']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            if args.verbose:
                print("e{}.b{}:".format(epoch, batch_index))
                print("mse_t: {}, mse_r: {}".format(losses['transmission'].item(), losses['reflection'].item()))
            if args.save_model:
                torch.save({'model': model,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'epoch': epoch},
                           "{}/last_model_optimizer.hdf5".format(args.weights_path))

    print("The training epoch ended in {} seconds".format(time.time() - time_start))


#TODO def eval(args, model, test_loaders_subject, test_loader_astigma, device):


def main():
    np.random.seed(9)

    args = train_parse_args()

    metaparameters = handle_args(args)

    device = metaparameters['device']
    model = metaparameters['model']
    optimizer = metaparameters['optimizer']
    epoch_start = metaparameters['epoch_start']

    dataloaders = make_dataloaders(args)

    train_loader_subject = dataloaders['train_loader_subject']
    train_loader_astigma = dataloaders['train_loader_astigma']
    test_loader_subject = dataloaders['test_loader_subject']
    test_loader_astigma = dataloaders['test_loader_astigma']

    for epoch in range(epoch_start, epoch_start + args.n_epochs):
        train(args, model, train_loader_subject, train_loader_astigma, optimizer, device, epoch)
        if args.save_model:
            torch.save({'model': model,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch},
                       "{}/{}_v{}_e{}.hdf5".format(args.weights_path, args.model, args.version, epoch))


if __name__ == "__main__":
    main()

