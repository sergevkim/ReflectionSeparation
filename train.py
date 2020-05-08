import argparse
import time

import numpy as np
import torch

from models.UNet import UNet
from models.ResNet import ResNet

from utils.args import train_parse_args
from utils.data import DummyDataset, make_dataloaders, filter_filenames, all_transform


def train(args, model, train_loader_transmission, train_loader_reflection, optimizer, device, epoch):
    time_start = time.time()
    model.train()

    dataloader_full = zip(train_loader_transmission, train_loader_reflection)

    for batch_index, (transmission, reflection) in enumerate(dataloader_full):
        batch = all_transform(transmission, reflection, device) #TODO remove all_transform: add it to train_loader
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
                            'epoch': epoch
                           },
                           "{}/last_model_optimizer.hdf5".format(args.weights_path))

    print("The training epoch ended in {} seconds".format(time.time() - time_start))


def val(args, model, test_loader_transmission, test_loader_reflection, device):
    time_start = time.time()
    model.eval()

    dataloader_full = zip(test_loader_transmission, test_loader_reflection)

    for batch_index, (transmission, reflection) in enumerate(dataloader_full):
        batch = all_transform(transmission, reflection)
        losses = model.compute_losses(batch)

        if batch_index % 100 == 0:
            if args.verbose:
                print("e{}.b{}:".format(epoch, batch_index))
                print("mse_t: {}, mse_r: {}".format(losses['transmission'].item(), losses['reflection'].item()))

    print("Validation ended in {} seconds".format(time.time() - time_start))


def main():
    np.random.seed(9)

    args = train_parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.from_checkpoint:
        checkpoint = torch.load(args.checkpoint_path)
        model = checkpoint['model'].to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        #optimizer = checkpoint['optimizer']
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
    else:
        if args.model == 'unet':
            model = UNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        epoch_start = 0

    dataloaders = make_dataloaders(args)
    train_loader_transmission = dataloaders['train_loader_transmission']
    train_loader_reflection = dataloaders['train_loader_reflection']
    test_loader_transmission = dataloaders['test_loader_transmission']
    test_loader_reflection = dataloaders['test_loader_reflection']

    for epoch in range(epoch_start, epoch_start + args.n_epochs):
        val(args, model, test_loader_transmission, test_loader_reflection, device)
        train(args, model, train_loader_transmission, train_loader_reflection, optimizer, device, epoch)
        if args.save_model:
            torch.save({'model': model,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                       },
                       "{}/{}_v{}_e{}.hdf5".format(args.weights_path, args.model, args.version, epoch))


if __name__ == "__main__":
    main()

