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

    history = {
        'mse_t': [],
        'mse_r': [],
        'psnr_t': [],
        'psnr_r': [],
    }

    for batch_index, (transmission, reflection) in enumerate(dataloader_full):
        batch = all_transform(transmission, reflection, device) #TODO remove all_transform: add it to train_loader

        #alpha = np.float32(np.random.uniform(0.75, 0.8)) #TODO temperature function
        #synthetic = alpha * transmission + (1 - alpha) * reflection

        losses = model.compute_losses(batch)

        loss = losses['full']
        mse_t = losses['transmission'].item()
        mse_r = losses['reflection'].item()
        psnr_t = 10 * np.log10(1 / mse_t)
        psnr_r = 10 * np.log10(1 / mse_r)

        history['mse_t'].append(mse_t)
        history['mse_r'].append(mse_r)
        history['psnr_t'].append(psnr_t)
        history['psnr_r'].append(psnr_r)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 0:
            if args.verbose:
                print("EPOCH {}, BATCH {}".format(epoch, batch_index))
                print("mse_t: {}, mse_r: {}".format(mse_t, mse_r))
                print("psnr_t: {}, psnr_r: {}".format(psnr_t, psnr_r))
            if args.save_model:
                checkpoint_dict = {
                    'model': model,
                    'optimizer': optimizer,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                }
                checkpoint_path = "{}/{}_v{}_e{}.hdf5".format(
                    args.weights_path,
                    args.model,
                    args.version,
                    epoch)
                torch.save(checkpoint_dict, checkpoint_path)

    print("The training epoch ended in {} seconds".format(time.time() - time_start))


def val(args, model, test_loader_transmission, test_loader_reflection, device, epoch):
    time_start = time.time()
    model.eval()

    dataloader_full = zip(test_loader_transmission, test_loader_reflection)

    history = {
        'mse_t': [],
        'mse_r': [],
        'psnr_t': [],
        'psnr_r': [],
    }

    for batch_index, (transmission, reflection) in enumerate(dataloader_full):
        batch = all_transform(transmission, reflection, device) #TODO remove all_transform: add it to train_loader
        losses = model.compute_losses(batch)

        mse_t = losses['transmission'].item()
        mse_r = losses['reflection'].item()
        psnr_t = 10 * np.log10(1 / mse_t)
        psnr_r = 10 * np.log10(1 / mse_r)

        history['mse_t'].append(mse_t)
        history['mse_r'].append(mse_r)
        history['psnr_t'].append(psnr_t)
        history['psnr_r'].append(psnr_r)

        if batch_index % 100 == 0:
            if args.verbose:
                print("EPOCH {}, BATCH {}".format(epoch, batch_index))
                print("mse_t: {}, mse_r: {}".format(mse_t, mse_r))
                print("psnr_t: {}, psnr_r: {}".format(psnr_t, psnr_r))

    print("Validation ended in {} seconds, mean mse_t: {}, mean psnr_t: {}".format(
        time.time() - time_start,
        sum(history['mse_t']) / len(history['mse_t']),
        sum(history['psnr_t']) / len(history['psnr_r'])))


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
        optimizer = checkpoint['optimizer']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
    else:
        if args.model == 'unet':
            model = UNet().to(device)
        elif args.model == 'resnet':
            model = ResNet().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        epoch_start = 1

    dataloaders = make_dataloaders(args)
    train_loader_transmission = dataloaders['train_loader_transmission']
    train_loader_reflection = dataloaders['train_loader_reflection']
    test_loader_transmission = dataloaders['test_loader_transmission']
    test_loader_reflection = dataloaders['test_loader_reflection']

    for epoch in range(epoch_start, epoch_start + args.n_epochs):
        print("Epoch {}".format(epoch))
        val(args, model, test_loader_transmission, test_loader_reflection, device, epoch)
        train(args, model, train_loader_transmission, train_loader_reflection, optimizer, device, epoch)
        if args.save_model:
            checkpoint_dict = {
                'model': model,
                'optimizer': optimizer,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }
            checkpoint_path = "{}/{}_v{}_e{}.hdf5".format(
                args.weights_path,
                args.model,
                args.version,
                epoch)
            torch.save(checkpoint_dict, checkpoint_path)


if __name__ == "__main__":
    main()
