import argparse
import time

import numpy as np
import torch

from models.UNet import UNet
from models.ResNet import ResNet

from utils.args import train_parse_args, handle_args
from utils.data import DummyDataset, make_dataloaders, filter_filenames, all_transform


def train(args, model, train_loader_transmission, train_loader_reflection, optimizer, device, epoch):
    time_start = time.time()
    model.train()

    train_loader_full = zip(train_loader_transmission, train_loader_reflection)

    for batch_index, (subject, astigma) in enumerate(train_loader_full):
        batch = all_transform(subject, astigma, device) #TODO remove all_transform and replace it with reflection + tranmission
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


#TODO def val(args, model, test_loaders_subject, test_loader_astigma, device):


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
        train(args, model, train_loader_transmission, train_loader_reflection, optimizer, device, epoch)
        #val(args, model, test_loader_subject, test_loader_astigma)
        if args.save_model:
            torch.save({'model': model,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'epoch': epoch
                       },
                       "{}/{}_v{}_e{}.hdf5".format(args.weights_path, args.model, args.version, epoch))


if __name__ == "__main__":
    main()

