import argparse
import time

import numpy as np
import cv2
import torch
from torch.utils.tensorboard import SummaryWriter

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

    for batch_index, (subject, astigma) in enumerate(dataloader_full):
        if batch_index == 1:
            out = subject[0]
            print('1', out.shape, args.color_space)
            out = out.data.numpy()
            print('2', out.shape, args.color_space)
            out = (255.0 * out[0, ...]).clip(0, 255).astype(np.uint8)
            print('3', out.shape, args.color_space)
            out = out.transpose((1, 2, 0))
            print('4', out.shape, args.color_space)
            if args.color_space == 'rgb':
                out = cv2.cvtColor(out, code=cv2.COLOR_RGB2BGR)
            elif args.color_space == 'lab':
                out = cv2.cvtColor(out, code=cv2.COLOR_LAB2BGR)
            cv2.imwrite("normal.jpg", out)

        batch = model.prepare_batch(subject, astigma, device, epoch)

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
                print("BATCH {}".format(batch_index))
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
                    args.checkpoints_path,
                    args.model,
                    args.version,
                    epoch)
                torch.save(checkpoint_dict, checkpoint_path)

    mean_mse = sum(history['mse_t']) / len(history['mse_t'])
    mean_psnr = sum(history['psnr_t']) / len(history['psnr_t'])

    template = "The training epoch ended in {} seconds,\nmean mse_t: {},\nmean psnr_t: {}"
    print(template.format(
        time.time() - time_start,
        mean_mse,
        mean_psnr))

    return mean_mse, mean_psnr


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

    for batch_index, (subject, astigma) in enumerate(dataloader_full):
        batch = model.prepare_batch(subject, astigma, device, epoch)
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
                print("VALIDATION BATCH {}".format(batch_index))
                print("mse_t: {}, mse_r: {}".format(mse_t, mse_r))
                print("psnr_t: {}, psnr_r: {}".format(psnr_t, psnr_r))

    mean_mse = sum(history['mse_t']) / len(history['mse_t'])
    mean_psnr = sum(history['psnr_t']) / len(history['psnr_r'])

    print("Validation ended in {} seconds,\nmean mse_t: {},\nmean psnr_t: {}".format(
        time.time() - time_start,
        mean_mse,
        mean_psnr))

    return mean_mse, mean_psnr


def main():
    np.random.seed(9)
    args = train_parse_args()

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if args.from_checkpoint:
        checkpoint = torch.load(args.cur_checkpoint_path)
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
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4) #TODO lr schedule
        epoch_start = 1

    writer_train = SummaryWriter("{}/tensorboard/train".format(args.logs_path))
    writer_val = SummaryWriter("{}/tensorboard/train".format(args.logs_path))

    for epoch in range(epoch_start, epoch_start + args.n_epochs):
        print("\n======== EPOCH {} ========".format(epoch))
        dataloaders = make_dataloaders(args, epoch)
        train_loader_transmission = dataloaders['train_loader_transmission']
        train_loader_reflection = dataloaders['train_loader_reflection']
        test_loader_transmission = dataloaders['test_loader_transmission']
        test_loader_reflection = dataloaders['test_loader_reflection']
        '''
        print("\nVALIDATIONVALIDATIONVALIDATION")
        mse_val, psnr_val = val(
            args=args,
            model=model,
            test_loader_transmission=test_loader_transmission,
            test_loader_reflection=test_loader_reflection,
            device=device,
            epoch=epoch)
        writer_val.add_scalar('psnr/val', psnr_val, epoch)
        '''
        print("\nTRAINTRAINTRAINTRAINTRAINTRAIN")
        mse_train, psnr_train = train(
            args=args,
            model=model,
            train_loader_transmission=train_loader_transmission,
            train_loader_reflection=train_loader_reflection,
            optimizer=optimizer,
            device=device,
            epoch=epoch)
        print(psnr_train)
        writer_train.add_scalar('psnr/train', psnr_train, epoch)

        if args.save_model:
            checkpoint_dict = {
                'model': model,
                'optimizer': optimizer,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
            }
            checkpoint_path = "{}/{}_v{}_e{}.hdf5".format(
                args.checkpoints_path,
                args.model,
                args.version,
                epoch)
            torch.save(checkpoint_dict, checkpoint_path)


if __name__ == "__main__":
    main()
