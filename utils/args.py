import argparse
from pathlib import Path


def train_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="unet", type=str, help="model type, default: unet")
    parser.add_argument("--batch-size", default=2, type=int, help="batch_size, default: 2")
    parser.add_argument("--n-epochs", default=10, type=int, help="number of epochs, default: 10")
    parser.add_argument("--version", default=5, type=int, help="version of the model, default: 5")

    parser.add_argument("--subject-limit", default=5400, type=int, help="max number of subject images, default: 5400")
    parser.add_argument("--astigma-limit", default=2700, type=int, help="number of epochs, default: 2700")
    parser.add_argument("--multi_reflection", default=8, type=int, help="multi reflection, default: 8")

    parser.add_argument("--disable-cuda", action='store_true', help="disable CUDA")
    parser.add_argument("--save-model", action='store_true', help="save model")
    parser.add_argument("--verbose", action='store_true', help="verbose")
    parser.add_argument("--from-checkpoint", action='store_true', help="from checkpoint")

    parser.add_argument("--subject-images-path", default="{}/data/subject_images".format(Path.cwd()), type=str,
                        help="subject images path")
    parser.add_argument("--astigma-images-path", default="{}/data/astigma_images".format(Path.cwd()), type=str,
                        help="astigma images path")
    parser.add_argument("--weights-path", default="{}/weights".format(Path.cwd()), type=str,
                        help="weigths path")
    parser.add_argument("--logs-path", default="{}/logs".format(Path.cwd()), type=str,
                        help="logs path")
    parser.add_argument("--checkpoint-path", default="{}/weights/last.hdf5".format(Path.cwd()), type=str,
                        help="last checkpoint")

    return parser.parse_args()


def eval_parse_args():
    parser = argparse.ArgumentParser()
    #TODO

    return parser.parse_args()


def test_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="{}/weights/unet_v5_e0.hdf5".format(Path.cwd()),
                        type=str, help="model")
    parser.add_argument("--input", default="{}/data/basket/serge.jpg".format(Path.cwd()),
                        type=str, help="image to handle")
    parser.add_argument("--output", default="{}/data/basket/output.jpg".format(Path.cwd()),
                        type=str, help="output")

    return parser.parse_args()


def prepare_data_parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--where", default="{}/data".format(Path.cwd()), type=str, help="where we want to place a dir with data")
    parser.add_argument("--subject-images-dir-name", default="subject_images", type=str, help="a dir for subject images")
    parser.add_argument("--astigma-images-dir-name", default="astigma_images", type=str, help="a dir for astigma images")
    parser.add_argument("--tar-name", default="indoorCVPR_09.tar", type=str, help="tar name that was downloaded before (README.md)")
    parser.add_argument("--url", default="https://www.hel-looks.com/archive/#20190810_13", type=str, help="url with astigma data")

    return parser.parse_args()


def handle_args(args):
    for arg in vars(args):
        print("{}: {}".format(arg, vars(args)[arg]))

    if not args.disable_cuda and torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is on")
    else:
        device = torch.device('cpu')
        print("CUDA is off")

    if not args.from_checkpoint:
        if args.model == 'unet':
            model = UNet()
        elif args.model == 'resnet':
            model = ResNet() #TODO ResNet
        else:
            print("args.model must be unet or resnet")
            return 0
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        epoch_start = 0
    else:
        checkpoint = torch.load(args.checkpoint_path)
        model = checkpoint['model']
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']

    return {'device': device,
            'model': model,
            'optimizer': optimizer,
            'epoch_start': epoch_start}#TODO random seed

