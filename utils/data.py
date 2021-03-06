from pathlib import Path
from tqdm.auto import tqdm

import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor


def filter_filenames(paths, limit=None):
    #removes images from filenames that have incorrect (small) shapes

    good_paths = []

    for path in tqdm(paths):
        try:
            img = cv2.imread(path)
            w, h, c = img.shape
            assert w > 400 and h > 400 and c == 3, "Small image: {}".format(img.shape)
            good_paths.append(path)
        except:
            pass

    return good_paths


def make_dataloaders(args, epoch):
    subject_filenames = [str(p) for p in Path(args.subject_images_path).glob("*.jpg")]
    astigma_filenames = [str(p) for p in Path(args.astigma_images_path).glob("*.jpg")]

    subject_filenames = filter_filenames(paths=subject_filenames, limit=args.subject_limit)
    astigma_filenames = filter_filenames(paths=astigma_filenames, limit=args.astigma_limit)

    subject_filenames = np.array(args.multi_reflection * subject_filenames)
    astigma_filenames = np.array(2 * args.multi_reflection * astigma_filenames)

    print("There are {} subject and {} astigma files".format(
        len(subject_filenames),
        len(astigma_filenames)))

    subject_filenames_train, subject_filenames_test = train_test_split(subject_filenames, test_size=0.1, shuffle=True)
    astigma_filenames_train, astigma_filenames_test = train_test_split(astigma_filenames, test_size=0.1, shuffle=True)

    train_loader_transmission = DataLoader(
        DummyDataset(
            mode='transmission',
            paths=subject_filenames_train,
            epoch=epoch,
            color_space=args.color_space),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    train_loader_reflection = DataLoader(
        DummyDataset(
            mode='reflection',
            paths=astigma_filenames_train,
            epoch=epoch,
            color_space=args.color_space),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    test_loader_transmission = DataLoader(
        DummyDataset(
            mode='transmission',
            paths=subject_filenames_test,
            epoch=epoch,
            color_space=args.color_space),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)
    test_loader_reflection = DataLoader(
        DummyDataset(
            mode='reflection',
            paths=astigma_filenames_test,
            epoch=epoch,
            color_space=args.color_space),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True)

    dataloaders = {
        'train_loader_transmission': train_loader_transmission,
        'train_loader_reflection': train_loader_reflection,
        'test_loader_transmission': test_loader_transmission,
        'test_loader_reflection': test_loader_reflection
    }

    return dataloaders


def random_crop(img, window=(None, None)):
    w, h = window
    img_w, img_h, img_c = img.shape
    if w is None and h is None:
        #w = h = min(img_w, img_h)
        w = 200
        h = 200

    assert (w + 200 <= img_w) and (h + 200 <= img_h) and (img_c == 3), "Bad image shape: {}".format(img.shape)

    border_w = img_w - w - 100
    border_h = img_h - h - 100

    if img_w > w:
        x = np.random.randint(
            min(100, border_w) - 1,
            border_w)
    else:
        x = 0

    if img_h > h:
        y = np.random.randint(
            min(100, border_h) - 1,
            border_h)
    else:
        y = 0

    return img[x:x+w, y:y+h, :]


class DummyDataset:
    def __init__(
            self,
            mode,
            paths,
            epoch,
            color_space="rgb",
            new_size=(128, 128)):
        self.mode = mode
        self.paths = paths
        self.epoch = epoch
        self.color_space = color_space
        self.new_size = new_size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = cv2.imread(path)

        if self.color_space == 'rgb':
            img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'lab':
            img_cvt = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        if self.mode == 'transmission':
            img_resized = cv2.resize(img_cvt, self.new_size)
            return ToTensor()(img_resized)
        elif self.mode == 'reflection':
            img_cropped = random_crop(img_cvt, (200, 200))
            img_resized = cv2.resize(img_cropped, self.new_size)
            return ToTensor()(img_resized)


def all_transform(
        subject, astigma,
        device,
        epoch,
        reflection_kernel_size=(8, 8),
        blur_kernel_size=(5, 5)):
    """
    :param subject: batch of images from one domain
    :param astigma: batch of images from another domain
    :param device: 'cuda' or 'cpu'
    """
    if epoch == 1:
        alpha = np.float32(np.random.uniform(0.6, 0.8))
    else:
        alpha = np.float32(np.random.uniform(0.25, 0.75))
    reflection_kernel = np.zeros(reflection_kernel_size)
    x1, y1, x2, y2 = np.random.randint(0, reflection_kernel_size[0], size=4)
    reflection_kernel[x1, y1] = 1.0 - np.sqrt(alpha)
    reflection_kernel[x2, y2] = np.sqrt(alpha) - alpha
    reflection_kernel = cv2.GaussianBlur(reflection_kernel, blur_kernel_size, 0)

    transmission = subject['resized'] * alpha
    reflection = cv2.filter2D(np.float32(astigma['cropped']), ddepth=-1, kernel=reflection_kernel)
    synthetic = transmission + reflection

    batch = {
        'synthetic': synthetic.to(device),
        'transmission': transmission.to(device),
        'reflection': reflection.to(device)
    }

    return batch

