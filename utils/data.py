import cv2
import numpy as np
from tqdm.auto import tqdm


def filter_filenames(paths, limit=None):
    '''
    removes images from filenames that have incorrect (small) shapes
    '''
    good_paths = []

    for path in tqdm(paths):
        try:
            img = cv2.imread(path)
            w, h, c = img.shape
            assert w > 128 and h > 128 and c == 3, "Small image: {}".format(img.shape)
            good_paths.append(path)
        except:
            pass

    return good_paths


def random_crop(img, w=None, h=None):
    img_w, img_h, img_c = img.shape
    if w is None and h is None:
        w = h = min(img_w, img_h)

    assert (img_w >= w) and (img_h >= h) and (img_c == 3), "Bad image shape: {}".format(img.shape)

    if img_w > w:
        x = np.random.randint(0, img_w - w)
    else:
        x = 0

    if img_h > h:
        y = np.random.randint(0, img_h - h)
    else:
        y = 0

    return img[x:x + w, y:y + h, :]


class DummyDataset:
    def __init__(self, paths, transform_fn=None, reflection_size=5, blur=5):
        self.paths = paths
        self.transform_fn = transform_fn
        self.reflection_size = reflection_size
        self.blur = blur

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        path = self.paths[item]
        img = cv2.imread(path)
        img = random_crop(img).astype(np.float32) / 255.0

        resized = cv2.resize(img, (128, 128))
        random_cropped = random_crop(img, 128, 128)
        alpha = np.float32(np.random.uniform(0.75, 0.8))

        kernel = np.zeros((self.reflection_size, self.reflection_size))
        x1, y1, x2, y2 = np.random.randint(0, self.reflection_size, size=4)
        """
        why so strange positions in kernel?
        """
        kernel[x1, y1] = 1.0 - np.sqrt(alpha)
        kernel[x2, y2] = np.sqrt(alpha) - alpha
        if self.blur > 0:
            kernel = cv2.GaussianBlur(kernel, (self.blur, self.blur), 0)
        """
        read about cv2 filter2D and others
        """
        reflected = cv2.filter2D(random_cropped, -1, kernel)

        return {'img': np.transpose(resized, (2, 0, 1)),
                'reflected': np.transpose(reflected, (2, 0, 1)),
                'alpha': alpha}


def all_transform(a, b):
    """
    :param a: batch of images from one domain
    :param b: batch of images from another domain
    :param device: 'cuda' or 'cpu'
    """
    alpha_transmitted = b['alpha'][:, None, None, None] * a['img']
    reflected = b['reflected']
    synthetic = alpha_transmitted + reflected

    return {'synthetic': synthetic,
            'alpha_transmitted': alpha_transmitted,
            'reflected': reflected}



