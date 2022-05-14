import random
from itertools import chain
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def run_canny(image, thr1=100, thr2=200, **kwargs):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.Canny(image, 100, 200)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image


def read_image(image_path, canny=False):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if canny:
        image = run_canny(image)

    return image


def randomly_merge(img1, img2):
    img = img1.clone()

    *_, h, w = img.shape
    rng_index = random.randint(0, 3)
    if rng_index == 0:
        left, right = 0, w
        top, bottom = 0, h // 2
    elif rng_index == 1:
        left, right = 0, w
        top, bottom = h // 2, h
    elif rng_index == 2:
        left, right = 0, w // 2
        top, bottom = 0, h
    elif rng_index == 3:
        left, right = w // 2, w
        top, bottom = 0, h
    else:
        left, right = 0, w
        top, bottom = 0, h

    img[..., top:bottom, left:right] = img2[..., top:bottom, left:right]
    mask = torch.ones_like(img)[0]
    mask[top:bottom, left:right] = 0

    return img, mask


def random_blend(img1, img2, unmask_zeros=False):
    transform = A.CoarseDropout(max_width=32, max_height=32, min_width=4, min_height=4, min_holes=1)
    *_, h, w, ch = img1.shape

    rng_index = random.randint(0, 1)
    if rng_index == 0:
        img = img1.copy()
        to_copy = img2
        fill_value = 1
    elif rng_index == 1:
        img = img2.copy()
        to_copy = img1
        fill_value = 0

    holes = transform.get_params_dependent_on_targets({'image': img})['holes']
    mask = np.full_like(img, 1 - fill_value)[..., 0]

    for x1, y1, x2, y2 in holes:
        img[y1:y2, x1:x2] = to_copy[y1:y2, x1:x2]
        mask[y1:y2, x1:x2] = fill_value

    if unmask_zeros:
        mask[(img[..., 0] == img1[..., 0]) & (img[..., 0] == 0)] = 0

    return img, mask


class SymbolDataset(Dataset):
    SYMBOLS = tuple([
        chr(num) for num in
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 65, 66, 67, 68, 69, 70,
         71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86,
         87, 88, 89, 90, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106,
         107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
         120, 121, 122]
    ])

    @staticmethod
    def char_from_path(path):
        path = Path(path)
        return chr(int(path.stem))

    def __init__(self, root_path, transform=None, same_font=False, canny=False, unmask_zeros=False, val=False):
        self.root_path = Path(root_path)
        self.fonts = {
            font_path.name: {
                self.char_from_path(image_path): image_path
                for image_path in font_path.glob('*.jpg')
            }
            for font_path in self.root_path.iterdir()
            if font_path.is_dir()
        }
        self.symbols = {
            ch: [font[ch] for font in self.fonts.values()]
            for ch in self.SYMBOLS
        }
        self.all_images = list(chain(*(
            font.values() for font in self.fonts.values()
        )))

        self.source_transform = A.Compose([
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.Rotate(value=0),
                A.RandomSizedCrop((8, 32), 64, 64),
            ],
            additional_targets={
                'image1': 'image'
            }
        )
        self.positive_transform = A.Compose([
            A.ShiftScaleRotate(rotate_limit=5, value=0)
        ])

        degradation_transform = [
            # A.CoarseDropout(max_height=16, max_width=16),
            A.RandomBrightnessContrast(brightness_by_max=False),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 20)),
                A.GaussianBlur(),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
            ]),
            A.Sharpen(),
            A.ImageCompression()
        ]

        if canny:
            degradation_transform.append(A.Lambda(image=run_canny))

        self.degradation_transform = A.Compose(degradation_transform)

        self.transform = transform
        self.same_font = same_font
        self.canny = canny
        self.unmask_zeros = unmask_zeros
        self.val = val

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        source_image_path = self.all_images[idx]

        source_image = read_image(source_image_path, canny=self.canny)
        source_ch = self.char_from_path(source_image_path)

        if self.same_font:
            same_sym = self.symbols[source_ch]
            positive_image = read_image(random.choice(same_sym), canny=self.canny)
        else:
            positive_image = source_image

        diff_sym = list(chain(*(self.symbols[ch] for ch in self.SYMBOLS if ch != source_ch)))
        negative_image = read_image(random.choice(diff_sym), canny=self.canny)

        if self.source_transform is not None:
            transform_res = self.source_transform(image=source_image, image1=positive_image)
            source_image, positive_image = transform_res['image'], transform_res['image1']
            transform_res = self.source_transform(image=negative_image)
            negative_image = transform_res['image']

        if self.positive_transform is not None:
            positive_image = self.positive_transform(image=positive_image)['image']

        if self.degradation_transform is not None:
            source_image = self.degradation_transform(image=source_image)['image']
            positive_image = self.degradation_transform(image=positive_image)['image']
            negative_image = self.degradation_transform(image=negative_image)['image']

        semi_image, mask = random_blend(positive_image, negative_image, self.unmask_zeros)

        if self.val and idx < 64:
            if self.transform is not None:
                source_image = self.transform(image=source_image)['image']
                positive_image = self.transform(image=positive_image)['image']
                negative_image = self.transform(image=negative_image)['image']
                semi_image = self.transform(image=semi_image)['image']

                mask = torch.from_numpy(mask).float()

            return source_image, positive_image, negative_image, semi_image, mask
        else:
            if self.transform is not None:
                source_image = self.transform(image=source_image)['image']
                semi_image = self.transform(image=semi_image)['image']

                mask = torch.from_numpy(mask).float()

            return source_image, torch.tensor(0), torch.tensor(0), semi_image, mask


class SRDataset(Dataset):
    def __init__(self, root_path):
        self.root_path = Path(root_path)
        self.images = list(self.root_path.glob('*.png'))
        self.gt = self.root_path / 'GT.png'

        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = read_image(image_path)
        gt = read_image(self.gt)

        image = run_canny(image)
        gt = run_canny(gt)

        image = self.transform(image=image)['image']
        gt = self.transform(image=gt)['image']

        return image_path.stem, gt, image


class SymbolDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, same_font=False, canny=False, unmask_zeros=False):
        super().__init__()

        self.data_dir = Path(data_dir)

        transforms = []

        transforms.extend([
            A.Normalize(),
            ToTensorV2()
        ])

        self.transform = A.Compose(transforms)
        self.same_font = same_font
        self.canny = canny
        self.unmask_zeros = unmask_zeros

    def setup(self, stage: Optional[str] = None):
        self.train_set = SymbolDataset(
            self.data_dir / 'fannet' / 'train',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros
        )
        self.train_subset = Subset(SymbolDataset(
            self.data_dir / 'fannet' / 'train',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros, val=True
        ), list(range(64)))
        self.val_set = SymbolDataset(
            self.data_dir / 'fannet' / 'valid',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros, val=True
        )
        self.sr_set = SRDataset(self.data_dir / 'sr-test')

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=64, shuffle=True, num_workers=16)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        # return DataLoader(self.val_set, batch_size=512, num_workers=16)
        return [
            DataLoader(self.val_set, batch_size=64, num_workers=16),
            DataLoader(self.sr_set, batch_size=1, num_workers=2),
            DataLoader(self.train_subset, batch_size=64, num_workers=16, shuffle=True)
        ]
