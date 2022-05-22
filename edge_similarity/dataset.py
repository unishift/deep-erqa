import random
from itertools import chain
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from albumentations.pytorch import ToTensorV2
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader, Subset


def run_canny(image, thr1=100, thr2=200, **kwargs):
    image = cv2.UMat(image)  # UMat for GPU acceleration

    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.Canny(image, thr1, thr2)
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    return image.get()


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
        fill_value = 1
    elif rng_index == 1:
        img = img2.copy()
        fill_value = 0

    holes = transform.get_params_dependent_on_targets({'image': img1})['holes']
    mask = np.full_like(img1, 1 - fill_value)[..., 0]

    for x1, y1, x2, y2 in holes:
        mask[y1:y2, x1:x2] = fill_value

    if unmask_zeros:
        mask[(img2[..., 0] == img1[..., 0]) & (img1[..., 0] == 0)] = 0

    return mask


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
            A.RandomBrightnessContrast(brightness_by_max=False),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 20)),
                A.GaussianBlur(),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
            ]),
            A.Sharpen(),
            A.ImageCompression()
        ]

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

        source_image = read_image(source_image_path)
        source_ch = self.char_from_path(source_image_path)

        if self.same_font:
            same_sym = self.symbols[source_ch]
            positive_image = read_image(random.choice(same_sym))
        else:
            positive_image = source_image

        diff_sym = list(chain(*(self.symbols[ch] for ch in self.SYMBOLS if ch != source_ch)))
        negative_image = read_image(random.choice(diff_sym))

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

        if self.canny:
            source_image = run_canny(source_image)
            positive_image = run_canny(positive_image)
            negative_image = run_canny(negative_image)

        mask = random_blend(positive_image, negative_image, self.unmask_zeros)

        semi_image = np.where(mask[..., None], negative_image, positive_image)
        mask = A.resize(mask.astype(float), mask.shape[0] // 4, mask.shape[1] // 4, cv2.INTER_AREA)
        mask = mask > 0

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


class VSRbenchmark(Dataset):
    def __init__(self, root_path, choose_frame=None, train_mode=False):
        self.choose_frame = choose_frame
        self.train_mode = train_mode

        self.root_path = Path(root_path)
        self.images_path = self.root_path / 'imgs'
        if choose_frame is None:
            self.frames = sorted(self.images_path.glob('*/*.PNG'))
            self.gt = sorted(self.images_path.joinpath('GT').glob('*.PNG'))
        else:
            self.frames = sorted(self.images_path.glob('*.png'))
            self.gt = self.images_path.joinpath('GT.png')

        self.subjective_path = self.root_path / 'subjectify_all.csv'
        self.subjective = {}
        with open(self.subjective_path, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                if idx == 0:
                    continue

                name, score = line.strip().split(';')
                name = name.replace('"', '')
                score = float(score.replace(',', '.'))
                self.subjective[name] = score

        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.choose_frame is None:
            name = frame.parent.stem
            frame_idx = int(frame.stem[len('frame_'):]) - 1
            gt = self.gt[frame_idx]
        else:
            name = frame.stem
            gt = self.gt

        frame = read_image(frame, canny=True)
        gt = read_image(gt, canny=True)

        frame = self.transform(image=frame)['image']
        gt = self.transform(image=gt)['image']

        if self.train_mode:
            subjective = self.subjective[name]
            return name, gt, frame, subjective
        else:
            return gt, frame


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
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros,
            val=True
        ), list(range(64)))
        self.val_set = SymbolDataset(
            self.data_dir / 'fannet' / 'valid',
            transform=self.transform, same_font=self.same_font, canny=self.canny, unmask_zeros=self.unmask_zeros,
            val=True
        )
        self.sr_set = VSRbenchmark(self.data_dir / 'sr-test', choose_frame=50, train_mode=True)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=256, shuffle=True, num_workers=16)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return [
            DataLoader(self.val_set, batch_size=64, num_workers=16),
            DataLoader(self.sr_set, batch_size=1, num_workers=2),
            DataLoader(self.train_subset, batch_size=64, num_workers=16, shuffle=True)
        ]


