import random
from itertools import chain
from pathlib import Path
from typing import Optional

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import albumentations as A
import cv2


def read_image(image_path):
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


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

    def __init__(self, root_path, transform=None, same_font=False):
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
                A.RandomResizedCrop(32, 32),
            ],
            additional_targets={
                'image1': 'image'
            }
        )
        self.positive_transform = A.Compose([
            A.ShiftScaleRotate(rotate_limit=5, value=0)
        ])

        self.transform = transform
        self.same_font = same_font

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

        if self.transform is not None:
            source_image = self.transform(image=source_image)['image']
            positive_image = self.transform(image=positive_image)['image']
            negative_image = self.transform(image=negative_image)['image']

        return source_image, positive_image, negative_image


class SymbolDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, same_font=False):
        super().__init__()

        self.data_dir = Path(data_dir)
        self.transform = A.Compose([
            A.RandomBrightnessContrast(brightness_by_max=False),
            A.OneOf([
                A.GaussNoise(var_limit=(10, 20)),
                A.GaussianBlur(),
                A.Downscale(interpolation=cv2.INTER_LINEAR),
            ]),
            A.Sharpen(),
            A.ImageCompression()
        ])
        self.same_font = same_font

    def setup(self, stage: Optional[str] = None):
        self.train_set = SymbolDataset(self.data_dir / 'train', transform=self.transform, same_font=self.same_font)
        self.val_set = SymbolDataset(self.data_dir / 'valid', transform=self.transform, same_font=self.same_font)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_set, batch_size=64, shuffle=True, num_workers=4)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.val_set, batch_size=64)
