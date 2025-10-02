import os
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader


class TrainTransform:
    def __init__(self, size):
        self.img_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0)),
            transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomOrder([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ]),
        ])
        self.color_jitter = transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
            p=0.8
        )

        self.mask_transform = transforms.Compose([
            transforms.RandomResizedCrop(size, scale=(0.6, 1.0), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomRotation(45, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.RandomOrder([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomHorizontalFlip(p=0.5),
            ]),
        ])

        self.tensor_transform = transforms.ToTensor()

    @staticmethod
    def _get_seed():
        return random.randint(0, 2147483647)

    @staticmethod
    def _set_seeds(seed):
        random.seed(seed)
        torch.manual_seed(seed)

    def __call__(self, image, masks):
        # same seed to ensure random transformations are applied consistently on both image and target
        seed = self._get_seed()

        mask_tensors = []
        for mask in masks:
            self._set_seeds(seed)
            mask = self.mask_transform(mask)
            mask_tensors.append(self.tensor_transform(mask))

        self._set_seeds(seed)
        image = self.img_transform(image)
        image = self.color_jitter(image)
        image_tensor = self.tensor_transform(image)

        return image_tensor, torch.cat(mask_tensors)


class EvalTransform:
    def __init__(self, size):
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor()
        ])

    def __call__(self, image, masks):
        image_tensor = self.transform(image)
        mask_tensors = [self.transform(mask) for mask in masks]

        return image_tensor, torch.cat(mask_tensors)


class REFUGEDataset(torch.utils.data.Dataset):
    CLASS_MAPPINGS = {
        0: "optic disc",
        1: "optic cup"
    }

    def __init__(self, image_dir, mask_dir, img_mask_transform):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = img_mask_transform

        self.image_ids = [filename.split(".")[0] for filename in os.listdir(self.image_dir)]

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image = Image.open(os.path.join(self.image_dir, f"{image_id}.jpg"))
        mask = Image.open(os.path.join(self.mask_dir, f"{image_id}.bmp"))
        disc_mask = mask.point(lambda p: 255 if (p == 0 or p != 255) else 0).convert("1")
        cup_mask = mask.point(lambda p: 255 if p == 0 else 0).convert("1")

        return self.transform(image, [disc_mask, cup_mask])


def get_train_loader(image_dir, mask_dir, size, batch_size):
    loader = DataLoader(
        REFUGEDataset(image_dir, mask_dir, TrainTransform(size)),
        batch_size=batch_size,
        shuffle=True,
        num_workers=5,
        drop_last=True
    )

    return loader


def get_val_loader(image_dir, mask_dir, size, batch_size):
    loader = DataLoader(
        REFUGEDataset(image_dir, mask_dir, EvalTransform(size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    return loader
