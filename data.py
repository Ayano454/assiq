from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CLASS_NAMES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


def build_transforms(strong_augment: bool = False):
    train_ops = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ]
    if strong_augment:
        train_ops.extend(
            [
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02
                ),
                transforms.RandomRotation(10),
            ]
        )

    train_transform = transforms.Compose(
        train_ops
        + [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ]
    )
    return train_transform, eval_transform



def build_dataloaders(
    data_dir: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    strong_augment: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_transform, eval_transform = build_transforms(strong_augment=strong_augment)

    full_train = datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=train_transform
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )

    # Split train into train/val.
    generator = torch.Generator().manual_seed(seed)
    train_size = 45000
    val_size = 5000
    train_set, val_set = random_split(full_train, [train_size, val_size], generator=generator)

    # Validation should not use augmentation.
    val_set.dataset = datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=eval_transform
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader



def build_test_loader(data_dir: str, batch_size: int, num_workers: int) -> DataLoader:
    _, eval_transform = build_transforms(strong_augment=False)
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=eval_transform
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
