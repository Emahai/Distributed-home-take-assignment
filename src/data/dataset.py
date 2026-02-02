import torch
from torchvision import datasets, transforms

def get_datasets(cfg):
    name = cfg["dataset"]["name"].lower()
    root = cfg["dataset"]["root"]

    if name == "cifar10":
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_ds = datasets.CIFAR10(root=root, train=True, download=True, transform=transform_train)
        test_ds = datasets.CIFAR10(root=root, train=False, download=True, transform=transform_test)
        num_classes = 10
        in_shape = (3, 32, 32)
        return train_ds, test_ds, num_classes, in_shape

    raise ValueError(f"Unsupported dataset: {name}")
