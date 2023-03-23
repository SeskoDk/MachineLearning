import torch
import yaml
from typing import Dict, Tuple

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split


def load_config_file(filename: str = "configuration.yaml") -> Dict:
    with open(filename, "r") as file:
        configfile = yaml.safe_load(file)
    return configfile


def create_MNIST_data(config: Dict) -> Tuple[torch.utils.data.dataset.Subset,
                                             torch.utils.data.dataset.Subset,
                                             torch.utils.data.dataset.Subset]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=config["MNIST"]["mean"], std=config["MNIST"]["std"])
    ])

    train_dataset = datasets.MNIST(root=config["path"], train=True, transform=transform, download=True)
    test_set = datasets.MNIST(root=config["path"], train=False, transform=transform)

    VALIDATION_SIZE = 10000
    TRAIN_SIZE = len(train_dataset) - VALIDATION_SIZE
    train_set, validation_set = random_split(train_dataset, lengths=[TRAIN_SIZE, VALIDATION_SIZE])

    print(f"Train set: {len(train_set)}",
          f"\nValidation set: {len(validation_set)}",
          f"\nTest set: {len(test_set)}")

    return train_set, validation_set, test_set



def main():
    config = load_config_file()
    train_set, validation_set, test_set = create_MNIST_data(config)


if __name__ == "__main__":
    main()
