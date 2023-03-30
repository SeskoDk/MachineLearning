from typing import Tuple

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader


class ResidualBlock(nn.Module):
    """
    Residual block with identity connection.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.BN2 = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        x = self.conv1(x)
        x = self.BN1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.BN2(x)
        x = x + identity
        x = self.relu(x)

        return x


class Flatten(nn.Module):
    @staticmethod
    def forward(x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return x.view(batch_size, -1)


class SimpleResNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_res_blocks: int = 5, n_classes: int = 10) -> None:
        super(SimpleResNet, self).__init__()

        hidden_channel = in_channels
        n_res_blocks = []
        for _ in range(num_res_blocks):
            res_model = ResidualBlock(hidden_channel, hidden_channel)
            n_res_blocks.append(res_model)

        self.resBlock = nn.ModuleList(n_res_blocks)
        self.classifier = nn.Sequential(
            Flatten(),
            nn.Linear(in_features=32 * 32 * hidden_channel, out_features=n_classes),
            # nn.Softmax(dim=1)
        )

    def compute_residuals(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.resBlock:
            x = layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.compute_residuals(x)
        x = self.classifier(x)
        return x


def transform_load_CIFAR10(transform: transforms.Compose, batch_size: int, download: bool = False) -> Tuple[
    DataLoader, DataLoader]:
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=download, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=download, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Training size: {:4}\nTest size: {:9}'.format(len(train_set), len(test_set)))

    return train_loader, test_loader


def train_model(model: SimpleResNet, data_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                criterion: torch.nn.modules.loss, epochs: int, device: str) -> None:
    
    print("Start training:")
    model.train()
    for epoch in range(epochs):
        losses = []
        accuracies = []

        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)

            # forward + backward + optimize
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()
            losses.append(loss)

            targets_pred = torch.argmax(predictions, dim=1)
            acc = (targets_pred == targets).sum() / targets.size(0) * 100
            accuracies.append(acc)

        accuracy = torch.Tensor(accuracies).mean()
        loss = torch.Tensor(losses).mean()

        print(f"Epoch {epoch + 1:{len(str(epochs))}}/{epochs}, ACC: {accuracy.item():.2f}, MSC_Loss: {loss.item():.3f}")

    print("Finished training")


def eval_model(model: SimpleResNet, data_loader: DataLoader, criterion: torch.nn.modules.loss, device: str) -> None:
    print("Start evaluation:")
    model.eval()

    with torch.no_grad():
        for idx, (features, targets) in enumerate(data_loader):
            features = features.to(device)
            targets = targets.to(device)

            predictions = model(features)
            loss = criterion(predictions, targets)

            targets_pred = torch.argmax(predictions, dim=1)
            acc = (targets_pred == targets).sum() / targets.size(0) * 100

            if idx % 5 == 0:
                print(f"Epoch {idx}, ACC: {acc.item():.2f}, MSC_Loss: {loss.item():.3f}")

    print("Finished evaluation")



def main():
    # Hyperparameters
    BATCH_SIZE = 256
    CHANNELS = 3
    LEARNING_RATE = 0.002
    EPOCHS = 10
    MEAN = (0.5,)
    STD = (0.5,)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN * CHANNELS, std=STD * CHANNELS)
    ])

    train_loader, test_loader = transform_load_CIFAR10(transform, BATCH_SIZE)

    model = SimpleResNet(in_channels=CHANNELS).to(DEVICE)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

    criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, criterion, EPOCHS, DEVICE)

    eval_model(model, test_loader, criterion, DEVICE)

if __name__ == "__main__":
    main()
