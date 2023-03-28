import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """
    ResNet with identity blocks.
    https://arxiv.org/abs/1512.03385
    """

    def __init__(self, in_channel: int, out_channel: int):
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


class SimpleResNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_res_blocks: int = 3, n_classes: int = 10):
        super(SimpleResNet, self).__init__()

        n_res_blocks = [ResidualBlock(in_channels, out_channels)]

        for _ in range(1, num_res_blocks):
            n_res_blocks.append(ResidualBlock(out_channels, out_channels))

        self.resBlock = nn.ModuleList(n_res_blocks)

        self.avg = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(in_features=32 * 32 * 1, out_features=n_classes)

    def forward(self, x):
        x = self.resBlock(x)
        x = torch.flatten(x, dims=1)
        x = self.avg(x)
        x = self.fc1(x)


        return self.resBlock(x)


BATCH_SIZE = 62
CHANNELS = 1
HEIGHT = 32
WIDTH = 32

# res_block = ResidualBlock(CHANNELS, CHANNELS)
batch = torch.normal(0, 1, size=(BATCH_SIZE, CHANNELS, HEIGHT, WIDTH))

# output = res_block(batch)
# print(output.shape)

model = SimpleResNet(CHANNELS, CHANNELS)
output = model(batch)
print(output.shape)
