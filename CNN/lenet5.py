import torch
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10) -> None:

        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels=6, kernel_size=5)
        self.avg1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.avg2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)
        self.fc1 = nn.Linear(in_features=120, out_features=80)
        self.fc2 = nn.Linear(in_features=80, out_features=num_classes)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.avg1(x)
        x = self.tanh(self.conv2(x))
        x = self.avg2(x)
        x = self.tanh(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = self.tanh(self.fc1(x))
        x = self.softmax(self.fc2(x))
        return x


def main():
    BATCH_SIZE = 5
    N_CHANNELS = 1
    HEIGHT = 32
    WIDTH = 32

    batch = torch.normal(mean=0, std=2, size=(BATCH_SIZE, N_CHANNELS, HEIGHT, WIDTH))
    model = LeNet5()
    output = model(batch)
    print(output)
    print(output.shape)


if __name__ == "__main__":
    main()
