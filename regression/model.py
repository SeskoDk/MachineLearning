import torch
import torch.nn as nn


class LinearRegression(nn.Module) -> None:
    def __init__(self, input_size: int = 1, output_size: int = 1):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
