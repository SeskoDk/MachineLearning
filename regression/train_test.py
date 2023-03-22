import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn
from typing import Tuple, List
import matplotlib.pyplot as plt
from model import LinearRegression
from regression import custom_regression_data
from regression import custom_classification_data
from regression import binary_classification_data


def load_data(n_samples: int = 1000, category: str = "regression") -> Tuple[torch.Tensor, torch.Tensor]:
    if category == "regression":
        X, y = custom_regression_data(N=n_samples)
    elif category == "classification":
        X, y = custom_classification_data(N=n_samples)
    else:
        X, y = binary_classification_data(N=n_samples)
    return X, y


def train_test_split(data: np.ndarray, targets: np.ndarray, split_ratio: Tuple = (3, 1),
                     shuffle: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    N_SAMPLES = len(data)
    SPLIT_UNIT = N_SAMPLES // np.sum(split_ratio)

    INDICES = np.arange(N_SAMPLES)
    if shuffle:
        np.random.shuffle(INDICES)

    TRAIN_SPLIT_INDEX = split_ratio[0] * SPLIT_UNIT
    TEST_SPLIT_INDEX = split_ratio[1] * SPLIT_UNIT

    train_indices = INDICES[: TRAIN_SPLIT_INDEX]
    test_indices = INDICES[TRAIN_SPLIT_INDEX:]

    train_samples = data[train_indices]
    test_samples = data[test_indices]
    train_targets = targets[train_indices]
    test_targets = targets[test_indices]

    train_data = np.array([train_samples, train_targets])
    test_data = np.array([test_samples, test_targets])

    return train_data, test_data


def train_linearRegression(model, data: np.ndarray, targets: np.ndarray, device: str = "cuda", epochs: int = 100,
                           learning_rate=0.5) -> List:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    model.to(device)

    X = data.astype("float32")
    X = torch.from_numpy(X).to(device)
    X = X[:, None]

    y_true = targets.astype("float32")
    y_true = torch.from_numpy(y_true)
    y_true = y_true[:, None].to(device)

    losses = []
    for _ in tqdm(range(epochs)):
        optimizer.zero_grad()
        y_pred = model(X)
        loss = criterion(y_pred, y_true)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return losses


def fit_linearRegression(model, data: np.ndarray, device: str = "cuda") -> torch.Tensor:
    X_test = data.astype("float32")
    X_test = torch.from_numpy(X_test).to(device)
    X_test = X_test[:, None]
    with torch.no_grad():
        y_pred = model(X_test)
    return y_pred


def plot_linearRegression():
    data = load_data()
    train_data, test_data = train_test_split(data=data[0], targets=data[1])
    X, y = train_data[0], train_data[1]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    linear_model = LinearRegression()
    losses = train_linearRegression(model=linear_model, data=X, targets=y, device=device)
    y_pred = fit_linearRegression(linear_model, data=test_data[0], device=device)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    axs[0].scatter(test_data[0], test_data[1])
    axs[0].scatter(test_data[0], y_pred[:, 0].cpu().detach().numpy())
    axs[0].grid()
    axs[0].set_title("Model fitting the data")
    axs[1].plot(np.arange(len(losses)), losses)
    axs[1].grid()
    axs[1].set_title("MSELoss")
    plt.show()


def main():
    plot_linearRegression()


if __name__ == "__main__":
    main()
