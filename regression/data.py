import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


def affine_function(X: np.ndarray, slope: float = 2.0, intercept: float = 5.0) -> np.ndarray:
    return slope * X + intercept


def dummy_linear_data(N: int = 300, value_range: Tuple = (-10, 10),
                      mean: float = 0.0, std: float = 5.0) -> Tuple[np.ndarray, np.ndarray]:
    LOWER_BOUND, UPPER_BOUND = value_range
    X = np.random.uniform(low=LOWER_BOUND, high=UPPER_BOUND, size=N)
    noise = np.random.normal(loc=mean, scale=std, size=N)
    y = affine_function(X) + noise
    return X, y


def dummy_classification_data(N: int = 300, n_classes: int = 3,
                              feat_dim: int = 2, seed: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    if N % n_classes != 0:
        raise ValueError("Sample size N must be divisible by the number of classes.")

    SAMPLES_PER_CLASS = int(N / n_classes)

    X = []
    y = []

    np.random.seed(seed)
    for c in range(n_classes):
        mean = np.random.uniform(-10, 10)
        std = np.random.uniform(0, 3)
        for _ in range(SAMPLES_PER_CLASS):
            r_sample = np.random.normal(loc=mean, scale=std, size=feat_dim)
            X.append(r_sample)
            y.append(c)
    X = np.vstack(X)
    y = np.vstack(y)

    return X, y


def main():
    X, y = dummy_classification_data()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


if __name__ == "__main__":
    main()
