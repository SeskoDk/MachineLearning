import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple


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
    X = np.vstack(X)  # N x feat_dim
    y = np.vstack(y)  # N x 1

    return X, y


def main():
    X, y = dummy_classification_data()
    plt.scatter(X[:, 0], X[:, 1], c=y)
    plt.show()


if __name__ == "__main__":
    main()
