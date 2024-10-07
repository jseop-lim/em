from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, TypeAlias
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_line_graphs(
    functions: dict[str, Callable[[float], float]],
    x: list[int | float],
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    """Plot a line graph with multiple lines."""
    for name, f in functions.items():
        y = [f(x_i) for x_i in x]
        plt.plot(x, y, label=name)

    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.legend()
    plt.show()


def parse_file_to_array(filepath: Path) -> npt.NDArray[np.float64]:
    """Parse a file with space-separated integers into a numpy array."""
    return np.loadtxt(filepath, dtype=np.float64)


def split_dataset_into_parts(
    dataset: npt.NDArray[np.float64],
    k: int,
) -> list[npt.NDArray[np.float64]]:
    """Split a dataset into k parts."""
    return np.array_split(dataset, k)


class TrainDataset(NamedTuple):
    """A dataset split into training and validation parts."""

    train: npt.NDArray[np.float64]
    validation: npt.NDArray[np.float64]


def generate_dataset_parts(
    dataset: npt.NDArray[np.float64], k: int
) -> Iterator[TrainDataset]:
    """Generate k parts of a dataset, each with a different validation part."""
    parts: list[npt.NDArray[np.float64]] = split_dataset_into_parts(dataset, k)

    return (
        TrainDataset(
            train=np.concatenate(parts[:i] + parts[i + 1 :], axis=0),
            validation=parts[i],
        )
        for i in range(k)
    )


# cluster 수 z, train data 주어질 때 학습된 모델을 반환


@dataclass
class MultivariateNormalDistribution:
    mean: npt.NDArray[np.float64]
    cov: npt.NDArray[np.float64]
    dim: int = 1

    def __post_init__(self) -> None:
        if self.mean.shape != (self.dim,):
            raise ValueError(f"Mean shape is {self.mean.shape}, expected {(self.dim,)}")
        if self.cov.shape != (self.dim, self.dim):
            raise ValueError(
                f"Covariance shape is {self.cov.shape},"
                f"expected {(self.dim, self.dim)}"
            )

    def pdf(self, x: npt.NDArray[np.float64]) -> np.float64:
        """Calculate the probability density function of the distribution."""
        x = x - self.mean
        return (  # type: ignore
            np.exp(-0.5 * x @ np.linalg.inv(self.cov) @ x)
            / np.sqrt((2 * np.pi) ** self.dim * np.linalg.det(self.cov))
        )


MVN: TypeAlias = MultivariateNormalDistribution
