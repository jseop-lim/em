from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from typing import NamedTuple
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_line_graphs(
    y_list: list[list],
    x: list,
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    """Plot a line graph with multiple lines."""
    if all(chain((len(y) for y in y_list))):
        raise ValueError("All y values must have the same length")
    if len(x) != len(y_list[0]):
        raise ValueError("x and y values must have the same length")

    for y in y_list:
        plt.plot(x, y)
    plt.title(title)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
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
    parts: list[npt.NDArray[np.float64]] = split_dataset_into_parts(dataset, k)

    return (
        TrainDataset(
            train=np.concatenate(parts[:i] + parts[i + 1 :], axis=0),
            validation=parts[i],
        )
        for i in range(k)
    )
