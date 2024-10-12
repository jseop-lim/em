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

    def pdf(self, x: npt.NDArray[np.float64]) -> float:
        """Calculate the probability density function of the distribution."""
        x = x - self.mean
        return (  # type: ignore
            np.exp(-0.5 * x @ np.linalg.inv(self.cov) @ x)
            / np.sqrt((2 * np.pi) ** self.dim * np.linalg.det(self.cov))
        )


MVN: TypeAlias = MultivariateNormalDistribution


class GMMParameter(NamedTuple):
    """Parameters of a Gaussian Mixture Model for a single cluster."""

    mean: npt.NDArray[np.float64]  # mu_z
    cov: npt.NDArray[np.float64]  # sigma_z
    weight: np.float64  # pi_z


def estimate_responsibilities(
    x: npt.NDArray[np.float64],
    parameters: list[GMMParameter],
) -> npt.NDArray[np.float64]:
    """Estimate the responsibilities of each cluster for each data instance.

    Args:
        x: Data instances for a class of shape (N, D)
        parameters: Parameters of the Gaussian Mixture Model for each cluster

    Returns: A responsibility of shape (N, Z)
        matrix W = [
            [w_1^1, w_2^1, ..., w_Z^1],
            [w_1^2, w_2^2, ..., w_Z^2],
            ...,
            [w_1^N, w_2^N, ..., w_Z^N],
        ]
        where w_z^t is the responsibility of cluster z for data instance t
    """
    n_clusters = len(parameters)
    n_instances = x.shape[0]

    return np.array(
        [
            [
                parameters[k].weight
                * MVN(
                    mean=parameters[k].mean,
                    cov=parameters[k].cov,
                    dim=x.shape[1],
                ).pdf(x[t])
                / evidence_t
                for k in range(n_clusters)
                if (
                    evidence_t := sum(
                        parameters[j].weight
                        * MVN(
                            mean=parameters[j].mean,
                            cov=parameters[j].cov,
                            dim=x.shape[1],
                        ).pdf(x[t])
                        for j in range(n_clusters)
                    )
                )
            ]
            for t in range(n_instances)
        ]
    )


def estimate_gmm_parameters(
    x: npt.NDArray[np.float64],
    responsibilities: npt.NDArray[np.float64],
) -> list[GMMParameter]:
    """Estimate the parameters of a Gaussian Mixture Model for each cluster.

    Args:
        x: Data instances for a class of shape (N, D)
        responsibilities: Responsibilities of each cluster for each data instance
            of shape (N, Z)

    Returns: Parameters of the Gaussian Mixture Model for each cluster of shape (Z,)
    """
    n_clusters = responsibilities.shape[1]
    n_instances = x.shape[0]

    return [
        GMMParameter(
            mean=mean,
            cov=np.cov(x.T, aweights=responsibilities[:, k], ddof=0)
            - np.outer(mean, mean),
            weight=np.sum(responsibilities[:, k]) / n_instances,
        )
        for k in range(n_clusters)
        if (mean := np.average(x, axis=0, weights=responsibilities[:, k]))
    ]
