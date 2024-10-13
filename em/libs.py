from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple, Sequence, TypeAlias
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt


def plot_line_graphs(
    x: Sequence[int | float],
    ys: dict[str, Sequence[int | float]],
    title: str,
    y_label: str,
    x_label: str,
) -> None:
    """Plot a line graph with multiple lines."""
    for name, y in ys.items():
        plt.plot(x, y, label=name, linestyle="-", marker="o")

    plt.xticks(x)
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


def estimate_gmm_responsibilities(
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

    # Precompute the pdf values for all clusters and data instances
    pdfs = np.array(  # shape: (N, Z)
        [
            [
                MVN(
                    mean=parameters[k].mean,
                    cov=parameters[k].cov,
                    dim=x.shape[1],
                ).pdf(x[t])
                for k in range(n_clusters)
            ]
            for t in range(n_instances)
        ]
    )

    # Extract weights for each cluster
    weights = np.array([p.weight for p in parameters])  # shape: (Z,)

    # Calculate evidence for all instances
    weighted_pdfs = pdfs * weights  # shape: (N, Z)
    evidence = np.sum(weighted_pdfs, axis=1).reshape(-1, 1)  # shape: (N, 1)

    # Calculate responsibilities
    responsibilities: npt.NDArray[np.float64] = weighted_pdfs / evidence

    return responsibilities


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
    n_features = x.shape[1]

    # Total responsibility for each cluster
    responsibility_sums = np.sum(responsibilities, axis=0)

    # Compute the means for each cluster (shape (Z, D))
    means = np.dot(responsibilities.T, x) / responsibility_sums[:, np.newaxis]

    # Compute covariances for each cluster
    covariances = np.zeros((n_clusters, n_features, n_features))

    for k in range(n_clusters):
        # Calculate the deviation of x from the mean
        x_centered = x - means[k]

        # Weighted covariance matrix for each cluster
        covariances[k] = (
            np.dot((responsibilities[:, k][:, np.newaxis] * x_centered).T, x_centered)
            / responsibility_sums[k]
        )

    # Compute weights for each cluster
    weights = responsibility_sums / n_instances

    # Create GMMParameter objects for each cluster
    return [
        GMMParameter(
            mean=means[k],
            cov=covariances[k],
            weight=weights[k],
        )
        for k in range(n_clusters)
    ]

    # n_clusters = responsibilities.shape[1]
    # n_instances = x.shape[0]

    # def get_mean(k: int) -> npt.NDArray[np.float64]:
    #     return np.average(x, axis=0, weights=responsibilities[:, k])  # type: ignore

    # return [
    #     GMMParameter(
    #         mean=get_mean(k),
    #         cov=np.sum(
    #             [
    #                 responsibilities[t, k]
    #                 * np.outer(x[t] - get_mean(k), x[t] - get_mean(k))
    #                 for t in range(n_instances)
    #             ],
    #             axis=0,
    #         )
    #         / np.sum(responsibilities[:, k]),
    #         weight=np.sum(responsibilities[:, k]) / n_instances,
    #     )
    #     for k in range(n_clusters)
    # ]


def em_algorithm(
    x: npt.NDArray[np.float64],
    init_parameters: list[GMMParameter],
    max_iter: int = 100,
    tol: float = 1e-6,
) -> list[GMMParameter]:
    """Run the EM algorithm to estimate the parameters of a Gaussian Mixture Model.

    Args:
        x: Data instances for a class of shape (N, D)
        init_parameters: Initial parameters of the Gaussian Mixture Model for each cluster
        max_iter: The maximum number of iterations
        tol: The tolerance to stop the algorithm

    Returns: Parameters of the Gaussian Mixture Model for each cluster
    """
    print("      EM algorithm")

    parameters = init_parameters
    for _ in range(max_iter):
        print(f"        Iteration: {_ + 1}")
        responsibilities = estimate_gmm_responsibilities(x, parameters)
        new_parameters = estimate_gmm_parameters(x, responsibilities)

        if all(
            np.linalg.norm(new_parameter.mean - parameter.mean) < tol
            and np.linalg.norm(new_parameter.cov - parameter.cov) < tol
            and abs(new_parameter.weight - parameter.weight) < tol
            for new_parameter, parameter in zip(new_parameters, parameters)
        ):
            break

        parameters = new_parameters

    return parameters


#### 예측 모델 ####


class GaussianMixtureModelClassifier:
    def __init__(
        self,
        # input_dataset: npt.NDArray[np.float64],
        # output_dataset: npt.NDArray[np.float64],
        n_classes: int,
    ) -> None:
        # self.input_dataset = input_dataset
        # self.output_dataset = output_dataset
        self.n_classes = n_classes
        self.parameters_list: list[list[GMMParameter]] = []

    def set_known_parameters(self, parameters_list: list[list[GMMParameter]]) -> None:
        self.parameters_list = parameters_list

    def predict(
        self,
        x_set: npt.NDArray[np.float64],
        y_set: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.int64]:
        """Predict the class of new data instances with bayesian decision rule.

        Args:
            x: Data instances of shape (N, D)

        Returns: Predicted classes of shape (N,)
        """
        return np.argmax(self.likelihood(x_set) * self.prior(y_set), axis=1)  # type: ignore

    def likelihood(self, x_set: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Calculate the likelihood of each data instance for each class.

        Args:
            x_set: Data instances of shape (N, D)

        Returns: Likelihood of each data instance for each class of shape (N, K)
        """
        n_features = x_set.shape[1]

        return np.array(
            [
                [
                    np.prod(
                        [
                            MVN(
                                mean=parameter.mean,
                                cov=parameter.cov,
                                dim=n_features,
                            ).pdf(x)
                            * parameter.weight
                            for parameter in parameters
                        ]
                    )
                    for x in x_set
                ]
                for parameters in self.parameters_list
            ]
        ).T

    def prior(self, y_set: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Calculate the prior probability of each class.

        Returns: Prior probability of each class of shape (K,)
        """
        return np.array(
            [np.sum(y_set == k) / y_set.shape[0] for k in range(self.n_classes)]
        )


def calculate_error_rate(
    y_true: npt.NDArray[np.int64], y_pred: npt.NDArray[np.int64]
) -> float:
    """Calculate the error rate of the prediction.

    Args:
        y_true: True classes of shape (N,)
        y_pred: Predicted classes of shape (N,)

    Returns: Error rate
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape is {y_true.shape}, y_pred shape is {y_pred.shape}"
        )

    n_instances = y_true.shape[0]
    return float(np.sum(y_true != y_pred, dtype=np.float64) / n_instances)
