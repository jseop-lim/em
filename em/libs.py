from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Sequence
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


class GMMParameter(NamedTuple):
    """Parameters of a Gaussian Mixture Model for a single cluster."""

    mean: npt.NDArray[np.float64]  # mu_z
    cov: npt.NDArray[np.float64]  # sigma_z
    weight: np.float64  # pi_z


def calculate_mvn_pdfs(
    x: npt.NDArray[np.float64],
    parameters: list[GMMParameter],
) -> npt.NDArray[np.float64]:
    """Calculate the probability density functions of each cluster for each data instance.

    Args:
        x: Data instances for a class of shape (N, D)
        parameters: Parameters of the Gaussian Mixture Model for each cluster of shape (Z,)

    Returns: Probability density functions of each cluster for each data instance of shape (N, Z)
        matrix P = [
            [p_1^1, p_2^1, ..., p_Z^1],
            [p_1^2, p_2^2, ..., p_Z^2],
            ...,
            [p_1^N, p_2^N, ..., p_Z^N],
        ]
        where p_z^t is the probability density of cluster z for data instance t
    """
    if x.ndim != 2:
        raise ValueError(f"Data instances must have 2 dimensions, got {x.ndim}")

    _, n_features = x.shape

    # 클러스터의 평균 및 공분산 행렬 가져오기
    means = np.array([p.mean for p in parameters])  # (Z, D)
    covariances = np.array([p.cov for p in parameters])  # (Z, D, D)

    # 공분산 행렬의 행렬식과 역행렬 구하기
    det_cov = np.linalg.det(covariances)  # (Z,)
    inv_cov = np.linalg.inv(covariances)  # (Z, D, D)

    # 데이터 포인트와 평균의 차이 계산, 브로드캐스팅을 통해 (N, Z, D) 크기로 만듦
    diff = x[:, np.newaxis, :] - means[np.newaxis, :, :]  # (N, Z, D)

    # Mahalanobis 거리 계산: (x - μ)^T Σ^{-1} (x - μ)
    # np.einsum에서 (N, Z, D)와 (Z, D, D)를 곱하여 (N, Z) 형태로 축소해야 합니다.
    mahalanobis_term = np.einsum("nzd,zde,nze->nz", diff, inv_cov, diff)  # (N, Z)

    # 정규화 상수 계산
    normalization_term = 1 / np.sqrt((2 * np.pi) ** n_features * det_cov)  # (Z,)
    normalization_term = normalization_term[
        np.newaxis, :
    ]  # (1, Z)로 확장하여 브로드캐스팅 준비

    # 지수 항 계산
    exp_term = np.exp(-0.5 * mahalanobis_term)  # (N, Z)

    # 최종 확률밀도 계산
    pdf_values: npt.NDArray[np.float64] = normalization_term * exp_term  # (N, Z)

    return pdf_values


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
    # Precompute the pdf values for all clusters and data instances
    pdfs = calculate_mvn_pdfs(x, parameters)  # shape: (N, Z)

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
    for i in range(max_iter):
        responsibilities = estimate_gmm_responsibilities(x, parameters)
        new_parameters = estimate_gmm_parameters(x, responsibilities)

        if all(
            np.linalg.norm(new_parameter.mean - parameter.mean) < tol
            and np.linalg.norm(new_parameter.cov - parameter.cov) < tol
            and abs(new_parameter.weight - parameter.weight) < tol
            for new_parameter, parameter in zip(new_parameters, parameters)
        ):
            parameters = new_parameters
            break

        parameters = new_parameters

    print(f"      Iterations: {i + 1}")
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
        self.parameters_list: list[
            list[GMMParameter]
        ] = []  # parameteres for each class

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
        return np.argmax(  # type: ignore
            self.log_likelihood(x_set, y_set) + self.log_prior(y_set),
            axis=1,
        )

    def log_likelihood(
        self,
        x_set: npt.NDArray[np.float64],
        y_set: npt.NDArray[np.int64],
    ) -> npt.NDArray[np.float64]:
        """

        Returns:
            (N, K)
        """
        log_likelihood = []

        for class_k in range(self.n_classes):
            pdfs = calculate_mvn_pdfs(
                x_set, self.parameters_list[class_k]
            )  # 모든 instance x에 대한 모든 group의 pdf shape of (N, Z)
            weights = np.array(
                [parameter.weight for parameter in self.parameters_list[class_k]]
            )  # (Z,)
            log_likelihood_k: npt.NDArray[np.float64] = np.log(
                np.dot(pdfs, weights)
            )  # (N,)
            log_likelihood.append(log_likelihood_k)

        return np.array(log_likelihood).T

    def log_prior(self, y_set: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
        """Calculate the prior probability of each class.

        Returns: Prior probability of each class of shape (K,)
        """
        return np.log(
            np.array(
                [np.sum(y_set == k) / y_set.shape[0] for k in range(self.n_classes)]
            ),
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
