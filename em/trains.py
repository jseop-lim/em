"""
TRAIN_DATA_PATH 환경변수로부터 파일경로를 가져옵니다.
파일경로에서 데이터를 읽습니다. parse_file_to_array()를 호출합니다.
데이터를 k개의 부분으로 나눕니다. generate_dataset_parts()를 호출합니다.
각 부분에 대해 모델을 학습하고, 검증합니다.
   (각 부분에 대해)
   학습 데이터를 클래스별로 나눕니다.
   클래스마다 EM 알고리즘을 사용해 모델을 학습합니다. 최종 GMMParameters를 반환합니다.
   검증 데이터에 대해 예측을 수행합니다.
   예측 결과를 저장합니다.
"""

import os
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from em import libs

if not (train_data_path_env := os.getenv("TRAIN_DATA_PATH")):
    raise ValueError("TRAIN_DATA_PATH environment variable is not set")

train_data_path = Path(train_data_path_env)
train_dataset = libs.parse_file_to_array(train_data_path)[:1000, :]

fold_size = 10
n_classes = 2


def generate_init_parameters(
    input_dataset: npt.NDArray[np.float64],
    n_clusters: int,
) -> list[libs.GMMParameter]:
    """EM 알고리즘을 위한 초기 파라미터를 생성합니다.

    Args:
        input_dataset: (n_samples, n_features) 크기의 2차원 배열
        n_clusters: 클러스터의 개수

    Returns:
        n_clusters개의 GMMParameter 객체를 원소로 갖는 리스트
    """
    return [
        libs.GMMParameter(
            mean=np.mean(input_dataset, axis=0),
            cov=np.outer(
                input_dataset - np.mean(input_dataset, axis=0),
                input_dataset - np.mean(input_dataset, axis=0),
            )
            / (input_dataset.shape[0]),
            weight=np.float64(1 / n_clusters),
        )
        for _ in range(n_clusters)
    ]


n_clusters_range = range(2, 3)
error_rates_of_n_clusters: list[float] = []

print("Training...")

for n_clusters in n_clusters_range:
    print(f"  Number of Clusters: {n_clusters}")

    error_rates_of_folds: list[float] = []

    for fold_k, fold_train_dataset in enumerate(
        libs.generate_dataset_parts(train_dataset, fold_size)
    ):
        print(f"    Fold: {fold_k + 1}")

        def get_input_dataset(class_k: int) -> Any:
            return fold_train_dataset.train[fold_train_dataset.train[:, -1] == class_k]

        gmm_parameters_list = [
            libs.em_algorithm(
                get_input_dataset(class_k),
                generate_init_parameters(get_input_dataset(class_k), n_clusters),
                n_clusters,
            )
            for class_k in range(n_classes)
        ]
        classifer = libs.GaussianMixtureModelClassifier(n_classes=n_classes)
        classifer.set_known_parameters(parameters_list=gmm_parameters_list)
        predicted_outputs = classifer.predict(
            fold_train_dataset.validation[:, :-1],
            fold_train_dataset.validation[:, -1],  # type: ignore
        )

        error_rates_of_folds.append(
            libs.calculate_error_rate(
                y_pred=predicted_outputs,
                y_true=fold_train_dataset.validation[:, -1],  # type: ignore
            )
        )

    error_rates_of_n_clusters.append(
        sum(error_rates_of_folds) / len(error_rates_of_folds)
    )


libs.plot_line_graphs(
    x=n_clusters_range,
    functions=lambda z: error_rates_of_n_clusters[z],  # type: ignore
    x_label="Number of Clusters",
    y_label="Error Rate",
    title="Error Rate vs. Number of Clusters",
)
