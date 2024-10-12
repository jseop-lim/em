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
train_dataset = libs.parse_file_to_array(train_data_path)

fold_size = 10
n_classes = 2


def generate_init_parameters(
    input_dataset: npt.NDArray[np.float64],
    n_clusters: int,
) -> list[libs.GMMParameter]:
    return [
        libs.GMMParameter(
            mean=np.mean(input_dataset, axis=0),
            cov=np.cov(input_dataset, ddof=0, rowvar=False),
            weight=np.float64(1 / n_clusters),
        )
        for _ in range(n_clusters)
    ]


n_clusters_range = range(2, 11)
error_rates_of_n_clusters: list[float] = []

for n_clusters in n_clusters_range:
    error_rates_of_folds: list[float] = []

    for fold_k, fold_train_dataset in enumerate(
        libs.generate_dataset_parts(train_dataset, fold_size)
    ):

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
        predicted_outputs = classifer.predict(fold_train_dataset.validation[:, :-1])

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
