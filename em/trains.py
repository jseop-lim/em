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

from datetime import datetime
import os
from pathlib import Path

import numpy as np
import numpy.typing as npt

from em import libs

if not (train_data_path_env := os.getenv("TRAIN_DATA_PATH")):
    raise ValueError("TRAIN_DATA_PATH environment variable is not set")

train_data_path = Path(train_data_path_env)
train_dataset = libs.parse_file_to_array(train_data_path)

fold_size = 5
n_classes = 2
n_min_clusters = 2
n_max_clusters = 5


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
    init_sample_size = 100

    return [
        libs.GMMParameter(
            mean=input_dataset[
                np.random.choice(len(input_dataset), init_sample_size)
            ].mean(axis=0),
            cov=np.cov(input_dataset, rowvar=False),
            weight=np.float64(1 / n_clusters),
        )
        for _ in range(n_clusters)
    ]


n_clusters_range = range(n_min_clusters, n_max_clusters + 1)
error_rates_of_n_clusters: dict[int, float] = {}  # {n_clusters: error_rate}

print("Training...")


def train_gmm(
    train_dataset: npt.NDArray[np.float64],
    n_classes: int,
    n_clusters: int,
) -> libs.GaussianMixtureModelClassifier:
    gmm_parameters_list: list[list[libs.GMMParameter]] = []

    for class_k in range(n_classes):
        input_dataset = train_dataset[train_dataset[:, -1] == class_k][:, :-1]
        gmm_parameters_list.append(
            libs.em_algorithm(
                x=input_dataset,
                init_parameters=generate_init_parameters(input_dataset, n_clusters),
                max_iter=2000,
            )
        )

    classifer = libs.GaussianMixtureModelClassifier(n_classes=n_classes)
    classifer.set_known_parameters(parameters_list=gmm_parameters_list)
    return classifer


for n_clusters in n_clusters_range:
    print(f"  Number of Clusters: {n_clusters}")

    error_rates_of_folds: list[float] = []

    for fold_k, fold_train_dataset in enumerate(
        libs.generate_dataset_parts(train_dataset, fold_size)
    ):
        print(f"    Fold: {fold_k + 1}")

        classifer = train_gmm(
            train_dataset=fold_train_dataset.train,
            n_classes=n_classes,
            n_clusters=n_clusters,
        )
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

    for fold_k, error_rate in enumerate(error_rates_of_folds):
        print(f"    Fold {fold_k + 1} Validation Error Rate: {error_rate}")

    error_rates_of_n_clusters[n_clusters] = np.mean(
        np.array(error_rates_of_folds), dtype=float
    )

# write result to file which name is current time
with open(f"result_{datetime.now()}.csv", "w") as f:
    f.write("Number of Clusters,Error Rate\n")
    for n_clusters, error_rate in error_rates_of_n_clusters.items():
        f.write(f"{n_clusters},{error_rate}\n")

libs.plot_line_graphs(
    x=n_clusters_range,
    ys={"": list(error_rates_of_n_clusters.values())},
    x_label="Number of Clusters",
    y_label="Validation Error Rate",
    title="Cross Validation",
)


print("Training is done.")

if not (test_data_path_env := os.getenv("TEST_DATA_PATH")):
    raise ValueError("TEST_DATA_PATH environment variable is not set")

test_data_path = Path(test_data_path_env)
test_dataset = libs.parse_file_to_array(test_data_path)

print("Final Training...")
selected_n_clusters: int = min(
    error_rates_of_n_clusters,
    key=error_rates_of_n_clusters.get,
)

final_classifer = train_gmm(
    train_dataset=train_dataset,
    n_classes=n_classes,
    n_clusters=selected_n_clusters,
)

print("Testing...")
predicted_outputs = final_classifer.predict(
    test_dataset[:, :-1],
    test_dataset[:, -1],  # type: ignore
)
error_rate = libs.calculate_error_rate(
    y_pred=predicted_outputs,
    y_true=test_dataset[:, -1],  # type: ignore
)
print(f"Test Error Rate: {error_rate}")
