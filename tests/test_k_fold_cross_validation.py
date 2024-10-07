import numpy as np
import pytest
import em


def test_split_dataset_into_parts() -> None:
    dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 3
    expected_parts = [
        np.array([[1, 2], [3, 4]]),
        np.array([[5, 6], [7, 8]]),
        np.array([[9, 10]]),
    ]

    parts = em.split_dataset_into_parts(dataset, k)

    assert len(parts) == 3
    for part, expected_part in zip(parts, expected_parts):
        assert np.array_equal(part, expected_part)


def test_generate_dataset_parts() -> None:
    dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 3
    expected_parts = [
        em.TrainDataset(
            train=np.array([[5, 6], [7, 8], [9, 10]]),
            validation=np.array([[1, 2], [3, 4]]),
        ),
        em.TrainDataset(
            train=np.array([[1, 2], [3, 4], [9, 10]]),
            validation=np.array([[5, 6], [7, 8]]),
        ),
        em.TrainDataset(
            train=np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            validation=np.array([[9, 10]]),
        ),
    ]

    parts = em.generate_dataset_parts(dataset, k)

    for part, expected_part in zip(parts, expected_parts):
        assert np.array_equal(part.train, expected_part.train)
        assert np.array_equal(part.validation, expected_part.validation)

    with pytest.raises(StopIteration):
        next(parts)
