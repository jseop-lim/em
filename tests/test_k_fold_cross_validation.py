import numpy as np
import pytest
import em


def test_generate_dataset_parts() -> None:
    dataset = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
    k = 3

    parts = em.generate_dataset_parts(dataset, k)

    part = next(parts)
    assert part.train == np.array([[3, 4], [5, 6], [7, 8], [9, 10]])
    assert part.validation == np.array([[1, 2]])
    part = next(parts)
    assert part.train == np.array([[1, 2], [5, 6], [7, 8], [9, 10]])
    assert part.validation == np.array([[3, 4]])
    part = next(parts)
    assert part.train == np.array([[1, 2], [3, 4], [7, 8], [9, 10]])
    assert part.validation == np.array([[5, 6]])
    part = next(parts)
    assert part.train == np.array([[1, 2], [3, 4], [5, 6], [9, 10]])
    assert part.validation == np.array([[7, 8]])
    part = next(parts)
    assert part.train == np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    assert part.validation == np.array([[9, 10]])

    pytest.raises(StopIteration, next, parts)
