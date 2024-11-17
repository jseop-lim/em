import numpy as np

from em.libs import GMMParameter, estimate_gmm_responsibilities


def test_estimate_responsibilities() -> None:
    x = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    parameters = [
        GMMParameter(
            weight=1 / 3,
            mean=np.array([0.0, 0.0]),
            cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
        GMMParameter(
            weight=1 / 3,
            mean=np.array([1.0, 1.0]),
            cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
        GMMParameter(
            weight=1 / 3,
            mean=np.array([2.0, 2.0]),
            cov=np.array([[1.0, 0.0], [0.0, 1.0]]),
        ),
    ]

    expected = np.array(
        [
            [0.39894228, 0.39894228, 0.39894228],
            [0.39894228, 0.39894228, 0.39894228],
            [0.39894228, 0.39894228, 0.39894228],
        ]
    )

    assert np.allclose(estimate_gmm_responsibilities(x, parameters), expected)
