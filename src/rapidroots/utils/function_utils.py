import numpy as np
from typing import Callable
from numpy.typing import ArrayLike

def condition_function(
    f: Callable[[ArrayLike], np.ndarray],
    inf_replacement: float = 1e10,
    zero_threshold: float = 1e-15
) -> Callable[[ArrayLike], np.ndarray]:
    """
    Wrap any function f(x) so that:
     - exceptions → NaN array
     - ±inf       → ±inf_replacement
     - |x|<zero_threshold → 0
     - other NaNs pass through
    """
    def conditioned(x):
        x = np.asarray(x, float)
        try:
            y = f(x)
            y = np.asarray(y, dtype=float)

            # Ensure y is an array for item assignment
            if np.isscalar(y) or y.ndim == 0:
                y = np.array(y, dtype=float, ndmin=1)
        except Exception:
            return np.full_like(x, np.nan)

        # sanitize infinities & NaNs
        y = np.nan_to_num(
            y,
            nan=np.nan,
            posinf=inf_replacement,
            neginf=-inf_replacement
        )
        # snap tiny
        y[np.abs(y) < zero_threshold] = 0.0
        return y.reshape(x.shape)

    return conditioned
