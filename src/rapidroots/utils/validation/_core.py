import numpy as np
import numpy.typing as npt

from typing import Tuple, Optional, Union

ArrayLike = npt.ArrayLike
DTypeLike = Union[npt.DTypeLike, type]

def _validate_bracket_core(
        a: ArrayLike,
        b: ArrayLike,
        fa: ArrayLike,
        fb: ArrayLike,
        *,
        tolerance: float = 0.0,
        dtype: Optional[DTypeLike] = np.float64,
) -> Tuple[npt.NDArray[np.bool_], npt.NDArray[np.int8]]:
    """
    Vector core Bracket validator 
    """

    if dtype is None:
        a_arr = np.asarray(a)
        b_arr = np.asarray(b)
        fa_arr = np.asarray(fa)
        fb_arr = np.asarray(fb)

    else: 
        dt = np.dtype(dtype)
        a_arr = np.asarray(a, dtype=dt)
        b_arr = np.asarray(b, dtype=dt)
        fa_arr = np.asarray(fa, dtype=dt)
        fb_arr = np.asarray(fb, dtype=dt)
    
    if not (a_arr.shape == b_arr.shape == fa_arr.shape == fb_arr.shape):
        raise ValueError ("Validate_bracket_core: shape mismatch")
    
    ordered = a_arr < b_arr
    finite = ~(
        np.isinf(a_arr) |
        np.isinf(b_arr) |
        np.isinf(fa_arr) |
        np.isinf(fb_arr)
    )

    if tolerance > 0.0:
        fa0 = np.abs(fa_arr) <= tolerance
        fb0 = np.abs(fb_arr) <= tolerance
    
    else:
        fa0 = fa_arr == 0.0
        fb0 = fb_arr == 0.0
    
    opp = np.signbit(fa_arr) ^ np.signbit(fb_arr)
    has_zero = fa0 | fb0
    bracket_ok = opp | has_zero

    is_valid = ordered & finite & bracket_ok

    reason = np.zeros(a_arr.shape, dtype=np.int8)
    m = ~ordered
    reason[m] = 1
    m = (~finite) & (reason == 0)
    reason[m] = 2
    m = (~bracket_ok) & (reason == 0)
    reason[m] = 3

    return is_valid, reason

