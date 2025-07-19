import sympy as sp
import numpy as np
import numpy.typing as npt

from typing import Tuple, Optional
from typing_extensions import Literal
from numba import vectorize, float64, float32
from .base import FunctionFamily

class SympyFamily(FunctionFamily):

    def __init__(self, name: str, expr: sp.Expr, param_syms: Tuple[sp.Symbol, ...]):
        self._name = name
        x = sp.Symbol('x')
        df = sp.diff(expr, x)
        d2f = sp.diff(df, x)
        modules = ['numpy', 'math']
        f_py = sp.lambdify((x, *param_syms), expr,      modules)
        df_py = sp.lambdify((x, *param_syms), df,       modules)
        d2f_py = sp.lambdify((x, *param_syms), d2f,     modules)

        sig = [
            float32(float32, *([float32]*len(param_syms))),
            float64(float64, *([float64]*len(param_syms))),
            ]

        self._ufuncs = {
            0: vectorize(sig, target='parallel', fastmath=True)(f_py),
            1: vectorize(sig, target='parallel', fastmath=True)(df_py),
            2: vectorize(sig, target='parallel', fastmath=True)(d2f_py)
        }

        self._param_count = len(param_syms)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def param_count(self) -> int:
        return self._param_count

    def evaluate(self, x: npt.ArrayLike , params_arr: npt.ArrayLike, derivative: Literal[0, 1, 2] = 0, dtype: Optional[npt.DTypeLike] = np.float64) -> npt.NDArray:
        
        x_arr = np.asarray(x, dtype=dtype)
        params = np.asarray(params_arr, dtype=dtype)


        ufunc = self._ufuncs[derivative]

        if params.ndim != 2 or params.shape[1] != self._param_count:
            raise ValueError(
                f"params_arr must be shape (n, {self._param_count}), got {params.shape}"
            )

        # unpack parameters once via transpose
        # params.T is shape (param_count, n)
        param_args = tuple(params.T)

        # one ufunc call over the entire array
        return ufunc(x_arr, *param_args)