import sympy as sp
import numpy as np
from numba import vectorize, float64
from typing import Dict, List, Callable, Tuple


class UniversalFuntionSympyDispatcher:
    
    def __init__(self):
        # Will hold, for each name, a dict with keys 'f','df','d2f','param_syms'
        self.compiled_functions: Dict[str, Dict[str, Callable]] = {}
    
    def register_symbolic_family(self,
                                 name: str,
                                 expr: sp.Expr,
                                 param_syms: Tuple[sp.Symbol, ...]):
        """
        Register a new family:
          name      – arbitrary key
          expr      – SymPy expression f(x, *params)
          param_syms– tuple of SymPy symbols (a, b, c, ...) matching expr’s parameters
        """
        # symbol for the variable
        x = sp.Symbol('x')
        
        # build f, f' and f''
        df_expr  = sp.diff(expr, x)
        d2f_expr = sp.diff(df_expr, x)

        # use these modules
        modules = ['math', ['numpy']]
        
        # lambdify to Python functions taking numpy arrays
        f_py   = sp.lambdify((x, *param_syms), expr,    modules)
        df_py  = sp.lambdify((x, *param_syms), df_expr,  modules)
        d2f_py = sp.lambdify((x, *param_syms), d2f_expr, modules)
        
        # now wrap them in true Numba ufuncs
        sig = [float64(float64, *([float64] * len(param_syms)))]
        f_ufunc   = vectorize(sig, target='parallel', fastmath=True)(f_py)
        df_ufunc  = vectorize(sig, target='parallel', fastmath=True)(df_py)
        d2f_ufunc = vectorize(sig, target='parallel', fastmath=True)(d2f_py)
        
        # store them
        self.compiled_functions[name] = {
            'f':   f_ufunc,
            'df':  df_ufunc,
            'd2f': d2f_ufunc,
            'param_count': len(param_syms)
        }


    def evaluate_batch(self,
                       func_types:     np.ndarray,
                       params_array:  np.ndarray,
                       x_array:       np.ndarray,
                       derivative:    int = 0
                      ) -> np.ndarray:
        """
        Dispatch batch evaluation:
          func_types   – array of strings naming the family for each x
          params_array – shape (n, param_count[name]) of parameters
          x_array      – shape (n,) of x values
          derivative   – 0=f, 1=df, 2=d2f
        """
        n = x_array.size
        results = np.empty(n, dtype=np.float64)
        
        for name, family in self.compiled_functions.items():
            mask = (func_types == name)
            if not mask.any():
                continue
            
            ufunc = {0: family['f'],
                     1: family['df'],
                     2: family['d2f']}[derivative]
            pcount = family['param_count']
            
            # slice out this batch
            xs = x_array[mask]
            ps = params_array[mask]   # shape (m, pcount)
            
            # Numba ufuncs accept each parameter vector separately
            args = (xs, *[ps[:, i] for i in range(pcount)])
            results[mask] = ufunc(*args)
        
        return results
