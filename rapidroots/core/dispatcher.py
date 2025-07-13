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
        modules = ['math', 'numpy']
        
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
                    params_array:   np.ndarray,
                    x_array:        np.ndarray,
                    derivative:     int = 0,
                    chunk_size:     int = 10000
                    ) -> np.ndarray:
        n = x_array.size
        results = np.empty(n, dtype=np.float64)

        # Process in fixed-size chunks
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            slice_types  = func_types[start:end]
            slice_params = params_array[start:end]
            slice_x      = x_array[start:end]

            # Your existing dispatch logic, but on the slice:
            slice_res = np.empty(end - start, dtype=np.float64)
            for name, family in self.compiled_functions.items():
                mask = (slice_types == name)
                if not mask.any():
                    continue

                ufunc = {0: family['f'],
                        1: family['df'],
                        2: family['d2f']}[derivative]
                pcount = family['param_count']

                xs = slice_x[mask]
                ps = slice_params[mask]  # shape (m, pcount)
                args = (xs, *[ps[:, i] for i in range(pcount)])
                slice_res[mask] = ufunc(*args)

            results[start:end] = slice_res

        return results

if __name__ == "__main__":

    # Build dispatcher
    dispatcher = UniversalFuntionSympyDispatcher()

    # 1) define the symbol
    x = sp.Symbol('x')

    # 2) Register a quadratic family: f(x) = a*x^2 + b*x + c
    a, b, c = sp.symbols('a b c')
    quad_expr = a*x**2 + b*x + c
    dispatcher.register_symbolic_family('quadratic', quad_expr, (a, b, c))

    # 3) Register a sine family: f(x) = A*sin(B*x + C)
    A, B, C = sp.symbols('A B C')
    sine_expr = A*sp.sin(B*x + C)
    dispatcher.register_symbolic_family('sine', sine_expr, (A, B, C))

    # 2) Register a quadratic family: f(x) = a*x^2 + b*x + c
    a, b, c = sp.symbols('a b c')
    quad_expr = a*x**2 + b*x + c
    dispatcher.register_symbolic_family('quadratic', quad_expr, (a, b, c))

    # 3) Register a sine family: f(x) = A*sin(B*x + C)
    A, B, C = sp.symbols('A B C')
    sine_expr = A*sp.sin(B*x + C)
    dispatcher.register_symbolic_family('sine', sine_expr, (A, B, C))

    # 4) Create sample data
    N = 10
    # Alternate families in the batch
    func_types = np.array(['quadratic', 'sine'] * (N//2))
    x_values   = np.linspace(0, 2*np.pi, N)
    # For quadratics: a=1, b=0.5, c=2; for sines: A=2, B=1, C=0
    params = []
    for name in func_types:
        if name == 'quadratic':
            params.append([1.0, 0.5, 2.0])
        else:
            params.append([2.0, 1.0, 0.0])
    params_array = np.array(params, dtype=np.float64)

    # 5) Evaluate f(x)
    y0 = dispatcher.evaluate_batch(func_types, params_array, x_values, derivative=0)
    print("f(x):", y0)

    # 6) Evaluate f'(x)
    y1 = dispatcher.evaluate_batch(func_types, params_array, x_values, derivative=1)
    print("f'(x):", y1)

    # 7) Evaluate f''(x)
    y2 = dispatcher.evaluate_batch(func_types, params_array, x_values, derivative=2)
    print("f''(x):", y2)