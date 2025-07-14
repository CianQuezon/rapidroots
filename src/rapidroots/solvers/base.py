"""
Improved architecture for root-finding methods in psychrometric calculations.

This design separates different categories of root-finding methods based on their 
fundamental characteristics and interface requirements.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple, Union, Optional
import numpy as np
import numpy.typing as npt
import logging


class VectorizationMixin:
    """
    Shared mixin for parameter extraction and chunked processing.
    
    This provides common vectorization utilities that all method types can use.
    """
    
    def _extract_array_params(self, **kwargs) -> dict:
        """Extract array-like parameters from kwargs"""
        array_params = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                array_params[key] = np.asarray(value)
            else:
                array_params[key] = value
        return array_params
    
    def _get_batch_size(self, array_params: dict) -> int:
        """Determine batch size from array parameters"""
        sizes = []
        for key, value in array_params.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                sizes.append(len(value))
        
        if not sizes:
            raise ValueError("No array parameters found for vectorized solve")
        
        if len(set(sizes)) > 1:
            raise ValueError(f"Inconsistent array sizes: {sizes}")
        
        return sizes[0]
    
    def _extract_chunk_params(self, array_params: dict, start: int, end: int) -> dict:
        """Extract parameters for a specific chunk"""
        chunk_params = {}
        for key, value in array_params.items():
            if isinstance(value, np.ndarray) and value.ndim > 0:
                chunk_params[key] = value[start:end]
            else:
                chunk_params[key] = value
        return chunk_params


class ConvergenceBase(ABC, VectorizationMixin):
    """
    Abstract base class for all convergence algorithms.
    
    This provides the most general interface that all root-finding methods
    must implement, regardless of their specific category.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize convergence algorithm.
        
        Parameters
        ----------
        logger : logging.Logger, optional
            Logger for error propagation and diagnostics
        """
        self.logger = logger or logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def solve(
        self,
        f_func: Callable[[npt.ArrayLike], npt.ArrayLike],
        tolerance: float = 0.01,
        max_iterations: int = 50,
        raise_on_fail: bool = False,
        **kwargs: Any
    ) -> Tuple[Union[float, npt.NDArray[np.floating]], 
               Union[bool, npt.NDArray[np.bool_]]]:
        """
        Solve for root(s) of f(x) = 0 using the specific algorithm.
        
        This is the most general interface that all methods must support.
        Specific method categories will have more specialized interfaces.
        
        Parameters
        ----------
        f_func : callable
            Function for which to find roots. Must accept scalar or array input
            and return corresponding scalar or array output.
        tolerance : float, default=0.01
            Convergence tolerance
        max_iterations : int, default=50
            Maximum iterations
        raise_on_fail : bool, default=False
            If True, raise exception on convergence failure. If False, return
            convergence status in second return value.
        **kwargs : dict
            Method-specific parameters (see concrete implementations)
            
        Returns
        -------
        solution : float or ndarray
            Root(s) found
        converged : bool or ndarray
            Convergence status (only returned if raise_on_fail=False)
        """
        pass
    
    def solve_vectorized(
        self,
        f_func: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        tolerance: float = 0.01,
        max_iterations: int = 50,
        raise_on_fail: bool = False,
        chunk_size: int = 1000,
        enable_logging: bool = True,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
        """
        Vectorized solve for multiple initial conditions simultaneously.
        
        See `solve` for parameter meanings; this parallelizes many calls for
        efficiency in psychrometric calculations with multiple state points.
        
        Note: Concrete implementations should override this method with optimized
        vectorized algorithms (e.g., Numba @njit(parallel=True)) for best performance
        on large datasets (30M+ points).
        
        Parameters
        ----------
        f_func : callable
            Vectorized function that accepts array input and returns array output
        chunk_size : int, default=1000
            Process data in chunks to manage memory for large arrays
        enable_logging : bool, default=True
            Enable per-point error logging for diagnostics
        **kwargs : dict
            Method-specific vectorized parameters (e.g., x0_array, a_array, b_array)
            
        Returns
        -------
        solutions : ndarray
            Array of roots found
        converged : ndarray of bool
            Array of convergence status for each initial condition
        """
        # Default implementation: chunked vectorized fallback
        return self._vectorized_fallback(
            f_func, tolerance, max_iterations, raise_on_fail, 
            chunk_size, enable_logging, **kwargs
        )
    
    def _vectorized_fallback(
        self,
        f_func: Callable[[npt.NDArray[np.floating]], npt.NDArray[np.floating]],
        tolerance: float,
        max_iterations: int,
        raise_on_fail: bool,
        chunk_size: int,
        enable_logging: bool,
        **kwargs: Any
    ) -> Tuple[npt.NDArray[np.floating], npt.NDArray[np.bool_]]:
        """
        Default vectorized implementation using chunked scalar calls.
        
        This provides batch capability out of the box for any new method.
        Concrete implementations should override solve_vectorized for better performance.
        """
        # Extract array parameters and determine batch size
        array_params = self._extract_array_params(**kwargs)
        n_points = self._get_batch_size(array_params)
        
        # Initialize result arrays
        solutions = np.empty(n_points, dtype=np.float64)
        converged = np.empty(n_points, dtype=bool)
        error_messages = {} if enable_logging else None
        
        # Process in chunks
        for start in range(0, n_points, chunk_size):
            end = min(start + chunk_size, n_points)
            
            # Extract chunk parameters
            chunk_kwargs = self._extract_chunk_params(array_params, start, end)
            
            # Create chunk function
            def chunk_f_func(x_chunk):
                # Expand scalar x to match chunk size if needed
                if np.isscalar(x_chunk):
                    x_full = np.full(end - start, x_chunk)
                else:
                    x_full = np.zeros(n_points)
                    x_full[start:end] = x_chunk
                
                result_full = f_func(x_full)
                return result_full[start:end]
            
            # Solve for each point in chunk
            for i, local_idx in enumerate(range(end - start)):
                global_idx = start + local_idx
                
                try:
                    # Extract scalar parameters for this point
                    point_kwargs = {k: (v[local_idx] if isinstance(v, np.ndarray) else v) 
                                  for k, v in chunk_kwargs.items()}
                    
                    # Create point-specific function
                    def point_f_func(x_scalar):
                        x_point = np.full(n_points, x_scalar)
                        result_full = f_func(x_point)
                        return result_full[global_idx]
                    
                    # Solve for this point
                    result, conv = self.solve(
                        point_f_func, tolerance, max_iterations, 
                        raise_on_fail=False, **point_kwargs
                    )
                    
                    solutions[global_idx] = result
                    converged[global_idx] = conv
                    
                    # Log convergence failure
                    if not conv and enable_logging:
                        error_msg = f"Convergence failed at point {global_idx}"
                        self.logger.warning(error_msg)
                        if error_messages is not None:
                            error_messages[global_idx] = error_msg
                    
                except Exception as e:
                    solutions[global_idx] = np.nan
                    converged[global_idx] = False
                    
                    # Log exception
                    if enable_logging:
                        error_msg = f"Exception at point {global_idx}: {str(e)}"
                        self.logger.error(error_msg)
                        if error_messages is not None:
                            error_messages[global_idx] = error_msg
        
        # Log summary statistics
        if enable_logging:
            n_failed = (~converged).sum()
            success_rate = (n_points - n_failed) / n_points * 100
            self.logger.info(f"Vectorized solve completed: {success_rate:.1f}% success rate "
                           f"({n_points - n_failed}/{n_points} points)")
        
        if raise_on_fail and not converged.all():
            failed_indices = np.where(~converged)[0]
            raise RuntimeError(f"Convergence failed for {len(failed_indices)} points")
        
        return solutions, converged


class OpenMethodBase(ConvergenceBase):
    """
    Abstract base class for open root-finding methods.
    
    These methods require an initial guess and optionally use derivatives.
    Examples: Newton-Raphson, Secant, Fixed-point iteration.
    
    Performance Note: Concrete implementations should override solve_vectorized
    with @njit(parallel=True) decorated methods for optimal performance.
    """
    
    @abstractmethod
    def solve(
        self,
        f_func: Callable[[npt.ArrayLike], npt.ArrayLike],
        x0: Union[float, npt.NDArray[np.floating]],
        tolerance: float = 0.01,
        max_iterations: int = 50,
        raise_on_fail: bool = False,
        df_func: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None,
        **kwargs: Any
    ) -> Tuple[Union[float, npt.NDArray[np.floating]], 
               Union[bool, npt.NDArray[np.bool_]]]:
        """
        Solve using open method with initial guess.
        
        Parameters
        ----------
        x0 : float or ndarray
            Initial guess(es)
        df_func : callable, optional
            Derivative function (required for Newton-type methods)
        **kwargs : dict
            Additional method-specific parameters
        """
        pass


class BracketingMethodBase(ConvergenceBase):
    """
    Abstract base class for bracketing root-finding methods.
    
    These methods require an interval [a, b] where f(a) and f(b) have opposite signs.
    Examples: Bisection, Brent's method, Ridder's method.
    
    Performance Note: For 30M+ points, override solve_vectorized with NumPy ufuncs
    or Numba-compiled kernels for maximum throughput.
    """
    
    @abstractmethod
    def solve(
        self,
        f_func: Callable[[npt.ArrayLike], npt.ArrayLike],
        a: Union[float, npt.NDArray[np.floating]],
        b: Union[float, npt.NDArray[np.floating]],
        tolerance: float = 0.01,
        max_iterations: int = 50,
        raise_on_fail: bool = False,
        **kwargs: Any
    ) -> Tuple[Union[float, npt.NDArray[np.floating]], 
               Union[bool, npt.NDArray[np.bool_]]]:
        """
        Solve using bracketing method with interval [a, b].
        
        Parameters
        ----------
        a : float or ndarray
            Left bracket endpoint(s)
        b : float or ndarray  
            Right bracket endpoint(s)
        **kwargs : dict
            Additional method-specific parameters
        """
        pass


class HybridMethodBase(ConvergenceBase):
    """
    Abstract base class for hybrid root-finding methods.
    
    These methods can operate in both open and bracketing modes,
    or combine multiple strategies.
    Examples: Brent-Dekker, scipy.optimize.root_scalar with multiple methods.
    
    Performance Note: Hybrid methods benefit most from JIT compilation due to
    their complex branching logic. Use @njit for optimal performance.
    """
    
    @abstractmethod
    def solve(
        self,
        f_func: Callable[[npt.ArrayLike], npt.ArrayLike],
        tolerance: float = 0.01,
        max_iterations: int = 50,
        raise_on_fail: bool = False,
        x0: Optional[Union[float, npt.NDArray[np.floating]]] = None,
        bracket: Optional[Tuple[Union[float, npt.NDArray[np.floating]], 
                               Union[float, npt.NDArray[np.floating]]]] = None,
        df_func: Optional[Callable[[npt.ArrayLike], npt.ArrayLike]] = None,
        **kwargs: Any
    ) -> Tuple[Union[float, npt.NDArray[np.floating]], 
               Union[bool, npt.NDArray[np.bool_]]]:
        """
        Solve using hybrid method with flexible parameters.
        
        Parameters
        ----------
        x0 : float or ndarray, optional
            Initial guess (for open mode)
        bracket : tuple of (a, b), optional
            Bracket endpoints (for bracketing mode)
        df_func : callable, optional
            Derivative function (if beneficial)
        **kwargs : dict
            Additional method-specific parameters
        """
        pass


# Example implementation template for performance optimization
"""
Example: High-performance Bisection implementation

from numba import njit, prange

class BisectionSolver(BracketingMethodBase):
    
    def solve_vectorized(self, f_func, tolerance=0.01, max_iterations=50, 
                        raise_on_fail=False, chunk_size=1000000, **kwargs):
        '''
        Optimized vectorized bisection with Numba JIT compilation.
        
        For 30M+ points, this will achieve C-speed performance.
        '''
        a_array = kwargs['a_array']
        b_array = kwargs['b_array']
        
        # Delegate to JIT-compiled kernel
        solutions, converged = self._bisection_kernel(
            a_array, b_array, tolerance, max_iterations, f_func
        )
        
        if raise_on_fail and not converged.all():
            failed_indices = np.where(~converged)[0]
            raise RuntimeError(f"Convergence failed for {len(failed_indices)} points")
        
        return solutions, converged
    
    @staticmethod
    @njit(parallel=True)
    def _bisection_kernel(a_array, b_array, tolerance, max_iterations, f_func):
        '''JIT-compiled bisection kernel for maximum performance'''
        n = len(a_array)
        solutions = np.empty(n)
        converged = np.empty(n, dtype=np.bool_)
        
        for i in prange(n):  # Parallel loop
            a, b = a_array[i], b_array[i]
            
            for iteration in range(max_iterations):
                c = (a + b) / 2
                fc = f_func(c)  # Note: Would need function evaluation strategy
                
                if abs(fc) < tolerance or abs(b - a) < tolerance:
                    solutions[i] = c
                    converged[i] = True
                    break
                
                # Update bracket (simplified logic)
                if fc * f_func(a) < 0:
                    b = c
                else:
                    a = c
            else:
                solutions[i] = (a + b) / 2
                converged[i] = False
        
        return solutions, converged
"""