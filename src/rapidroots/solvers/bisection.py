"""
Simple Bisection Method Implementation using the provided base class.
"""

import numpy as np
import numpy.typing as npt
import logging

from rapidroots.solvers.base import BracketingMethodBase
from rapidroots.solvers.mixin import BracketPreparationMixin
from rapidroots.utils.function_utils import validate_bracket, condition_function
from typing import Any, Callable, Tuple, Union, Optional, NamedTuple




class BisectionOptions(NamedTuple):
    """Configuration options for enhanced bisection solving"""
    condition: bool = True
    validate: bool = True
    auto: bool = False


class BisectionSolver(BracketingMethodBase, BracketPreparationMixin):
    """
    Bisection method implementation with clean scalar/vectorized separation.
    
    Uses bracketing approach with guaranteed convergence for continuous functions
    where f(a) and f(b) have opposite signs.
    """
    
    def __init__(self, 
                    logger: Optional[logging.Logger] = None, 
                    dispatcher: Optional[Any] = None,
                    default_chunk_size: int = 1000, 
                    default_enable_logging: bool = True):
            """
            Initialize bisection solver with configurable defaults.
            
            Parameters
            ----------
            logger : logging.Logger, optional
                Logger for error reporting and diagnostics
            dispatcher : object, optional
                Function dispatcher for auto-bracketing and validation
            default_chunk_size : int, default=1000
                Default chunk size for vectorized operations
            default_enable_logging : bool, default=True
                Default logging setting for vectorized operations
            """
            super().__init__(logger)
            self.logger = logger or logging.getLogger(self.__class__.__name__)
            self.dispatcher = dispatcher  # Can be None
            self.default_chunk_size = default_chunk_size
            self.default_enable_logging = default_enable_logging

    def solve(
            self,
            f_func: Callable[[npt.ArrayLike], npt.ArrayLike],
            a: Union[float, npt.NDArray[np.floating]],
            b: Union[float, npt.NDArray[np.floating]],
            tolerance: float = 0.01,
            max_iterations: int = 50,
            raise_on_fail: bool = False,
            opts: BisectionOptions = None,
            func_types: Optional[npt.NDArray] = None,
            params_array: Optional[npt.NDArray] = None,
            initial_guess: Optional[npt.NDArray] = None,
            preferred_domain: Optional[Tuple[float, float]] = None,
            **kwargs: Any
        ) -> Tuple[Union[float, npt.NDArray[np.floating]], 
                Union[bool, npt.NDArray[np.bool_]]]:
            """
            Solve f(x) = 0 using enhanced bisection: conditioning → bracket finding → validation → bisection.
            
            Parameters
            ----------
            f_func : callable
                Function for which to find roots
            a, b : float or ndarray
                Bracket endpoints (or initial guess if opts.auto=True)
            tolerance : float, default=0.01
                Convergence tolerance for |f(x)| or interval width
            max_iterations : int, default=50
                Maximum iterations before giving up
            raise_on_fail : bool, default=False
                Whether to raise exception on convergence failure
            opts : BisectionOptions, default=BisectionOptions()
                Configuration options (condition, validate, auto)
            func_types : ndarray, optional
                Array of function type names (required if opts.auto=True)
            params_array : ndarray, optional
                Parameters for each function (required if opts.auto=True)
            initial_guess : ndarray, optional
                Initial guess for bracket finding
            preferred_domain : tuple, optional
                Preferred domain for bracket search (min, max)
            **kwargs : dict
                Additional parameters
                
            Returns
            -------
            root : float or ndarray
                Root(s) found by bisection
            converged : bool or ndarray
                Convergence status
                
            Notes
            -----
            - When opts.auto=True, dispatcher is required for auto-bracketing
            - If validation fails and raise_on_fail=False, logs warning and continues
            - Vectorized fallback expects a_array/b_array in kwargs
            """
            if opts is None:
                opts = BisectionOptions()

            # Stage 1: Condition the function
            f_func = BracketPreparationMixin.apply_conditioning(self, f_func, opts.condition)

            # Stage 2: Auto-bracket finder
            a, b = BracketPreparationMixin.find_missing_brackets(self,
                a, b, opts.auto, func_types, params_array, 
                initial_guess, preferred_domain, raise_on_fail
            )
            # Stage 3: Validate brackets
            BracketPreparationMixin.validate_all_brackets(self,
                f_func, a, b, opts.validate, func_types, 
                params_array, raise_on_fail
            )
            
            # Stage 4: Dispatch to bisection
            return self._bisect_dispatch(f_func, a, b, tolerance, max_iterations, raise_on_fail, **kwargs)

        
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
        Vectorized bisection solve with configurable performance options.
        
        This method provides the performance tuning interface for vectorized operations.
        Override this method with Numba-compiled kernels for ultimate speed on 30M+ points.
        
        Parameters
        ----------
        f_func : callable
            Vectorized function that accepts array input and returns array output
        tolerance : float, default=0.01
            Convergence tolerance
        max_iterations : int, default=50
            Maximum iterations per solve
        raise_on_fail : bool, default=False
            Whether to raise on convergence failure
        chunk_size : int, default=1000
            Process data in chunks to manage memory
        enable_logging : bool, default=True
            Enable per-point error logging for diagnostics
        **kwargs : dict
            Must contain 'a_array' and 'b_array' with bracket endpoints
            
        Returns
        -------
        solutions : ndarray
            Array of roots found
        converged : ndarray of bool
            Convergence status for each solve
            
        Notes
        -----
        Current implementation uses Python-level chunked fallback from base class.
        For production performance on large datasets, override with:
        
        @staticmethod
        @njit(parallel=True)
        def _bisection_kernel(a_array, b_array, tolerance, max_iterations):
            # JIT-compiled vectorized bisection kernel
            pass
        """
        # Use base class chunked fallback (Python-level but robust)
        return super().solve_vectorized(
            f_func, tolerance, max_iterations, raise_on_fail,
            chunk_size, enable_logging, **kwargs
        )
    
    def _solve_scalar(
        self, 
        f_func: Callable[[float], float], 
        a: float, 
        b: float, 
        tolerance: float, 
        max_iterations: int, 
        raise_on_fail: bool
    ) -> Tuple[float, bool]:
        """
        Core bisection algorithm for scalar inputs.
        
        Implements the standard bisection method with robust error handling
        and multiple convergence criteria.
        
        Parameters
        ----------
        f_func : callable
            Function to find root of
        a, b : float
            Bracket endpoints
        tolerance : float
            Convergence tolerance
        max_iterations : int
            Maximum iterations
        raise_on_fail : bool
            Whether to raise on convergence failure
            
        Returns
        -------
        root : float
            Best estimate of root
        converged : bool
            Whether convergence was achieved
            
        Raises
        ------
        ValueError
            If bracketing condition not satisfied and raise_on_fail=True
        RuntimeError
            If max iterations exceeded and raise_on_fail=True
        """
        # Validate inputs
        if not (np.isfinite(a) and np.isfinite(b)):
            if raise_on_fail:
                raise ValueError(f"Invalid bracket endpoints: a={a}, b={b}")
            return np.nan, False
        
        if a >= b:
            if raise_on_fail:
                raise ValueError(f"Invalid bracket order: a={a} >= b={b}")
            return (a + b) / 2, False
        
        # Check bracketing condition  
        try:
            fa, fb = f_func(a), f_func(b)
        except Exception as e:
            if raise_on_fail:
                raise RuntimeError(f"Function evaluation failed: {e}")
            return (a + b) / 2, False
        
        # Check for exact roots at endpoints
        if abs(fa) < tolerance:
            return a, True
        if abs(fb) < tolerance:
            return b, True
        
        if fa * fb >= 0:
            if raise_on_fail:
                raise ValueError(f"f(a) and f(b) must have opposite signs: f({a:.6e})={fa:.6e}, f({b:.6e})={fb:.6e}")
            return (a + b) / 2, False
        
        # Main bisection loop
        for iteration in range(max_iterations):
            c = (a + b) / 2
            
            try:
                fc = f_func(c)
            except Exception as e:
                if raise_on_fail:
                    raise RuntimeError(f"Function evaluation failed at c={c}: {e}")
                return c, False
            
            # Check convergence criteria
            if abs(fc) < tolerance or abs(b - a) < tolerance:
                return c, True
            
            # Update bracket based on sign of fc
            if fa * fc < 0:
                # Root is in [a, c]
                b, fb = c, fc
            else:
                # Root is in [c, b]
                a, fa = c, fc
        
        # Max iterations reached
        final_root = (a + b) / 2
        if raise_on_fail:
            try:
                final_fc = f_func(final_root)
                raise RuntimeError(
                    f"Bisection failed to converge after {max_iterations} iterations. "
                    f"Final bracket: [{a:.6e}, {b:.6e}], "
                    f"Final root estimate: {final_root:.6e}, "
                    f"Final function value: {final_fc:.6e}"
                )
            except Exception:
                raise RuntimeError(f"Bisection failed to converge after {max_iterations} iterations")
        
        return final_root, False

    def _bisect_dispatch(self, f_func, a, b, tolerance, max_iterations, raise_on_fail, **kwargs):
        """Stage 4: Dispatch to appropriate bisection implementation"""
        # Handle scalar case with optimized path
        if np.isscalar(a) and np.isscalar(b):
            return self._solve_scalar(f_func, a, b, tolerance, max_iterations, raise_on_fail)
        
        # Handle array case - delegate to parent's vectorized fallback
        return super().solve_vectorized(
            f_func, tolerance, max_iterations, raise_on_fail, 
            chunk_size=self.default_chunk_size, enable_logging=self.default_enable_logging, 
            a_array=a, b_array=b, **kwargs
        )

# Comprehensive unit tests for _solve_scalar
def test_solve_scalar():
    """Direct unit tests for _solve_scalar method covering all branches."""
    import math
    
    solver = BisectionSolver()
    
    print("=== Testing _solve_scalar directly ===")
    
    # Test 1: Normal convergence
    f = lambda x: x**2 - 4  # Root at x=2
    root, conv = solver._solve_scalar(f, 1.0, 3.0, 1e-6, 50, False)
    print(f"Test 1 - Normal: root={root:.6f}, converged={conv}, expected=2.0")
    assert conv and abs(root - 2.0) < 1e-5
    
    # Test 2: Exact root at endpoint 'a'
    g = lambda x: x - 1.5  # Root exactly at x=1.5
    root, conv = solver._solve_scalar(g, 1.5, 2.0, 1e-6, 50, False)
    print(f"Test 2 - Root at a: root={root:.6f}, converged={conv}, expected=1.5")
    assert conv and abs(root - 1.5) < 1e-10
    
    # Test 3: Exact root at endpoint 'b'
    root, conv = solver._solve_scalar(g, 1.0, 1.5, 1e-6, 50, False)
    print(f"Test 3 - Root at b: root={root:.6f}, converged={conv}, expected=1.5")
    assert conv and abs(root - 1.5) < 1e-10
    
    # Test 4: Non-bracketed, raise_on_fail=False
    h = lambda x: x**2 + 1  # No real roots
    root, conv = solver._solve_scalar(h, 1.0, 2.0, 1e-6, 50, False)
    print(f"Test 4 - Non-bracketed (no raise): root={root:.6f}, converged={conv}")
    assert not conv and root == 1.5  # Should return midpoint
    
    # Test 5: Non-bracketed, raise_on_fail=True
    try:
        root, conv = solver._solve_scalar(h, 1.0, 2.0, 1e-6, 50, True)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        print(f"Test 5 - Non-bracketed (raise): Correctly raised ValueError")
        assert "opposite signs" in str(e)
    
    # Test 6: Function evaluation exception, raise_on_fail=False
    def bad_func(x):
        if x > 1.5:
            raise RuntimeError("Simulated function error")
        return x - 1.0
    
    root, conv = solver._solve_scalar(bad_func, 0.5, 2.0, 1e-6, 50, False)
    print(f"Test 6 - Function error (no raise): root={root:.6f}, converged={conv}")
    assert not conv  # Should handle gracefully
    
    # Test 7: Function evaluation exception, raise_on_fail=True
    try:
        root, conv = solver._solve_scalar(bad_func, 0.5, 2.0, 1e-6, 50, True)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Test 7 - Function error (raise): Correctly raised RuntimeError")
        assert "evaluation failed" in str(e)
    
    # Test 8: Maximum iterations, raise_on_fail=False
    def slow_func(x):
        return x - math.pi  # Root at pi, will need many iterations for high precision
    
    root, conv = solver._solve_scalar(slow_func, 3.0, 4.0, 1e-15, 5, False)  # Very tight tolerance, few iterations
    print(f"Test 8 - Max iters (no raise): root={root:.6f}, converged={conv}")
    assert not conv  # Should not converge in 5 iterations
    
    # Test 9: Maximum iterations, raise_on_fail=True
    try:
        root, conv = solver._solve_scalar(slow_func, 3.0, 4.0, 1e-15, 5, True)
        assert False, "Should have raised RuntimeError"
    except RuntimeError as e:
        print(f"Test 9 - Max iters (raise): Correctly raised RuntimeError")
        assert "failed to converge" in str(e).lower()
    
    # Test 10: Invalid inputs - infinite values
    root, conv = solver._solve_scalar(f, float('inf'), 2.0, 1e-6, 50, False)
    print(f"Test 10 - Invalid input: root={root}, converged={conv}")
    assert not conv and math.isnan(root)
    
    # Test 11: Invalid inputs - wrong order (a >= b)
    root, conv = solver._solve_scalar(f, 3.0, 1.0, 1e-6, 50, False)
    print(f"Test 11 - Wrong order: root={root:.6f}, converged={conv}")
    assert not conv and root == 2.0  # Should return midpoint
    
    print("All _solve_scalar tests passed!")


# Example usage and comprehensive tests
if __name__ == "__main__":
    
    # Run direct unit tests first
    test_solve_scalar()
    
    print("\n" + "="*50)
    print("=== Integration Tests ===")
    
    # Create solver with custom defaults
    solver = BisectionSolver(
        logger=logging.getLogger("bisection_test"),
        default_chunk_size=500,
        default_enable_logging=False
    )
    
    # Test 1: Scalar solve - find square root of 2
    print("Test 1: Finding √2")
    f = lambda x: x**2 - 2
    root, converged = solver.solve(f_func=f, a=1.0, b=2.0, tolerance=1e-6)
    print(f"√2 ≈ {root:.8f}, converged: {converged}")
    print(f"Verification: f(root) = {f(root):.2e}")
    
    # Test 2: Error handling - non-bracketing interval
    print("\nTest 2: Non-bracketing interval")
    try:
        root, converged = solver.solve(f_func=f, a=2.0, b=3.0, raise_on_fail=True)
        print(f"root: {root}, converged: {converged}")
    except ValueError as e:
        print(f"Expected error caught: {e}")
    
    # Test 3: Vectorized solve with custom performance options
    print("\nTest 3: Vectorized solve with custom options")
    n_points = 100
    a_vals = np.linspace(1.0, 1.4, n_points)
    b_vals = np.linspace(1.5, 2.0, n_points)
    
    roots, conv = solver.solve_vectorized(
        f, tolerance=1e-4, chunk_size=25, enable_logging=True,
        a_array=a_vals, b_array=b_vals
    )
    
    success_rate = conv.mean() * 100
    print(f"Success rate: {success_rate:.1f}%")
    if conv.any():
        print(f"Mean root: {roots[conv].mean():.6f}")
        print(f"Expected: {np.sqrt(2):.6f}")
    
    # Test 4: Standard solve() method (should use defaults)
    print("\nTest 4: Standard solve() using class defaults")
    roots2, conv2 = solver.solve(f, a_vals, b_vals, tolerance=1e-4)
    print(f"Success rate with defaults: {conv2.mean() * 100:.1f}%")
    
    print("\nAll tests completed!")