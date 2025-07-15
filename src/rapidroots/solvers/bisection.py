"""
Simple Bisection Method Implementation using the provided base class.
"""

import numpy as np
import sympy as sp
import numpy.typing as npt
import logging

from rapidroots.solvers.base import BracketingMethodBase
from rapidroots.solvers.mixin import BracketPreparationMixin
from rapidroots.utils.function_utils import validate_bracket, condition_function
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher
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
        # Check dispatcher mode consistency - guard against partial specification
        has_func_types = "func_types" in kwargs
        has_params_array = "params_array" in kwargs
        
        if has_func_types ^ has_params_array:  # XOR - exactly one is present
            missing = "params_array" if has_func_types else "func_types"
            present = "func_types" if has_func_types else "params_array"
            raise ValueError(
                f"Dispatcher mode requires both 'func_types' and 'params_array'. "
                f"Found '{present}' but missing '{missing}'"
            )
        
        # Check if this is dispatcher mode or generic function mode
        if not (has_func_types and has_params_array):
            # Generic function mode - delegate to base class
            return super().solve_vectorized(
                f_func, tolerance, max_iterations, raise_on_fail, 
                chunk_size, enable_logging, **kwargs
            )
        
        # Dispatcher mode - validate required kwargs
        required = ("func_types", "params_array", "a_array", "b_array")
        missing = [k for k in required if k not in kwargs]
        if missing:
            raise ValueError(f"Missing required kwarg(s) for dispatcher bisection: {missing}")
        
        # Extract inputs with graceful error handling
        try:
            func_types   = kwargs.pop("func_types")
            params_array = kwargs.pop("params_array")
            a_array      = np.asarray(kwargs.pop("a_array"), dtype=np.float64)
            b_array      = np.asarray(kwargs.pop("b_array"), dtype=np.float64)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Invalid array data in kwargs: {e}")
        
        n = len(a_array)
        
        # Validate array dimensions and shapes
        if len(b_array) != n:
            raise ValueError(f"a_array and b_array must have same length: {len(a_array)} vs {len(b_array)}")
        if len(func_types) != n:
            raise ValueError(f"func_types must match array length: {len(func_types)} vs {n}")
        if len(params_array) != n:
            raise ValueError(f"params_array must match array length: {len(params_array)} vs {n}")
        
        # Additional shape validation for params_array
        if not isinstance(params_array, np.ndarray):
            params_array = np.asarray(params_array)
        if params_array.ndim != 2:
            raise ValueError(f"params_array must be 2D, got shape {params_array.shape}")

        # Handle chunk_size defaulting - explicitly check for None
        if chunk_size is None:
            chunk_size = self.default_chunk_size
        elif chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")

        roots     = np.empty(n, dtype=np.float64)
        converged = np.zeros(n, dtype=bool)

        # Process each chunk
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            
            # Initialize chunk using helper method
            try:
                (a_ch, b_ch, fa, fb, valid_idx, exact_a_idx, exact_b_idx, 
                bad_idx, zero_width_idx) = self._initialize_chunk(
                    start, end, func_types, params_array, a_array, b_array, tolerance
                )
            except Exception as e:
                # If initialization fails, mark entire chunk as failed
                if enable_logging:
                    self.logger.error(f"Chunk initialization failed for {start}-{end}: {e}")
                roots[start:end] = (a_array[start:end] + b_array[start:end]) * 0.5
                converged[start:end] = False
                continue
            
            # Set results for exact solutions
            if len(exact_a_idx) > 0:
                roots[start + exact_a_idx] = a_ch[exact_a_idx]
                converged[start + exact_a_idx] = True
            if len(exact_b_idx) > 0:
                roots[start + exact_b_idx] = b_ch[exact_b_idx]
                converged[start + exact_b_idx] = True
            
            # Mark invalid cases as failed
            if len(bad_idx) > 0:
                roots[start + bad_idx] = (a_ch[bad_idx] + b_ch[bad_idx]) * 0.5
                converged[start + bad_idx] = False
            
            # Handle zero-width intervals - mark as converged if f(a) is small enough
            if len(zero_width_idx) > 0:
                zero_width_converged = np.abs(fa[zero_width_idx]) < tolerance
                roots[start + zero_width_idx] = a_ch[zero_width_idx]
                converged[start + zero_width_idx] = zero_width_converged
                
                # Log/raise for zero-width intervals that don't converge
                zero_width_failed = zero_width_idx[~zero_width_converged]
                if len(zero_width_failed) > 0:
                    if enable_logging:
                        self.logger.warning(
                            f"Found {len(zero_width_failed)} zero-width intervals with |f(a)| >= tolerance"
                        )
                    if raise_on_fail:
                        raise ValueError(
                            f"Found {len(zero_width_failed)} zero-width intervals that don't satisfy convergence"
                        )

            # Only iterate on the valid ones that need bisection
            if len(valid_idx) == 0:
                continue

            # Main bisection loop
            active_idx = valid_idx.copy()  # Track which indices are still active
            
            for iteration in range(max_iterations):
                if len(active_idx) == 0:
                    break
                    
                try:
                    # Midpoints for active problems only
                    c = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                    
                    # Get function types and params for active indices
                    ft_active = func_types[start:end][active_idx]
                    pa_active = params_array[start:end][active_idx]
                    
                    fc = self.dispatcher.evaluate_batch(ft_active, pa_active, c, derivative=0)
                    
                except Exception as e:
                    # If dispatcher fails mid-iteration, mark remaining as failed
                    if enable_logging:
                        self.logger.error(f"Dispatcher failed at iteration {iteration} for chunk {start}-{end}: {e}")
                    roots[start + active_idx] = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                    converged[start + active_idx] = False
                    break

                # Which ones have converged?
                func_converged = np.abs(fc) < tolerance
                interval_converged = np.abs(b_ch[active_idx] - a_ch[active_idx]) < tolerance
                done = func_converged | interval_converged

                # Record converged solutions
                done_idx = active_idx[done]
                if len(done_idx) > 0:
                    roots[start + done_idx] = c[done]
                    converged[start + done_idx] = True

                # Update remaining active problems
                keep_local = ~done
                if not np.any(keep_local):
                    break

                keep_idx = active_idx[keep_local]
                c_keep = c[keep_local]
                fc_keep = fc[keep_local]
                
                # For those that remain, decide which side to replace
                fa_keep = fa[keep_idx]
                left_mask = (fa_keep * fc_keep < 0)

                # Update intervals: if fa*fc < 0, root is in [a,c], so b=c
                # Otherwise root is in [c,b], so a=c
                b_ch[keep_idx[left_mask]] = c_keep[left_mask]
                fb[keep_idx[left_mask]] = fc_keep[left_mask]
                
                a_ch[keep_idx[~left_mask]] = c_keep[~left_mask]
                fa[keep_idx[~left_mask]] = fc_keep[~left_mask]

                # Update active indices for next iteration
                active_idx = keep_idx

            # Any that never converged get midpoint
            if len(active_idx) > 0:
                # Final convergence check
                final_c = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                try:
                    ft_active = func_types[start:end][active_idx]
                    pa_active = params_array[start:end][active_idx]
                    final_fc = self.dispatcher.evaluate_batch(ft_active, pa_active, final_c, derivative=0)
                    
                    final_func_converged = np.abs(final_fc) < tolerance
                    final_interval_converged = np.abs(b_ch[active_idx] - a_ch[active_idx]) < tolerance
                    final_done = final_func_converged | final_interval_converged
                    
                    # Set converged points
                    final_converged_idx = active_idx[final_done]
                    if len(final_converged_idx) > 0:
                        roots[start + final_converged_idx] = final_c[final_done]
                        converged[start + final_converged_idx] = True
                    
                    # Set non-converged points
                    final_not_converged_idx = active_idx[~final_done]
                    if len(final_not_converged_idx) > 0:
                        roots[start + final_not_converged_idx] = final_c[~final_done]
                        converged[start + final_not_converged_idx] = False
                        
                        if enable_logging:
                            self.logger.warning(
                                f"Chunk {start}-{end}: {len(final_not_converged_idx)}/{len(valid_idx)} failed to converge after {max_iterations} iterations"
                            )
                except Exception as e:
                    # If final evaluation fails, just use midpoints
                    roots[start + active_idx] = final_c
                    converged[start + active_idx] = False
                if enable_logging:
                    self.logger.warning(
                        f"Chunk {start}-{end}: {len(active_idx)}/{len(valid_idx)} failed to converge after {max_iterations} iterations"
                    )

        # Final error handling
        if raise_on_fail and not converged.all():
            nfail = np.sum(~converged)
            raise RuntimeError(f"Bisection failed to converge for {nfail}/{n} points")

        return roots, converged

            
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
    
    def _initialize_chunk(
        self,
        start: int,
        end: int,
        func_types: np.ndarray,
        params_array: np.ndarray,
        a_array: np.ndarray,
        b_array: np.ndarray,
        tolerance: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, 
            np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Initialize a chunk for bisection: evaluate endpoints and categorize intervals.
        
        Returns
        -------
        a_ch, b_ch : ndarray
            Chunk interval endpoints (copies)
        fa, fb : ndarray
            Function values at endpoints
        valid_idx : ndarray
            Indices of intervals that need bisection
        exact_a_idx, exact_b_idx : ndarray
            Indices where exact roots found at endpoints
        bad_idx : ndarray
            Indices of invalid intervals
        zero_width_idx : ndarray
            Indices of zero-width intervals
        """
        ft_chunk = func_types[start:end]
        pa_chunk = params_array[start:end]
        a_ch = a_array[start:end].copy()
        b_ch = b_array[start:end].copy()
        
        # Evaluate at endpoints using dispatcher
        fa = self.dispatcher.evaluate_batch(ft_chunk, pa_chunk, a_ch, derivative=0)
        fb = self.dispatcher.evaluate_batch(ft_chunk, pa_chunk, b_ch, derivative=0)
        
        # Categorize intervals - these categories exhaustively partition the chunk:
        # 1. finite_and_ordered: basic validity (finite values, a < b)
        # 2. zero_width: special case where a == b
        # 3. valid: proper intervals that bracket a root and need bisection
        # 4. exact_a/exact_b: exact roots found at endpoints
        # 5. bad: everything else (invalid)
        
        finite_and_ordered = (
            np.isfinite(a_ch) & np.isfinite(b_ch) & (a_ch < b_ch)
        )
        zero_width = (
            np.isfinite(a_ch) & np.isfinite(b_ch) & (a_ch == b_ch)
        )
        finite_function_vals = np.isfinite(fa) & np.isfinite(fb)
        properly_bracketed = (fa * fb) <= 0  # allow zero
        
        valid_base = finite_and_ordered & finite_function_vals & properly_bracketed
        
        # Handle exact roots at endpoints
        exact_a_mask = valid_base & (np.abs(fa) < tolerance)
        exact_b_mask = valid_base & ~exact_a_mask & (np.abs(fb) < tolerance)
        
        # Valid intervals are those that need bisection (not exact solutions)
        valid_mask = valid_base & ~exact_a_mask & ~exact_b_mask
        
        # Everything else is bad
        bad_mask = ~valid_base & ~exact_a_mask & ~exact_b_mask & ~zero_width
        
        # Convert boolean masks to index arrays for efficiency
        valid_idx = np.where(valid_mask)[0]
        exact_a_idx = np.where(exact_a_mask)[0]
        exact_b_idx = np.where(exact_b_mask)[0]
        bad_idx = np.where(bad_mask)[0]
        zero_width_idx = np.where(zero_width & finite_function_vals)[0]
        
        return (a_ch, b_ch, fa, fb, valid_idx, exact_a_idx, exact_b_idx, 
                bad_idx, zero_width_idx)
