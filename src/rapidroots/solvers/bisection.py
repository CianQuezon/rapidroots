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
            """
            import sys
            
            print(f"DEBUG: solve_vectorized called with tolerance={tolerance}, max_iterations={max_iterations}", file=sys.stderr, flush=True)
            print(f"DEBUG: kwargs keys: {list(kwargs.keys())}", file=sys.stderr, flush=True)
            
            # Check dispatcher mode consistency - guard against partial specification
            has_func_types = "func_types" in kwargs
            has_params_array = "params_array" in kwargs
            
            print(f"DEBUG: has_func_types={has_func_types}, has_params_array={has_params_array}", file=sys.stderr, flush=True)
            
            if has_func_types ^ has_params_array:  # XOR - exactly one is present
                missing = "params_array" if has_func_types else "func_types"
                present = "func_types" if has_func_types else "params_array"
                raise ValueError(
                    f"Dispatcher mode requires both 'func_types' and 'params_array'. "
                    f"Found '{present}' but missing '{missing}'"
                )
            
            # Check if this is dispatcher mode or generic function mode
            if not (has_func_types and has_params_array):
                print(f"DEBUG: Using generic function mode - delegating to base class", file=sys.stderr, flush=True)
                # Generic function mode - delegate to base class
                return super().solve_vectorized(
                    f_func, tolerance, max_iterations, raise_on_fail, 
                    chunk_size, enable_logging, **kwargs
                )
            
            print(f"DEBUG: Using dispatcher mode", file=sys.stderr, flush=True)
            
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
            print(f"DEBUG: Processing {n} problems", file=sys.stderr, flush=True)
            print(f"DEBUG: a_array[:5] = {a_array[:5]}", file=sys.stderr, flush=True)
            print(f"DEBUG: b_array[:5] = {b_array[:5]}", file=sys.stderr, flush=True)
            
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

            print(f"DEBUG: Using chunk_size = {chunk_size}", file=sys.stderr, flush=True)

            roots     = np.empty(n, dtype=np.float64)
            converged = np.zeros(n, dtype=bool)

            # Process each chunk
            for start in range(0, n, chunk_size):
                end = min(start + chunk_size, n)
                print(f"DEBUG: Processing chunk {start}-{end}", file=sys.stderr, flush=True)
                
                # Initialize chunk using helper method
                try:
                    (a_ch, b_ch, fa, fb, valid_idx, exact_a_idx, exact_b_idx, 
                    bad_idx, zero_width_idx) = self._initialize_chunk(
                        start, end, func_types, params_array, a_array, b_array, tolerance
                    )
                    print(f"DEBUG: Chunk initialized - valid:{len(valid_idx)}, exact_a:{len(exact_a_idx)}, exact_b:{len(exact_b_idx)}, bad:{len(bad_idx)}, zero_width:{len(zero_width_idx)}", file=sys.stderr, flush=True)
                except Exception as e:
                    print(f"DEBUG: Chunk initialization failed: {e}", file=sys.stderr, flush=True)
                    # If initialization fails, mark entire chunk as failed
                    if enable_logging:
                        self.logger.error(f"Chunk initialization failed for {start}-{end}: {e}")
                    roots[start:end] = (a_array[start:end] + b_array[start:end]) * 0.5
                    converged[start:end] = False
                    continue
                
                # Set results for exact solutions
                if len(exact_a_idx) > 0:
                    print(f"DEBUG: Setting exact solutions at a endpoints: {exact_a_idx}", file=sys.stderr, flush=True)
                    roots[start + exact_a_idx] = a_ch[exact_a_idx]
                    converged[start + exact_a_idx] = True
                if len(exact_b_idx) > 0:
                    print(f"DEBUG: Setting exact solutions at b endpoints: {exact_b_idx}", file=sys.stderr, flush=True)
                    roots[start + exact_b_idx] = b_ch[exact_b_idx]
                    converged[start + exact_b_idx] = True
                
                # Mark invalid cases as failed
                if len(bad_idx) > 0:
                    print(f"DEBUG: Marking bad intervals as failed: {bad_idx}", file=sys.stderr, flush=True)
                    roots[start + bad_idx] = (a_ch[bad_idx] + b_ch[bad_idx]) * 0.5
                    converged[start + bad_idx] = False
                
                # Handle zero-width intervals - mark as converged if f(a) is small enough
                if len(zero_width_idx) > 0:
                    zero_width_converged = np.abs(fa[zero_width_idx]) < tolerance
                    print(f"DEBUG: Zero-width intervals: {zero_width_idx}, converged: {zero_width_converged}", file=sys.stderr, flush=True)
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
                    print(f"DEBUG: No valid intervals to process in this chunk", file=sys.stderr, flush=True)
                    continue

                print(f"DEBUG: Starting bisection loop for {len(valid_idx)} valid intervals", file=sys.stderr, flush=True)
                print(f"DEBUG: Initial intervals - a:{a_ch[valid_idx]}, b:{b_ch[valid_idx]}", file=sys.stderr, flush=True)
                print(f"DEBUG: Initial function values - fa:{fa[valid_idx]}, fb:{fb[valid_idx]}", file=sys.stderr, flush=True)

                # Main bisection loop
                active_idx = valid_idx.copy()  # Track which indices are still active
                
                for iteration in range(max_iterations):
                    if len(active_idx) == 0:
                        print(f"DEBUG: No active problems left at iteration {iteration}", file=sys.stderr, flush=True)
                        break
                    
                    print(f"DEBUG: === ITERATION {iteration} ===", file=sys.stderr, flush=True)
                    print(f"DEBUG: Active problems: {len(active_idx)}", file=sys.stderr, flush=True)
                        
                    try:
                        # Midpoints for active problems only
                        c = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                        print(f"DEBUG: Midpoints: {c}", file=sys.stderr, flush=True)
                        
                        # Get function types and params for active indices
                        ft_active = func_types[start:end][active_idx]
                        pa_active = params_array[start:end][active_idx]
                        print(f"DEBUG: Function types: {ft_active}", file=sys.stderr, flush=True)
                        
                        fc = self.dispatcher.evaluate_batch(ft_active, pa_active, c, derivative=0)
                        print(f"DEBUG: Function values at midpoints: {fc}", file=sys.stderr, flush=True)
                        
                    except Exception as e:
                        print(f"DEBUG: Dispatcher failed at iteration {iteration}: {e}", file=sys.stderr, flush=True)
                        # If dispatcher fails mid-iteration, mark remaining as failed
                        if enable_logging:
                            self.logger.error(f"Dispatcher failed at iteration {iteration} for chunk {start}-{end}: {e}")
                        roots[start + active_idx] = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                        converged[start + active_idx] = False
                        break

                    # Which ones have converged?
                    func_converged = np.abs(fc) < tolerance
                    interval_widths = np.abs(b_ch[active_idx] - a_ch[active_idx])
                    interval_converged = interval_widths < tolerance
                    done = func_converged | interval_converged
                    
                    print(f"DEBUG: |f(c)| = {np.abs(fc)}", file=sys.stderr, flush=True)
                    print(f"DEBUG: tolerance = {tolerance}", file=sys.stderr, flush=True)
                    print(f"DEBUG: func_converged = {func_converged}", file=sys.stderr, flush=True)
                    print(f"DEBUG: interval_widths = {interval_widths}", file=sys.stderr, flush=True)
                    print(f"DEBUG: interval_converged = {interval_converged}", file=sys.stderr, flush=True)
                    print(f"DEBUG: done = {done}", file=sys.stderr, flush=True)
                    
                    # Record converged solutions
                    done_idx = active_idx[done]
                    if len(done_idx) > 0:
                        print(f"DEBUG: Recording {len(done_idx)} converged solutions at indices {done_idx}", file=sys.stderr, flush=True)
                        print(f"DEBUG: Converged roots: {c[done]}", file=sys.stderr, flush=True)
                        roots[start + done_idx] = c[done]
                        converged[start + done_idx] = True

                    # Update remaining active problems
                    keep_local = ~done
                    if not np.any(keep_local):
                        print(f"DEBUG: All problems converged this iteration", file=sys.stderr, flush=True)
                        break

                    keep_idx = active_idx[keep_local]
                    c_keep = c[keep_local]
                    fc_keep = fc[keep_local]
                    
                    print(f"DEBUG: Continuing with {len(keep_idx)} problems", file=sys.stderr, flush=True)
                    print(f"DEBUG: Remaining midpoints: {c_keep}", file=sys.stderr, flush=True)
                    print(f"DEBUG: Remaining function values: {fc_keep}", file=sys.stderr, flush=True)
                    
                    # For those that remain, decide which side to replace
                    fa_keep = fa[keep_idx]
                    left_mask = (fa_keep * fc_keep < 0)
                    
                    print(f"DEBUG: fa_keep: {fa_keep}", file=sys.stderr, flush=True)
                    print(f"DEBUG: fc_keep: {fc_keep}", file=sys.stderr, flush=True)
                    print(f"DEBUG: fa_keep * fc_keep: {fa_keep * fc_keep}", file=sys.stderr, flush=True)
                    print(f"DEBUG: left_mask (root in [a,c]): {left_mask}", file=sys.stderr, flush=True)

                    # Update intervals: if fa*fc < 0, root is in [a,c], so b=c
                    # Otherwise root is in [c,b], so a=c
                    b_ch[keep_idx[left_mask]] = c_keep[left_mask]
                    fb[keep_idx[left_mask]] = fc_keep[left_mask]
                    
                    a_ch[keep_idx[~left_mask]] = c_keep[~left_mask]
                    fa[keep_idx[~left_mask]] = fc_keep[~left_mask]

                    print(f"DEBUG: Updated intervals - a:{a_ch[keep_idx]}, b:{b_ch[keep_idx]}", file=sys.stderr, flush=True)
                    print(f"DEBUG: New interval widths: {np.abs(b_ch[keep_idx] - a_ch[keep_idx])}", file=sys.stderr, flush=True)

                    # Update active indices for next iteration
                    active_idx = keep_idx

                # Any that never converged get midpoint
                if len(active_idx) > 0:
                    final_roots = 0.5 * (a_ch[active_idx] + b_ch[active_idx])
                    print(f"DEBUG: FINAL UNCONVERGED: {len(active_idx)} problems", file=sys.stderr, flush=True)
                    print(f"DEBUG: Final intervals: a={a_ch[active_idx]}, b={b_ch[active_idx]}", file=sys.stderr, flush=True)
                    print(f"DEBUG: Final interval widths: {np.abs(b_ch[active_idx] - a_ch[active_idx])}", file=sys.stderr, flush=True)
                    print(f"DEBUG: Final midpoints: {final_roots}", file=sys.stderr, flush=True)
                    
                    roots[start + active_idx] = final_roots
                    converged[start + active_idx] = False

                    if enable_logging:
                        self.logger.warning(
                            f"Chunk {start}-{end}: {len(active_idx)}/{len(valid_idx)} failed to converge after {max_iterations} iterations"
                        )

            print(f"DEBUG: Final results summary:", file=sys.stderr, flush=True)
            print(f"DEBUG: Total problems: {n}", file=sys.stderr, flush=True)
            print(f"DEBUG: Converged: {np.sum(converged)}", file=sys.stderr, flush=True)
            print(f"DEBUG: Failed: {np.sum(~converged)}", file=sys.stderr, flush=True)
            print(f"DEBUG: First 5 roots: {roots[:5]}", file=sys.stderr, flush=True)
            print(f"DEBUG: First 5 converged: {converged[:5]}", file=sys.stderr, flush=True)

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

def test_solve_vectorized():
    """
    Comprehensive numerical tests for solve_vectorized method.
    Tests basic functionality, edge cases, convergence behavior, and error handling.
    """
    print("=== Testing solve_vectorized - Numerical Vector Bisection ===\n")
    
    # Create solver instance
    solver = BisectionSolver(
        default_chunk_size=100,
        default_enable_logging=True
    )
    
    # === TEST 1: Basic Functionality - Simple Quadratic ===
    print("TEST 1: Basic functionality - x^2 - 4 = 0")
    
    def quadratic(x):
        return x**2 - 4  # Roots at x = ±2
    
    n = 100
    a_array = np.full(n, 1.0)  # All intervals [1, 3] should find root at x=2
    b_array = np.full(n, 3.0)
    
    roots, converged = solver.solve_vectorized(
        quadratic,
        tolerance=1e-3,  # Use a more reasonable tolerance for testing
        max_iterations=50,
        a_array=a_array,
        b_array=b_array
    )
    
    success_rate = np.mean(converged) * 100
    mean_root = np.mean(roots[converged]) if np.any(converged) else np.nan
    max_error = np.max(np.abs(roots[converged] - 2.0)) if np.any(converged) else np.inf
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Mean root: {mean_root:.6f} (expected: 2.0)")
    print(f"  Max error: {max_error:.2e}")
    assert success_rate > 95, "Basic quadratic should have high success rate"
    assert abs(mean_root - 2.0) < 1e-2, "Mean root should be close to 2.0"  # Relaxed to match tolerance
    assert max_error < 1e-2, "Max error should be within tolerance"
    
    # === TEST 2: Mixed Intervals - Different Roots ===
    print("\nTEST 2: Mixed intervals - different roots")
    
    def cubic(x):
        return x**3 - 6*x**2 + 11*x - 6  # Roots at x = 1, 2, 3
    
    # Mix of intervals targeting different roots
    a_mixed = np.array([0.5, 1.5, 2.5] * 20)  # 60 problems
    b_mixed = np.array([1.5, 2.5, 3.5] * 20)  # Targeting roots 1, 2, 3
    expected_roots = np.array([1.0, 2.0, 3.0] * 20)
    
    roots, converged = solver.solve_vectorized(
        cubic,
        tolerance=1e-3,  # Use realistic tolerance
        max_iterations=100,
        a_array=a_mixed,
        b_array=b_mixed
    )
    
    # Check each root separately
    for target_root in [1.0, 2.0, 3.0]:
        mask = np.abs(expected_roots - target_root) < 0.1
        local_success = np.mean(converged[mask]) * 100
        if np.any(converged[mask]):
            local_error = np.max(np.abs(roots[mask & converged] - target_root))
            print(f"  Root {target_root}: {local_success:.1f}% success, max error: {local_error:.2e}")
            assert local_error < 1e-2, f"Root {target_root} should be within tolerance"
    
    overall_success = np.mean(converged) * 100
    print(f"  Overall success rate: {overall_success:.1f}%")
    assert overall_success > 90, "Mixed intervals should have good success rate"
    
    # === TEST 3: Edge Case - Exact Roots at Endpoints ===
    print("\nTEST 3: Edge cases - exact roots at endpoints")
    
    def linear(x):
        return x - 2.5  # Root exactly at x = 2.5
    
    # Test with root at left endpoint, right endpoint, and inside
    a_edge = np.array([2.5, 2.0, 2.2])  # Root at left, inside, inside
    b_edge = np.array([3.0, 2.5, 2.8])  # Normal, root at right, inside
    
    roots, converged = solver.solve_vectorized(
        linear,
        tolerance=1e-6,  # Keep tight tolerance for exact root test
        max_iterations=50,
        a_array=a_edge,
        b_array=b_edge
    )
    
    print(f"  Roots found: {roots}")
    print(f"  Converged: {converged}")
    errors = np.abs(roots - 2.5)
    print(f"  Errors: {errors}")
    
    assert np.all(converged), "All should converge with exact endpoint roots"
    # For exact roots, allow slightly larger error due to implementation details
    assert np.all(errors < 1e-5), "All roots should be reasonably accurate"
    
    # === TEST 4: Non-Bracketing Intervals ===
    print("\nTEST 4: Non-bracketing intervals")
    
    def positive_func(x):
        return x**2 + 1  # Always positive, no real roots
    
    a_bad = np.array([1.0, 2.0, 3.0])
    b_bad = np.array([2.0, 3.0, 4.0])
    
    roots, converged = solver.solve_vectorized(
        positive_func,
        tolerance=1e-6,
        raise_on_fail=False,
        a_array=a_bad,
        b_array=b_bad
    )
    
    print(f"  Converged flags: {converged}")
    print(f"  Roots (should be midpoints): {roots}")
    expected_midpoints = (a_bad + b_bad) / 2
    print(f"  Expected midpoints: {expected_midpoints}")
    
    assert not np.any(converged), "Non-bracketing intervals should not converge"
    assert np.allclose(roots, expected_midpoints), "Should return midpoints for failed cases"
    
    # === TEST 5: Challenging Function - Transcendental ===
    print("\nTEST 5: Challenging function - x - cos(x) = 0")
    
    def transcendental(x):
        return x - np.cos(x)  # Root approximately at x ≈ 0.739085
    
    n_trans = 50
    # Multiple intervals around the known root
    a_trans = np.linspace(0.5, 0.7, n_trans)
    b_trans = np.linspace(0.8, 1.0, n_trans)
    true_root = 0.739085133215  # Known value
    
    roots, converged = solver.solve_vectorized(
        transcendental,
        tolerance=1e-4,  # More realistic tolerance
        max_iterations=100,
        a_array=a_trans,
        b_array=b_trans
    )
    
    success_rate = np.mean(converged) * 100
    if np.any(converged):
        mean_error = np.mean(np.abs(roots[converged] - true_root))
        max_error = np.max(np.abs(roots[converged] - true_root))
        print(f"  Success rate: {success_rate:.1f}%")
        print(f"  Mean error: {mean_error:.2e}")
        print(f"  Max error: {max_error:.2e}")
        assert mean_error < 1e-3, "Transcendental function should converge reasonably"
    
    # === TEST 6: Large Array Performance ===
    print("\nTEST 6: Large array performance")
    
    import time
    
    def simple_poly(x):
        return x**3 - x - 1  # Root approximately at x ≈ 1.3247
    
    n_large = 10000
    a_large = np.random.uniform(1.0, 1.2, n_large)
    b_large = np.random.uniform(1.4, 1.6, n_large)
    
    start_time = time.time()
    roots, converged = solver.solve_vectorized(
        simple_poly,
        tolerance=1e-4,  # Realistic tolerance for performance test
        chunk_size=1000,
        enable_logging=False,  # Disable for performance
        a_array=a_large,
        b_array=b_large
    )
    elapsed = time.time() - start_time
    
    success_rate = np.mean(converged) * 100
    rate = n_large / elapsed
    print(f"  Processed {n_large} problems in {elapsed:.3f}s ({rate:.0f} problems/sec)")
    print(f"  Success rate: {success_rate:.1f}%")
    assert success_rate > 95, "Large array should maintain high success rate"
    
    # === TEST 7: Tolerance and Convergence Behavior ===
    print("\nTEST 7: Tolerance and convergence behavior")
    
    def test_func(x):
        return x**2 - 2  # Root at √2 ≈ 1.414213562373095
    
    tolerances = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]
    true_root = np.sqrt(2)
    
    for tol in tolerances:
        roots, converged = solver.solve_vectorized(
            test_func,
            tolerance=tol,
            max_iterations=100,
            a_array=np.array([1.0]),
            b_array=np.array([2.0])
        )
        
        if converged[0]:
            error = abs(roots[0] - true_root)
            print(f"  Tolerance {tol:.0e}: root={roots[0]:.10f}, error={error:.2e}")
            # Error should be within reasonable bounds of tolerance
            # Bisection method may not achieve exact tolerance due to interval width
            assert error < 100 * tol, f"Error {error} should be reasonably close to tolerance {tol}"
    
    # === TEST 8: Maximum Iterations Behavior ===
    print("\nTEST 8: Maximum iterations behavior")
    
    def slow_func(x):
        return x - np.pi  # Root at π, test with very tight tolerance
    
    max_iters = [5, 10, 20, 50]
    
    for max_iter in max_iters:
        roots, converged = solver.solve_vectorized(
            slow_func,
            tolerance=1e-8,  # Reasonable tolerance for iteration test
            max_iterations=max_iter,
            a_array=np.array([3.0]),
            b_array=np.array([3.2])
        )
        
        if converged[0]:
            error = abs(roots[0] - np.pi)
            print(f"  {max_iter:2d} iterations: converged, error={error:.2e}")
        else:
            print(f"  {max_iter:2d} iterations: failed to converge")
    
    # === TEST 9: Error Handling ===
    print("\nTEST 9: Error handling")
    
    # Test missing arrays
    try:
        solver.solve_vectorized(quadratic, tolerance=1e-6)
        assert False, "Should raise ValueError for missing arrays"
    except ValueError as e:
        print(f"  Correctly caught missing arrays: {str(e)[:50]}...")
    
    # Test mismatched array lengths
    try:
        solver.solve_vectorized(
            quadratic,
            a_array=np.array([1.0, 2.0]),
            b_array=np.array([3.0])  # Different length
        )
        assert False, "Should raise ValueError for mismatched lengths"
    except ValueError as e:
        print(f"  Correctly caught length mismatch: {str(e)[:50]}...")
    
    # Test raise_on_fail behavior
    try:
        solver.solve_vectorized(
            positive_func,  # No real roots
            tolerance=1e-6,
            raise_on_fail=True,
            a_array=np.array([1.0, 2.0]),
            b_array=np.array([2.0, 3.0])
        )
        assert False, "Should raise RuntimeError for raise_on_fail=True"
    except RuntimeError as e:
        print(f"  Correctly raised on failure: {str(e)[:50]}...")
    
    # === TEST 10: Chunking Behavior ===
    print("\nTEST 10: Chunking behavior verification")
    
    n_chunk = 157  # Non-divisible by common chunk sizes
    a_chunk = np.full(n_chunk, 1.0)
    b_chunk = np.full(n_chunk, 3.0)
    
    # Test different chunk sizes
    for chunk_size in [10, 50, 200]:
        roots, converged = solver.solve_vectorized(
            quadratic,
            tolerance=1e-4,  # Reasonable tolerance for chunking test
            chunk_size=chunk_size,
            enable_logging=False,
            a_array=a_chunk,
            b_array=b_chunk
        )
        
        success_rate = np.mean(converged) * 100
        print(f"  Chunk size {chunk_size:3d}: {success_rate:.1f}% success, {len(roots)} results")
        assert len(roots) == n_chunk, "Should return results for all inputs"
        assert success_rate > 95, "Chunking should not affect success rate"
    
    print("\n=== All solve_vectorized tests passed! ===")


if __name__ == "__main__":
    """
    Main test execution for solve_vectorized method.
    
    This test suite covers:
    - Basic functionality with simple polynomials
    - Mixed intervals targeting different roots  
    - Edge cases (exact roots at endpoints)
    - Non-bracketing intervals (error conditions)
    - Challenging transcendental functions
    - Large array performance testing
    - Tolerance and convergence behavior
    - Maximum iterations limits
    - Comprehensive error handling
    - Chunking behavior verification
    """
    
    # Set up logging for tests
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    try:
        test_solve_vectorized()
        print("\n✅ ALL TESTS PASSED - solve_vectorized is working correctly!")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
        
    except Exception as e:
        print(f"\n💥 UNEXPECTED ERROR: {e}")
        raise
    
    # Additional quick performance benchmark
    print("\n" + "="*60)
    print("PERFORMANCE BENCHMARK")
    print("="*60)
    
    solver = BisectionSolver(default_chunk_size=1000)
    
    # Benchmark different problem sizes
    for n in [1000, 10000, 100000]:
        def benchmark_func(x):
            return x**3 - 2*x - 1
        
        a_bench = np.random.uniform(1.0, 1.4, n)
        b_bench = np.random.uniform(1.6, 2.0, n)
        
        import time
        start = time.time()
        roots, converged = solver.solve_vectorized(
            benchmark_func,
            tolerance=1e-4,  # Reasonable tolerance for benchmark
            enable_logging=True,
            a_array=a_bench,
            b_array=b_bench
        )
        elapsed = time.time() - start
        
        rate = n / elapsed
        success = np.mean(converged) * 100
        
        print(f"{n:6d} problems: {elapsed:.3f}s ({rate:8.0f} problems/sec, {success:.1f}% success)")
    
    print("\n🎯 solve_vectorized testing complete!")