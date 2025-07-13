import numpy as np
from typing import Callable, Union, Tuple
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

@staticmethod
def validate_bracket(
    dispatcher,
    func_types: np.ndarray,
    params_array: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    derivative: int = 0,
    chunk_size: Union[int, None] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized bracket validation using dispatcher - handles ALL brackets at once
    
    Fixed version that properly handles subnormal numbers by testing
    signs directly rather than using multiplication which can underflow.
    
    Args:
        dispatcher: The function dispatcher
        func_types: Array of function type names for each bracket
        params_array: Parameters for each function evaluation
        a: Array of left endpoints
        b: Array of right endpoints  
        derivative: Derivative order (0=f, 1=f', 2=f'')
        chunk_size: Chunk size for dispatcher evaluation (None = auto-select)
        
    Returns:
        Tuple of (is_valid_array, message_array)
        - is_valid_array: boolean array indicating valid brackets
        - message_array: object array with validation messages
        
    Mathematical basis:
    - Bracketing requires f(a) and f(b) to have opposite signs
    - Uses Intermediate Value Theorem: if continuous function changes
      signs over interval, a root exists within that interval
    - Testing signs directly avoids IEEE 754 underflow issues when
      multiplying very small subnormal numbers
    """
    # Convert inputs to arrays
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_brackets = a.size
    
    # Auto-select chunk_size if not provided (avoid overhead for small arrays)
    if chunk_size is None:
        chunk_size = max(10000, n_brackets // 4) if n_brackets > 1000 else n_brackets
    
    try:
        # Vectorized function evaluation using dispatcher
        fa = dispatcher.evaluate_batch(
            func_types, params_array, a,
            derivative=derivative, chunk_size=chunk_size
        )
        fb = dispatcher.evaluate_batch(
            func_types, params_array, b, 
            derivative=derivative, chunk_size=chunk_size
        )
        
        # Fast path: compute validation masks
        is_valid, finite_mask, same_sign_mask = _is_bracket_vectorized(fa, fb)
        
        # Generate messages efficiently (avoiding Python loop overhead)
        messages = _format_bracket_messages(a, b, fa, fb, is_valid, finite_mask, same_sign_mask, n_brackets)
        
    except Exception as e:
        # If dispatcher evaluation fails, mark all as invalid
        is_valid = np.zeros(n_brackets, dtype=bool)
        messages = np.full(n_brackets, f"Function evaluation failed: {str(e)}", dtype=object)
    
    return is_valid, messages


def _is_bracket_vectorized(fa: np.ndarray, fb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fast path: vectorized bracket validation logic without message generation
    
    Returns:
        - is_valid: boolean array of valid brackets
        - finite_mask: mask of brackets with finite function values
        - same_sign_mask: mask of brackets with same signs (for message generation)
    """
    # Vectorized finite value checks
    finite_a = np.isfinite(fa)
    finite_b = np.isfinite(fb)
    both_finite = finite_a & finite_b
    
    # Vectorized sign change detection (avoid underflow)
    # FIXED: Test signs directly instead of product for subnormal safety
    sign_a = np.sign(fa)
    sign_b = np.sign(fb)
    opposite_signs = (sign_a != sign_b) & both_finite
    
    # Handle zero values (special case)
    zero_a = (fa == 0.0) & finite_a
    zero_b = (fb == 0.0) & finite_b
    has_zero = zero_a | zero_b
    
    # Valid brackets: opposite signs OR one endpoint is exactly zero
    is_valid = (opposite_signs | has_zero) & both_finite
    
    # Same sign mask for message generation
    same_sign_mask = both_finite & ~opposite_signs & ~has_zero
    
    return is_valid, both_finite, same_sign_mask


def _format_bracket_messages(
    a: np.ndarray, 
    b: np.ndarray, 
    fa: np.ndarray, 
    fb: np.ndarray,
    is_valid: np.ndarray,
    finite_mask: np.ndarray,
    same_sign_mask: np.ndarray,
    n_brackets: int
) -> np.ndarray:
    """
    Generate diagnostic messages efficiently, minimizing object array overhead
    """
    # Pre-allocate with default invalid message (better than empty strings)
    msgs = np.full(n_brackets, "Invalid bracket", dtype=object)
    
    # Vectorized message assignment
    msgs[is_valid] = "Valid bracket"
    msgs[~finite_mask] = "Function values at endpoints are not finite"
    
    # Build same-sign messages only for needed indices (minimizes Python loop)
    same_indices = np.nonzero(same_sign_mask)[0]
    if len(same_indices) > 0:
        msgs[same_indices] = [
            f"No sign change: f({a[i]:.6g})={fa[i]:.6g}, f({b[i]:.6g})={fb[i]:.6g}"
            for i in same_indices
        ]
    
    return msgs