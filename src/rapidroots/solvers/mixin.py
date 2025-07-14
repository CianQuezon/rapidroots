import numpy as np

from rapidroots.utils.function_utils import validate_bracket, condition_function
from rapidroots.utils.intelligent_bracket import IntelligentBracketFinder


class BracketPreparationMixin:
    """
    Mixin for conditioning, auto-bracketing and validation
    of [a, b] before any bracketing solver runs.
    """

    def apply_conditioning(self, f_func, should_condition):
        """Stage 1: Apply function conditioning if requested"""
        if should_condition:
            return condition_function(f_func)
        return f_func

    def find_missing_brackets(self, a, b, auto_bracket, func_types, params_array, 
                           initial_guess, preferred_domain, raise_on_fail):
        """Stage 2: Find brackets automatically if requested"""
        if not auto_bracket:
            return a, b
        
        if not (np.isscalar(a) and np.isscalar(b)):
            if self.dispatcher is None or func_types is None or params_array is None:
                if raise_on_fail:
                    raise ValueError("auto_bracket=True requires dispatcher, func_types, and params_array")
                # Return failure for all points
                n_points = len(func_types) if func_types is not None else len(np.asarray(a))
                return np.full(n_points, np.nan), np.zeros(n_points, dtype=bool)
            
            # Use IntelligentBracketFinder to find brackets
            from rapidroots.utils.intelligent_bracket import IntelligentBracketFinder
            bracket_finder = IntelligentBracketFinder(self.dispatcher)
            try:
                a, b = bracket_finder.find_brackets(
                    func_types=func_types,
                    params_array=params_array,
                    initial_guess=initial_guess,
                    preferred_domain=preferred_domain,
                    global_fallback=True,
                    strict=raise_on_fail
                )
            except Exception as e:
                if raise_on_fail:
                    raise RuntimeError(f"Auto-bracketing failed: {e}")
                # Return failure for all points
                n_points = len(func_types)
                return np.full(n_points, np.nan), np.zeros(n_points, dtype=bool)
        
        return a, b

    def validate_all_brackets(self, f_func, a, b, validate_brackets, func_types, params_array, raise_on_fail):
        """Stage 3: Validate brackets if requested"""
         # Handle scalar inputs with basic validation
    
        if np.isscalar(a) and np.isscalar(b):
            fa, fb = f_func(a), f_func(b)
            
            # Check basic bracket conditions
            if a >= b:
                if raise_on_fail:
                    raise ValueError(f"Invalid bracket order: a={a} >= b={b}")
                else:
                    self.logger.warning(f"Invalid bracket order: a={a} >= b={b}")
                    return
            
            if not (np.isfinite(fa) and np.isfinite(fb)):
                if raise_on_fail:
                    raise ValueError(f"Function values not finite: f({a})={fa}, f({b})={fb}")
                else:
                    self.logger.warning(f"Function values not finite: f({a})={fa}, f({b})={fb}")
                    return
            
            if not ((np.sign(fa) != np.sign(fb)) or (fa == 0) or (fb == 0)):
                if raise_on_fail:
                    raise ValueError(f"No sign change: f({a})={fa}, f({b})={fb}")
                else:
                    self.logger.warning(f"No sign change: f({a})={fa}, f({b})={fb}")
                    return
            
            return  # Valid scalar bracket
        
        if self.dispatcher is None or func_types is None or params_array is None:
            # Basic validation without dispatcher
            a_arr = np.asarray(a)
            b_arr = np.asarray(b)
            
            # Simplified basic validation
            fa, fb = f_func(a_arr), f_func(b_arr)
            valid = (a_arr < b_arr) & np.isfinite(fa) & np.isfinite(fb)
            valid &= (np.sign(fa) != np.sign(fb)) | (fa == 0) | (fb == 0)
            
            if not valid.all():
                invalid_indices = np.where(~valid)[0]
                if raise_on_fail:
                    raise ValueError(f"Invalid brackets found at indices: {invalid_indices[:10]}")
                else:
                    self.logger.warning(f"Invalid brackets at {len(invalid_indices)} points")
        else:
            # Use full validation with dispatcher
            is_valid, messages = validate_bracket(
                dispatcher=self.dispatcher,
                func_types=func_types,
                params_array=params_array,
                a=np.asarray(a),
                b=np.asarray(b),
                derivative=0
            )
            
            if not is_valid.all():
                invalid_indices = np.where(~is_valid)[0]
                if raise_on_fail:
                    first_invalid_msg = messages[invalid_indices[0]]
                    raise ValueError(f"Bracket validation failed for {len(invalid_indices)} points. "
                                   f"First error: {first_invalid_msg}")
                else:
                    self.logger.warning(f"Invalid brackets at {len(invalid_indices)} points")
