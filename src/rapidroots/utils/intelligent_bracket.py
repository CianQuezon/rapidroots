import numpy as np
from typing import List, Tuple, Optional, Union, Dict
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher
from rapidroots.utils.bulk_bracket import BulkBracketFinder
from rapidroots.utils.grid_bracket import GridBracketScanner

class IntelligentBracketFinder:
    """
    Production-ready multi-stage bracket finder using existing tested components
    """
    
    def __init__(self, dispatcher: UniversalFunctionSympyDispatcher):
        self.dispatcher = dispatcher
    
    def find_brackets(
        self,
        func_types: np.ndarray,
        params_array: np.ndarray,
        initial_guess: Optional[np.ndarray] = None,
        preferred_domain: Optional[Tuple[float, float]] = None,
        global_fallback: bool = True,
        chunk_size: int = 50000,
        strict: bool = True,
        return_report: bool = False
    ) -> Union[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Dict[str, int]]]:
        """
        Multi-stage smart bracket search with vectorized operations
        
        Args:
            func_types: Array of function type names for each point
            params_array: Parameters for each function evaluation
            initial_guess: Array of initial guess points (optional)
            preferred_domain: Domain to search in (min, max)
            global_fallback: Whether to try global domains if other strategies fail
            chunk_size: Chunk size for dispatcher evaluation
            strict: If True, raise error for failures; if False, return NaN
            return_report: If True, return diagnostic report of success per stage
            
        Returns:
            Tuple of (left_brackets, right_brackets) arrays
            If return_report=True, also returns dict with success counts per stage
        """
        n_points = len(func_types)
        
        # Initialize result arrays
        left_brackets = np.full(n_points, np.nan)
        right_brackets = np.full(n_points, np.nan)
        found_mask = np.zeros(n_points, dtype=bool)
        
        # Diagnostic report
        report = {
            'total_points': n_points,
            'local_search_success': 0,
            'grid_search_success': 0,
            'global_fallback_success': 0,
            'final_failures': 0
        }
        
        # Strategy 1: Local exponential search using existing BulkBracketFinder
        if initial_guess is not None:
            initial_found = found_mask.sum()
            found_mask = self._try_local_search(
                func_types, params_array, initial_guess, 
                left_brackets, right_brackets, found_mask, chunk_size
            )
            report['local_search_success'] = found_mask.sum() - initial_found
        
        # Strategy 2: Preferred domain grid search using existing GridBracketScanner
        if preferred_domain is not None and not found_mask.all():
            initial_found = found_mask.sum()
            found_mask = self._try_grid_search(
                func_types, params_array, preferred_domain,
                left_brackets, right_brackets, found_mask, 
                initial_guess, chunk_size
            )
            report['grid_search_success'] = found_mask.sum() - initial_found
        
        # Strategy 3: Global fallback domains
        if global_fallback and not found_mask.all():
            initial_found = found_mask.sum()
            found_mask = self._try_global_fallback(
                func_types, params_array, initial_guess,
                left_brackets, right_brackets, found_mask, chunk_size
            )
            report['global_fallback_success'] = found_mask.sum() - initial_found
        
        # Handle failures
        report['final_failures'] = (~found_mask).sum()
        if not found_mask.all():
            if strict:
                failed_indices = np.where(~found_mask)[0]
                raise ValueError(f"No valid brackets found for {len(failed_indices)} points at indices: {failed_indices[:10]}...")
            # If not strict, NaN values remain in result arrays
        
        if return_report:
            return left_brackets, right_brackets, report
        return left_brackets, right_brackets
    
    def _try_local_search(
        self, 
        func_types: np.ndarray,
        params_array: np.ndarray,
        initial_guess: np.ndarray,
        left_brackets: np.ndarray,
        right_brackets: np.ndarray,
        found_mask: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        """Strategy 1: Local exponential search using existing BulkBracketFinder"""
        try:
            initial_guess = np.asarray(initial_guess)
            if initial_guess.size == 1:
                initial_guess = np.full(len(func_types), initial_guess.item())
            
            # Use existing tested auto_bracket_search
            left_local, right_local = BulkBracketFinder.auto_bracket_search(
                self.dispatcher, func_types, params_array, 
                initial_guess=initial_guess, chunk_size=chunk_size
            )
            
            # Check which points found valid brackets
            local_valid = np.isfinite(left_local) & np.isfinite(right_local)
            
            # Update results for successful local searches
            left_brackets[local_valid] = left_local[local_valid]
            right_brackets[local_valid] = right_local[local_valid]
            found_mask[local_valid] = True
            
        except Exception as e:
            print(f"Strategy 1 (local search) failed: {e}")
        
        return found_mask
    
    def _try_grid_search(
        self,
        func_types: np.ndarray,
        params_array: np.ndarray,
        domain: Tuple[float, float],
        left_brackets: np.ndarray,
        right_brackets: np.ndarray,
        found_mask: np.ndarray,
        initial_guess: Optional[np.ndarray],
        chunk_size: int
    ) -> np.ndarray:
        """Strategy 2: Grid search using existing GridBracketScanner"""
        remaining_mask = ~found_mask
        if not remaining_mask.any():
            return found_mask
        
        # Precompute unique function types and their masks once
        unique_types = np.unique(func_types[remaining_mask])
        type_masks = {}
        for func_name in unique_types:
            type_mask = func_types == func_name
            remaining_type_mask = remaining_mask & type_mask
            type_masks[func_name] = remaining_type_mask
        
        # Process each function type
        for func_name, remaining_type_mask in type_masks.items():
            type_indices = np.where(remaining_type_mask)[0]
            
            if len(type_indices) == 0:
                continue
            
            # Process each function of this type
            for idx in type_indices:
                if found_mask[idx]:  # Skip if already found
                    continue
                    
                try:
                    params = params_array[idx]
                    
                    # Use existing tested GridBracketScanner
                    brackets = GridBracketScanner.find_all_brackets(
                        self.dispatcher, func_name, params,
                        domain=domain, chunk_size=chunk_size
                    )
                    
                    if brackets:
                        # Select best bracket based on initial guess if available
                        if initial_guess is not None and not np.isnan(initial_guess[idx]):
                            best_bracket = self._select_closest_bracket(brackets, initial_guess[idx])
                        else:
                            best_bracket = brackets[0]
                        
                        left_brackets[idx] = best_bracket[0]
                        right_brackets[idx] = best_bracket[1]
                        found_mask[idx] = True
                        
                except Exception:
                    continue
        
        return found_mask
    
    def _try_global_fallback(
        self,
        func_types: np.ndarray,
        params_array: np.ndarray,
        initial_guess: Optional[np.ndarray],
        left_brackets: np.ndarray,
        right_brackets: np.ndarray,
        found_mask: np.ndarray,
        chunk_size: int
    ) -> np.ndarray:
        """Strategy 3: Global domain sweep using existing components"""
        global_domains = [(-1, 1), (-10, 10), (-100, 100), (-1000, 1000)]
        
        for domain in global_domains:
            if found_mask.all():
                break
            
            found_mask = self._try_grid_search(
                func_types, params_array, domain,
                left_brackets, right_brackets, found_mask,
                initial_guess, chunk_size
            )
        
        return found_mask
    
    def _select_closest_bracket(
        self, 
        brackets: List[Tuple[float, float]], 
        target: float
    ) -> Tuple[float, float]:
        """Select the bracket closest to the target point"""
        def distance_to_bracket(bracket: Tuple[float, float]) -> float:
            a, b = bracket
            if a > b:
                a, b = b, a
            
            if a <= target <= b:
                return 0.0
            elif target < a:
                return a - target
            else:
                return target - b
        
        return min(brackets, key=distance_to_bracket)