import numpy as np
import numpy.typing as npt

from numba import cuda
from typing import Optional
from typing_extensions import Literal, Sequence
from .registry import get_family, all_families, registry_version

class UniversalFunctionDispatcher:


    def __init__(self):
        self._family_indices = None
        self._known_registry_version: int = registry_version()
        self._warmup_done = False

    def evaluate_batch(self,
                       func_types: Sequence[str],
                       params_arr: Sequence[Sequence[float]],
                       x_arr: npt.ArrayLike,
                       derivatives: Literal[0, 1, 2] = 0,
                       chunk_size: int = 10_000,
                       dtype: Optional[npt.DTypeLike] = np.float64) -> npt.NDArray:
        
        # Invalidate if registry has changed
        current_version = registry_version()
        if current_version != self._known_registry_version:
            self._warmup_done = False
            self._family_indices = None
            self._known_registry_version = current_version
        
        # Warm up JIT once
        if not self._warmup_done:
            self._warmup()
        
        # Encode func_types into integer codes
        uniques, codes, x_array, _, _ = self._prepare_batch(
            func_types=func_types,
            params_arr=params_arr,
            x_arr=x_arr,
            dtype=dtype
        )

        # if x_array is zero then return an empty numpy array
        if x_array.size == 0:
            return np.array([], dtype=dtype)

        n = x_array.size
        out = np.empty(n, dtype=dtype)

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            xs = x_array[start:end]
            codes_chunk = codes[start:end]
            chunk_idx = np.arange(start, end)
            chunk_out = np.empty(end - start, dtype=dtype)

            # for each family-code, call evaluate once
            for code in np.unique(codes_chunk):
                mask = (codes_chunk == code)
                fam = get_family(uniques[code])

                # global indices for this family in the chunk
                global_idx = chunk_idx[mask]
                
                
                p_sub = [params_arr[i] for i in global_idx]

                # Fix parameter array shape
                if len(p_sub) == 1:
                    p_arr = np.array(p_sub[0], dtype=dtype).reshape(1, -1)
                else:
                    p_arr = np.array(p_sub, dtype=dtype)
                
                # Call fam.evaluate
                result = fam.evaluate(xs[mask], p_arr, derivatives)
                
                # Handle multiple return values
                if result.size != xs[mask].size:
                    n_inputs = xs[mask].size
                    result = result[:n_inputs]  # Take only the function values
                
                chunk_out[mask] = result

            out[start:end] = chunk_out
        
        return out
    
    def _warmup(self, dtype: Optional[npt.DTypeLike] = np.float64):
        
        for fam in all_families().values():

            # compile all derivatives on a 1-element array
            dummy = np.asarray([0.0], dtype=dtype)
            p = np.zeros((1, fam.param_count), dtype=dtype)

            fam.evaluate(dummy, p, 0)
            fam.evaluate(dummy, p, 1)
            fam.evaluate(dummy, p, 2)
        self._warmup_done = True

    def _prepare_batch(self,
                       func_types: Sequence[str],
                       params_arr: Sequence[Sequence[float]],
                       x_arr: npt.ArrayLike,
                       dtype: Optional[npt.DTypeLike] = np.float64):
        
        # Safety Guard check 
        if len(func_types) == 0:
            x = np.asarray(x_arr, dtype=dtype)
            return np.array([], dtype=str), np.array([], dtype=np.int32), x, None, None

        
        # Encode families into integer once per batch
        uniques, inv = np.unique(func_types, return_inverse=True)
        codes = inv.astype(np.int32)

        x = np.asarray(x_arr, dtype=dtype)

        # Find max parameter count and pad all parameter vectors to the same length
        families = [get_family(name) for name in uniques]
        param_counts = np.array([fam.param_count for fam in families],
                                dtype=np.int32)
        max_p = param_counts.max()
        n = x.size
        params = np.zeros((n, max_p), dtype=dtype)

        # fill the padded params block 
        for i, p_i in enumerate(params_arr):
            params[i, : len(p_i)] = p_i

        return uniques, codes, x, params, param_counts