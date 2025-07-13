import numpy as np
from typing import List, Tuple
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher


class GridBracketScanner:

    @staticmethod
    def find_all_brackets(
        dispatcher: UniversalFunctionSympyDispatcher,
        func_name:  str,
        params:     np.ndarray,
        domain:     Tuple[float, float] = (-10, 10),
        resolution: int                 = 1000,
        chunk_size: int                 = 10000,
        mag_pct:    float               = 95.0,
        mag_fac:    float               = 10.0
    ) -> List[Tuple[float, float]]:
        """
        Fully‐vectorized single‐function bracket finder with chunking
        and adaptive magnitude filtering.
        """
        # Validate inputs
        if domain[1] <= domain[0] or resolution < 2:
            return []



        x = np.linspace(domain[0], domain[1], resolution)
        n = x.size

        # Preallocate f_values
        fvals = np.empty(n, dtype=np.float64)

        # Broadcast types/params only once
        types_full  = np.full(n, func_name)
        params_full = np.tile(params.reshape(1, -1), (n, 1))

        # Chunked evaluations with exception handling
        try:
            for i in range(0, n, chunk_size):
                j = min(i + chunk_size, n)
                fvals[i:j] = dispatcher.evaluate_batch(
                    types_full[i:j], params_full[i:j], x[i:j], derivative=0, chunk_size=chunk_size
                )
        
        except Exception as e:
            raise RuntimeError(f"Chunking Evaluation failed: {e}")

        # Masks & shifted views
        fa, fb   = fvals[:-1],     fvals[1:]
        xa, xb   = x[:-1],         x[1:]
        ok       = np.isfinite(fa) & np.isfinite(fb)

        # Sign/zero detection
        sign_change = (fa * fb < 0)
        zero_cross  = ((fa == 0) & (fb != 0)) | ((fb == 0) & (fa != 0))

        # Adaptive magnitude filter
        if mag_fac > 0:
            finite_vals = np.abs(fvals[np.isfinite(fvals)])
            if finite_vals.size:
                thresh = np.percentile(finite_vals, mag_pct) * mag_fac
            else:
                thresh = np.inf
            mag_ok = (np.abs(fa) < thresh) & (np.abs(fb) < thresh)
        else:
            mag_ok = np.ones_like(ok)

        # Final mask
        mask = ok & (sign_change | zero_cross) & mag_ok

        # Return bracket pairs
        return list(zip(xa[mask].tolist(), xb[mask].tolist()))