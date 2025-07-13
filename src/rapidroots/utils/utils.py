import numpy as np

from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher
from numpy.typing import NDArray, ArrayLike
from typing import Callable, Tuple, List, Optional
from numba import njit, prange


class _VectorBracketFinder:

    @staticmethod
    def auto_bracket_search(
        dispatcher: UniversalFunctionSympyDispatcher,
        func_types:    np.ndarray,
        params_array:  np.ndarray,
        initial_guess: ArrayLike      = 0,
        max_range:     float          = 1000,
        base_step:     float          = 0.1,
        expansion_factor: float        = 1.4,
        max_iterations:  Optional[int] = None,
        chunk_size:      int          = 50000,
        concat_threshold: int         = 100000
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """
        Find valid brackets by exponential expansion,
        using dispatcher.evaluate_batch for vectorized f evaluations.
        Switch between one or two dispatcher calls per iteration
        based on concat_threshold to balance copy vs call overhead.
        """
        # — prepare inputs —
        x0   = np.asarray(initial_guess, dtype=float)
        shape= x0.shape
        flat = x0.ravel()
        n    = flat.size

        # validate shapes
        if func_types.shape[0] != n or params_array.shape[0] != n:
            raise ValueError("func_types and params_array must match initial_guess length")

        # compute iterations if needed (hoist log)
        inv_log = 1.0 / np.log(expansion_factor)
        if max_iterations is None:
            max_iterations = max(10, int(np.log(max_range/base_step) * inv_log) + 5)

        # output buffers
        left  = flat.copy()
        right = flat.copy()
        found = np.zeros(n, dtype=bool)

        # initial f0
        f0 = dispatcher.evaluate_batch(func_types, params_array, flat,
                                       derivative=0, chunk_size=chunk_size)

        # mask of points still pending
        active = ~found

        # main loop
        for k in range(max_iterations):
            if not active.any():
                break

            step = base_step * (expansion_factor ** k)
            idx_act = np.nonzero(active)[0]
            x_act   = flat[idx_act]

            # candidate points
            a_vals = x_act - step
            b_vals = x_act + step

            # range‐filter
            legal = (np.abs(a_vals - x_act) <= max_range) & (np.abs(b_vals - x_act) <= max_range)
            if not legal.any():
                break

            idx_leg = idx_act[legal]
            a_leg   = a_vals[legal]
            b_leg   = b_vals[legal]

            ft = func_types[idx_leg]
            pt = params_array[idx_leg]

            # choose single or double dispatcher call
            m = len(a_leg)
            if m >= concat_threshold:
                # single combined call
                combined_x      = np.concatenate([a_leg, b_leg])
                combined_types  = np.repeat(ft, 2)
                combined_params = np.vstack([pt, pt])
                res = dispatcher.evaluate_batch(combined_types,
                                                combined_params,
                                                combined_x,
                                                derivative=0,
                                                chunk_size=chunk_size)
                fa = res[:m]
                fb = res[m:]
            else:
                # two separate calls
                fa = dispatcher.evaluate_batch(ft, pt, a_leg, derivative=0, chunk_size=chunk_size)
                fb = dispatcher.evaluate_batch(ft, pt, b_leg, derivative=0, chunk_size=chunk_size)

            f0_sub = f0[idx_leg]

            # detect sign changes
            ab = (fa * fb) < 0
            oa = (f0_sub * fa) < 0
            ob = (f0_sub * fb) < 0

            # assign brackets
            # ab
            sel = idx_leg[ab]
            left [sel] = a_leg[ab]
            right[sel] = b_leg[ab]
            found[sel] = True

            # oa
            mask_oa = oa & ~ab
            sel = idx_leg[mask_oa]
            left [sel] = a_leg[mask_oa]
            right[sel] = flat[sel]
            found[sel] = True

            # ob
            mask_ob = ob & ~ab & ~oa
            sel = idx_leg[mask_ob]
            left [sel] = flat[sel]
            right[sel] = b_leg[mask_ob]
            found[sel] = True

            # shrink active
            active = ~found

        # final check
        if not found.all():
            bad = np.where(~found)[0]
            raise ValueError(f"Failed to bracket {len(bad)}/{n} points. Indices: {bad[:10]}…")

        return left.reshape(shape), right.reshape(shape)

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
        x = np.linspace(domain[0], domain[1], resolution)
        n = x.size

        # Preallocate f_values
        fvals = np.empty(n, dtype=np.float64)

        # Broadcast types/params only once
        types_full  = np.full(n, func_name)
        params_full = np.tile(params.reshape(1, -1), (n, 1))

        # Chunked evaluations
        for i in range(0, n, chunk_size):
            j = min(i + chunk_size, n)
            fvals[i:j] = dispatcher.evaluate_batch(
                types_full[i:j], params_full[i:j], x[i:j], derivative=0, chunk_size=chunk_size
            )

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