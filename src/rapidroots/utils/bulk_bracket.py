import numpy as np
from numpy.typing import NDArray, ArrayLike
from typing import Tuple, Optional
from rapidroots.core.dispatcher import UniversalFunctionSympyDispatcher


class BulkBracketFinder:

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