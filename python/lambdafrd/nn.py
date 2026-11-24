"""rdrobust-style same-side nearest-neighbour residuals.

Internal module; not part of the public API.
"""

"""
PACKAGE SETUP
"""

import numpy as np

"""
RDROBUST-STYLE SAME-SIDE NEAREST-NEIGHBOUR RESIDUALS

For each i, take the J nearest observations in x ON THE SAME SIDE of the
cutoff (the caller splits by side) and return

    sqrt(J_i / (J_i + 1)) * ( data_i - mean_{j in NN(i)} data_j ),

where J_i >= J is the realised number of matches: all observations tied at
the J-th nearest distance are included, which matters when x has mass points.
Returns rows in the ORIGINAL input order.
"""

def nn_residuals_side(
        x: np.ndarray,
        data: np.ndarray,
        J: int
    ) -> np.ndarray:

    """Nearest-neighbour residuals for every column of data, on one side"""
    data = np.asarray(data, dtype=float)
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    n = len(x)
    if n < 2:
        return np.full((n, data.shape[1]), np.nan)

    # Sort by x, keeping ties in their original order
    ord_ = np.argsort(x, kind="stable")
    xs = x[ord_]
    ds = data[ord_, :]

    # Size of each tie group (dups) and position within it (dupsid, 1-based)
    _, counts = np.unique(xs, return_counts=True)
    dups = np.repeat(counts, counts)
    dupsid = np.concatenate([np.arange(1, cnt + 1) for cnt in counts])

    matches = min(int(J), n - 1)
    res = np.zeros((n, ds.shape[1]))
    nn_tol_eps = np.sqrt(np.finfo(float).eps)

    for i in range(n):
        rpos = dups[i] - dupsid[i]
        lpos = dupsid[i] - 1
        while lpos + rpos < matches:
            if i - lpos - 1 < 0:
                rpos += dups[i + rpos + 1]
            elif i + rpos + 1 > n - 1:
                lpos += dups[i - lpos - 1]
            else:
                dleft = xs[i] - xs[i - lpos - 1]
                dright = xs[i + rpos + 1] - xs[i]
                nn_tol = max(dleft, dright) * nn_tol_eps
                if dleft - dright > nn_tol:
                    rpos += dups[i + rpos + 1]
                elif dright - dleft > nn_tol:
                    lpos += dups[i - lpos - 1]
                else:
                    rpos += dups[i + rpos + 1]
                    lpos += dups[i - lpos - 1]

        lo = i - lpos
        hi = i + rpos
        Ji = hi - lo
        sf = np.sqrt(Ji / (Ji + 1))
        neigh_sum = ds[lo:hi + 1, :].sum(axis=0) - ds[i, :]
        res[i, :] = sf * (ds[i, :] - neigh_sum / Ji)

    # Back to the original order
    out = np.zeros_like(res)
    out[ord_, :] = res

    return out
