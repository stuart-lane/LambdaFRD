"""Results container returned by :func:`lambdafrd.LambdaFRD`."""

"""
PACKAGE SETUP
"""

import numpy as np
from scipy import stats

"""
RESULTS CONTAINER
"""

class LambdaFRDResults(dict):

    """Results of a :func:`~lambdafrd.LambdaFRD` fit.

    A ``dict`` subclass, so every quantity is available by key, e.g.
    ``fit['tau_BC']`` or ``fit['ci_lower_BC']``. See :func:`~lambdafrd.LambdaFRD`
    for the list of keys. It adds the Python equivalents of the R package's
    ``print``, ``summary`` and ``coef`` methods:

    - ``repr(fit)`` (or just ``fit`` at the prompt): short description of the fit.
    - :meth:`summary`: rdrobust-style table of estimates.
    - :meth:`coefficients`: conventional and bias-corrected point estimates.
    """

    def __repr__(self) -> str:

        """Short description: sample size, kernel, VCE method and lambda"""
        lines = [
            f"{'Number of Obs.':<22} {int(self['N']):>12d}",
            f"{'Kernel':<22} {self['kernel']:>12s}",
            f"{'VCE method':<22} {self['vce'].upper():>12s}",
            f"{'lambda':<22} {self['lambda']:>12.4f}"
        ]
        return "\n".join(lines)

    def coefficients(self) -> dict:

        """Conventional and bias-corrected point estimates.

        Returns
        -------
        dict
            ``{'Conventional': tau_lambda, 'Robust': tau_BC}``.
        """
        return {'Conventional': float(self['tau_lambda']), 'Robust': float(self['tau_BC'])}

    def summary(self) -> None:

        """Print an rdrobust-style summary table.

        Shows the sample and specification (observations, kernel, VCE method,
        lambda, first-stage F, effective observations, polynomial orders and
        bandwidths on each side), then the conventional and robust
        bias-corrected estimates with standard errors, z-statistics for
        ``tau = tau_0``, p-values and ``(1 - alpha)`` confidence intervals.
        """
        print("Fuzzy RD estimates using the lambda-class estimator.\n")

        print(f"{'Number of Obs.':<22} {int(self['N']):>12d}")
        print(f"{'Kernel':<22} {self['kernel']:>12s}")
        print(f"{'VCE method':<22} {self['vce'].upper():>12s}")
        print(f"{'lambda':<22} {self['lambda']:>12.4f}")
        print(f"{'First-stage F':<22} {self['F_stat']:>12.3f}")
        print()

        print(f"{'':<22} {'Left':>12s} {'Right':>12s}")
        print(f"{'Eff. Number of Obs.':<22} {self['N_h']['left']:>12d} {self['N_h']['right']:>12d}")
        print(f"{'Order est. (p)':<22} {self['p']:>12d} {self['p']:>12d}")
        print(f"{'Order bias (q)':<22} {self['q']:>12d} {self['q']:>12d}")
        print(f"{'BW est. (h)':<22} {self['h']:>12.3f} {self['h']:>12.3f}")
        print(f"{'BW bias (b)':<22} {self['b']:>12.3f} {self['b']:>12.3f}")
        print()

        z = (self['coef'] - self['tau_0']) / self['se']
        pv = 2 * stats.norm.cdf(-np.abs(z))
        lvl = round(100 * (1 - self['alpha']))
        line = "=" * 77
        print(line)
        print(f"{'Method':>14s} {'Coef.':>10s} {'Std. Err.':>10s} {'z':>10s} "
              f"{'P>|z|':>10s} {f'[ {lvl}% C.I. ]':>18s}")
        print(line)
        for i, method in enumerate(("Conventional", "Robust")):
            print(f"{method:>14s} {self['coef'][i]:>10.3f} {self['se'][i]:>10.3f} "
                  f"{z[i]:>10.3f} {pv[i]:>10.3f}   "
                  f"[{self['ci'][i, 0]:>6.3f} , {self['ci'][i, 1]:>6.3f}]")
        print(line)
