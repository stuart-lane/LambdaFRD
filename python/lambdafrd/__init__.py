"""LambdaFRD: lambda-class estimation for fuzzy regression discontinuity.

Lambda-class estimation and robust bias-corrected inference for fuzzy
regression discontinuity designs (Lane, 2025). The main function is
:func:`LambdaFRD`, which returns a :class:`LambdaFRDResults` object with the
uncorrected and robust bias-corrected estimates, standard errors and
confidence intervals.

Example
-------
>>> from lambdafrd import LambdaFRD
>>> fit = LambdaFRD(y, x, d, c=0, h=0.4)    # doctest: +SKIP
>>> fit.summary()                           # doctest: +SKIP

Use ``help(LambdaFRD)`` for the options.
"""

"""
PACKAGE SETUP
"""

from .lambdafrd import LambdaFRD
from .results import LambdaFRDResults

__version__ = "1.0.0"

__all__ = ["LambdaFRD", "LambdaFRDResults", "__version__"]
