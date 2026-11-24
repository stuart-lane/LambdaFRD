# lambdafrd

Lambda-class estimation and robust bias-corrected inference for fuzzy regression
discontinuity designs. Returns uncorrected and robust bias-corrected estimates,
standard errors and confidence intervals.

This is the Python port of the R package `LambdaFRD`. The two give the same
results, to numerical precision.

## Installation

```bash
pip install "git+https://github.com/stuart-lane/LambdaFRD.git#subdirectory=python"
```

or, from a local copy of this folder:

```bash
pip install .
```

## Example

```python
from lambdafrd import LambdaFRD

fit = LambdaFRD(y, x, d, c=0, h=0.4)
fit.summary()
fit["tau_BC"], fit["ci_lower_BC"], fit["ci_upper_BC"]
```

See `help(LambdaFRD)` for the options and the returned fields.

### Differences from the R interface

| R                          | Python                                  |
|----------------------------|-----------------------------------------|
| `LambdaFRD(y, x, fuzzy = d, ...)` | `LambdaFRD(y, x, d, ...)` or `fuzzy=d`  |
| `lambda = 1`               | `_lambda=1` (`lambda` is reserved in Python) |
| `fit$tau_BC`               | `fit["tau_BC"]`                         |
| `print(fit)`               | `print(fit)` / `fit`                    |
| `summary(fit)`             | `fit.summary()`                         |
| `coef(fit)`                | `fit.coefficients()`                    |
| `fit$coef`, `fit$se`, `fit$ci` (named matrices) | NumPy arrays, rows `[Conventional, Robust]` |

## Tests

```bash
pip install ".[test]"
pytest
```

## Reference

Lane, S. (2025). *The moment is here: a generalized class of estimators for fuzzy
regression discontinuity designs.* https://arxiv.org/abs/2511.03424

## License

GPL (>= 2), as for the R package.
