# LambdaFRD
The `LambdaFRD` packages provide facilities for lambda class estimation and inference in fuzzy regression discontinuity (FRD) designs, available in R and python. The theory for this class is found in:

Lane (2025). The moment is here: a generalized class of estimators for fuzzy regression discontinuity designs. Working paper. https://arxiv.org/abs/2511.03424

Please note that this repo is just for the code for estimator and confidence interval implementation. Replication code for the paper can be found in my other repo https://github.com/stuart-lane/LaneLambdaFRD-Replication.

The R and python versions give the same results to numerical precision, and with `lambda = 1` both reproduce the conventional fuzzy RD estimates and standard errors of `rdrobust`.

## Installation and use

Each folder has its own README with an example: [R/README.md](R/README.md) and [python/README.md](python/README.md). Full documentation of the options and returned values is available with `?LambdaFRD` in R and `help(LambdaFRD)` in python.

### R
Install from GitHub:
```r
# install.packages("devtools")
devtools::install_github("stuart-lane/LambdaFRD", subdir="R")
```
```r
library(LambdaFRD)
fit <- LambdaFRD(y, x, fuzzy = d, c = 0, h = 0.4)
summary(fit)
```

### Python
Install from GitHub (requires numpy and scipy, which are installed automatically):
```bash
pip install "git+https://github.com/stuart-lane/LambdaFRD.git#subdirectory=python"
```
```python
from lambdafrd import LambdaFRD
fit = LambdaFRD(y, x, d, c=0, h=0.4)
fit.summary()
```

## Development Status

Current version: 1.0.0

Please report any issues you find, either through the GitHub issues page or by email to stuart.lane@bristol.ac.uk.

## License

GPL (>= 2). See [LICENSE](LICENSE).
