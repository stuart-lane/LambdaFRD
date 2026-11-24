# LambdaFRD

Lambda-class estimation and robust bias-corrected inference for fuzzy regression discontinuity
designs. Returns uncorrected and robust bias-corrected estimates, standard
errors and confidence intervals.

## Installation

```r
# install.packages("devtools")
devtools::install_github("stuart-lane/LambdaFRD", subdir="R")
```

## Example

```r
library(LambdaFRD)
fit <- LambdaFRD(y, x, fuzzy = d, c = 0, h = 0.4)
summary(fit)
```

See `?LambdaFRD` for the options.

## Reference

Lane, S. (2025). *The moment is here: a generalized class of estimators for fuzzy
regression discontinuity designs.* https://arxiv.org/abs/2511.03424

