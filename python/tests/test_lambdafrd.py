"""Tests for lambdafrd, mirroring the R package's tests/testthat/test-LambdaFRD.R."""

import io
import warnings
from contextlib import redirect_stdout

import numpy as np
import pytest

from lambdafrd import LambdaFRD, LambdaFRDResults


def sim_frd(seed=1, n=1500):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    d = (rng.normal(size=n) < -0.5 + 1 * (x >= 0)).astype(float)
    y = x + 2 * x ** 2 + d + rng.normal(scale=0.3, size=n)
    w = np.column_stack([rng.normal(size=n), rng.uniform(size=n)])
    return y, x, d, w


@pytest.mark.parametrize("p, q", [(1, 2), (1, 3), (2, 3)])
@pytest.mark.parametrize("vce", ["hc0", "hc3", "nn"])
def test_lambda_one_nests_rdrobust(p, q, vce):
    rdrobust = pytest.importorskip("rdrobust").rdrobust
    y, x, d, _ = sim_frd()
    with redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = rdrobust(y, x, fuzzy=d, p=p, q=q, h=0.5, b=0.7, vce=vce)
        m = LambdaFRD(y, x, d, p=p, q=q, h=0.5, b=0.7, _lambda=1, vce=vce)
    assert m["tau_lambda"] == pytest.approx(r.coef.iloc[0, 0], rel=1e-10)
    assert m["tau_BC"] == pytest.approx(r.coef.iloc[1, 0], rel=1e-10)
    assert m["se_conv"] == pytest.approx(r.se.iloc[0, 0], rel=1e-10)
    assert m["se_BC"] == pytest.approx(r.se.iloc[2, 0], rel=1e-10)


def test_default_q_is_p_plus_one():
    y, x, d, _ = sim_frd(2)
    a = LambdaFRD(y, x, d, h=0.4, p=2)
    b = LambdaFRD(y, x, d, h=0.4, p=2, q=3)
    assert a["tau_BC"] == b["tau_BC"]
    assert a["se_BC"] == b["se_BC"]


def test_fields_returned_and_finite():
    y, x, d, _ = sim_frd(3)
    f = LambdaFRD(y, x, d, h=0.4, trace_target="bc")
    for nm in ["lambda", "F_stat", "G_P", "C_P", "kappa_N", "psi_star", "rho_cross",
               "tau_lambda", "ci_lower_conv", "ci_upper_conv",
               "tau_BC", "ci_lower_BC", "ci_upper_BC"]:
        assert np.ndim(f[nm]) == 0 and np.isfinite(f[nm]), nm
    assert 0 < f["lambda"] < 1
    assert isinstance(f, LambdaFRDResults)
    assert list(f.coefficients()) == ["Conventional", "Robust"]
    assert f.coefficients()["Robust"] == f["tau_BC"]


def test_covariates_and_clustering_run(capsys):
    y, x, d, w = sim_frd(4)
    cl = np.random.default_rng(0).integers(1, 51, len(y))
    f = LambdaFRD(y, x, d, h=0.4, b=0.6, covs=w, vce="cluster", cluster=cl)
    assert np.isfinite(f["se_BC"])
    assert f["n_clusters"] == 50
    assert "lambda" in repr(f)
    f.summary()
    assert "Robust" in capsys.readouterr().out


def test_invalid_inputs_rejected():
    y, x, d, _ = sim_frd(6)
    with pytest.raises(ValueError, match="bandwidth"):
        LambdaFRD(y, x, d)
    with pytest.raises(ValueError, match="q must be"):
        LambdaFRD(y, x, d, h=0.4, p=1, q=1)
    with pytest.raises(ValueError, match="cluster"):
        LambdaFRD(y, x, d, h=0.4, vce="cluster")
    with pytest.raises(ValueError, match="same length"):
        LambdaFRD(y[1:], x, d, h=0.4)
    with pytest.raises(ValueError, match="kernel"):
        LambdaFRD(y, x, d, h=0.4, kernel="gaussian")


def test_missing_values_dropped():
    y, x, d, _ = sim_frd(7)
    y_na = y.copy()
    y_na[[2, 9]] = np.nan
    keep = np.ones(len(y), bool)
    keep[[2, 9]] = False
    f = LambdaFRD(y_na, x, d, h=0.4)
    g = LambdaFRD(y[keep], x[keep], d[keep], h=0.4)
    assert f["tau_BC"] == g["tau_BC"]
    assert f["N"] == len(y) - 2


def test_b_less_than_h_warns():
    y, x, d, _ = sim_frd(8)
    with pytest.warns(UserWarning, match="b < h"):
        LambdaFRD(y, x, d, h=0.5, b=0.4)
