from __future__ import annotations

from typing import Any, Literal

import numpy as np
from scipy import stats

FitMethod = Literal["ols", "odr"]


class FitError(ValueError):
    """Raised when a requested regression cannot be computed reliably."""


def validate_fit_method(fit_method: str) -> FitMethod:
    if fit_method not in {"ols", "odr"}:
        raise FitError("fit_method must be one of: 'ols', 'odr'")
    return fit_method  # type: ignore[return-value]


def _linear_model(x: np.ndarray, beta: np.ndarray) -> np.ndarray:
    return beta[0] + beta[1] * x


def fit_relation(
    log_x: np.ndarray,
    log_y: np.ndarray,
    fit_method: str = "ols",
    sigma_x: np.ndarray | None = None,
    sigma_y: np.ndarray | None = None,
) -> dict[str, Any]:
    """Fit a straight line in log10 space with OLS or weighted ODR."""

    method = validate_fit_method(fit_method)
    if method == "ols":
        if sigma_x is not None or sigma_y is not None:
            raise FitError("Measurement uncertainties can only be used with ODR")
        result = stats.linregress(log_x, log_y)
        slope = float(result.slope)
        intercept = float(result.intercept)
        return {
            "fit_method": "ols",
            "slope": slope,
            "intercept": intercept,
            "slope_std_error": float(result.stderr),
            "intercept_std_error": float(result.intercept_stderr),
            "stderr": float(result.stderr),
            "r_value": float(result.rvalue),
            "pearson_r": float(result.rvalue),
            "p_value": float(result.pvalue),
            "diagnostics": {
                "weighted": False,
                "converged": True,
            },
        }

    from odrpack import odr_fit

    beta0 = np.asarray(np.polyfit(log_x, log_y, 1)[::-1], dtype=float)
    weight_x = None if sigma_x is None else 1.0 / np.square(sigma_x)
    weight_y = None if sigma_y is None else 1.0 / np.square(sigma_y)
    result = odr_fit(
        _linear_model,
        log_x,
        log_y,
        beta0,
        weight_x=weight_x,
        weight_y=weight_y,
        maxit=100,
    )
    if not result.success:
        raise FitError(f"ODR did not converge: {result.stopreason}")

    slope = float(result.beta[1])
    intercept = float(result.beta[0])
    return {
        "fit_method": "odr",
        "slope": slope,
        "intercept": intercept,
        "slope_std_error": float(result.sd_beta[1]),
        "intercept_std_error": float(result.sd_beta[0]),
        "stderr": float(result.sd_beta[1]),
        "r_value": float(np.corrcoef(log_x, log_y)[0, 1]),
        "pearson_r": float(np.corrcoef(log_x, log_y)[0, 1]),
        "p_value": None,
        "diagnostics": {
            "weighted": sigma_x is not None or sigma_y is not None,
            "converged": bool(result.success),
            "stop_reason": result.stopreason,
            "residual_variance": float(result.res_var),
            "sum_square": float(result.sum_square),
        },
    }


def fit_with_bootstrap(
    log_x: np.ndarray,
    log_y: np.ndarray,
    fit_method: str = "ols",
    sigma_x: np.ndarray | None = None,
    sigma_y: np.ndarray | None = None,
    bootstrap: int = 300,
) -> tuple[dict[str, Any], list[float]]:
    """Fit once and return successful bootstrap slope estimates."""

    method = validate_fit_method(fit_method)
    payload = fit_relation(log_x, log_y, method, sigma_x, sigma_y)
    iterations = max(0, min(int(bootstrap), 2000))
    slopes: list[float] = []
    if not iterations:
        return payload, slopes

    rng = np.random.default_rng(20260927)
    n = len(log_x)
    for _ in range(iterations):
        idx = rng.integers(0, n, n)
        if np.std(log_x[idx]) == 0:
            continue
        try:
            resampled = fit_relation(
                log_x[idx],
                log_y[idx],
                method,
                None if sigma_x is None else sigma_x[idx],
                None if sigma_y is None else sigma_y[idx],
            )
        except (FitError, ValueError, FloatingPointError):
            continue
        slopes.append(float(resampled["slope"]))
    return payload, slopes
