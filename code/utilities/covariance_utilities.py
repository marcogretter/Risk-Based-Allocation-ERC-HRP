from typing import Any

import numpy as np
import pandas as pd


def _validate_covariance_matrix(
    covariance: np.ndarray,
    name: str,
    require_positive_definite: bool = False,
    positive_definite_message: str | None = None,
) -> np.ndarray:
    """
    Validate a covariance matrix used by the portfolio utilities.

    Parameters:
        covariance (np.ndarray): Covariance matrix to validate.
        name (str): Parameter name used in error messages.
        require_positive_definite (bool): Whether to enforce positive definiteness.
        positive_definite_message (str | None): Optional custom error message for the
            positive-definite check.

    Returns:
        np.ndarray: The validated covariance matrix.
    """
    if not isinstance(covariance, np.ndarray):
        raise TypeError(f"{name} must be numpy array, got {type(covariance)}")

    if covariance.ndim != 2:
        raise ValueError(f"{name} must be 2D array, got shape {covariance.shape}")

    if covariance.shape[0] != covariance.shape[1]:
        raise ValueError(f"{name} must be square, got shape {covariance.shape}")

    if not np.isfinite(covariance).all():
        raise ValueError(f"{name} contains NaN or Inf values")

    if require_positive_definite:
        try:
            np.linalg.cholesky(covariance)
        except np.linalg.LinAlgError as exc:
            if positive_definite_message is None:
                positive_definite_message = f"{name} must be positive definite"
            raise ValueError(positive_definite_message) from exc

    return covariance


def prepare_rolling_estimation_window(
    returns: pd.DataFrame,
    rebalance_date: pd.Timestamp,
    lookback: int,
    min_coverage: float = 0.95,
    return_diagnostics: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    """
    Build a trailing estimation window for covariance-based analysis.

    Parameters:
        returns (pd.DataFrame): Asset return history indexed by date.
        rebalance_date (pd.Timestamp): Last date included in the window.
        lookback (int): Target number of rows in the trailing window.
        min_coverage (float): Minimum fraction of non-missing observations required
            for an asset to remain in the window.
        return_diagnostics (bool): Whether to also return filtering diagnostics.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]: The filtered window, and
            optionally diagnostics describing the retained and dropped assets.
    """
    if not 0 < min_coverage <= 1:
        raise ValueError(
            f"min_coverage must be in the interval (0, 1], got {min_coverage}"
        )

    if lookback <= 0:
        raise ValueError(f"lookback must be positive, got {lookback}")

    trailing_window = (
        returns.loc[:rebalance_date]
        .dropna(axis=0, how="all")
        .dropna(axis=1, how="all")
        .iloc[-lookback:]
    )
    coverage = trailing_window.notna().mean(axis=0)
    eligible_assets = coverage[coverage >= min_coverage].index
    filtered_window = trailing_window.loc[:, eligible_assets].fillna(0.0)

    if not return_diagnostics:
        return filtered_window

    diagnostics = {
        "row_count": trailing_window.shape[0],
        "asset_count_before_filter": trailing_window.shape[1],
        "asset_count_after_filter": filtered_window.shape[1],
        "coverage": coverage.sort_values(ascending=False),
        "retained_assets": list(eligible_assets),
        "dropped_assets": list(coverage[coverage < min_coverage].index),
    }
    return filtered_window, diagnostics


def covariance_to_correlation(
    covariance: np.ndarray,
) -> np.ndarray:
    """
    Given a covariance matrix return the associated correlation matrix.

    Parameters:
        covariance (np.ndarray): Covariance matrix

    Returns:
        np.ndarray: Correlation matrix

    Raises:
        ValueError: If any asset has zero or negative variance, or if matrix contains NaN/Inf
    """

    # Validate the covariance matrix:
    covariance = _validate_covariance_matrix(covariance, name = "covariance")

    # We take the elements of the diagonal of the covariance matrix (ie the variances)
    variances_diag=np.diag(covariance)

    # Check for ≤0 values
    if np.any(variances_diag <= 0):
        raise ValueError("Covariance matrix contains zero or negative variances.")
    
    # From variances to std dev
    std_devs = np.sqrt(variances_diag)

    # Compute the std devs matrix, in order to divide covariance matriz and find the correlation one
    std_devs_mtx = np.outer(std_devs,std_devs)

    corr_mtx = covariance / std_devs_mtx

    # Correction to the elements of the matrix (DA CONFRONTARE CON ALTRI):
    np.fill_diagonal(corr_mtx, 1.0)
    corr_mtx = np.clip(corr_mtx, -1.0, 1.0)

    return corr_mtx


def risk_contribution(portfolios: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """
    Calculate the risk contributions for given portfolios and covariance matrix.

    Parameters:
        portfolios (np.ndarray): An array of shape (n_portfolios, n_assets) representing the
            portfolio weights.
        cov (np.ndarray): The covariance matrix of asset returns (n_assets x n_assets).

    Returns:
        np.ndarray: Risk contributions for each portfolio. Shape is (n_portfolios, n_assets).

    Raises:
        ValueError: If dimensions don't match or inputs contain NaN/Inf
    """

    # Particular case in 1-dim:
    is_1d = False
    if portfolios.ndim == 1:
        portfolios = portfolios.reshape(1, -1)
        is_1d = True

    # Initial control on the inputs:
    if not np.isfinite(portfolios).all() or not np.isfinite(cov).all():
        raise ValueError("Inputs contain NaN or Inf")
    if portfolios.shape[1] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
        raise ValueError("Matrix dimensions do not match")

    # Portfolio volatility:
    # We start by computing covariances for all the ptfs that we have (sigma * w)
    w_times_sigma = portfolios @ cov

    # Compute the tot variance of the ptf
    ptf_var = np.sum(portfolios*w_times_sigma, axis=1, keepdims=True)

    # ptf volatility:
    sigma = np.sqrt(ptf_var)

    # Total risk contribution:
    RC = (portfolios * w_times_sigma)/sigma

    if is_1d:
        return RC.flatten()
    
    return RC
