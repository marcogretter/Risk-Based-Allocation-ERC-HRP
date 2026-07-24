"""
Portfolio optimization utilities.

Implements:
- Minimum variance portfolio (closed-form solution)
- Mean-variance portfolio (closed-form solution)
- Equal risk contribution portfolio (optimization-based)

References:
- Maillard, S., Roncalli, T., & Teïletche, J. (2010).
  "The properties of equally weighted risk contribution portfolios."
- Markowitz, H. (1952). "Portfolio Selection." The Journal of Finance.
"""

from typing import Any
import numpy as np
from functools import partial
from scipy.optimize import minimize
from utilities.covariance_utilities import (
    _validate_covariance_matrix,
    risk_contribution,
)


def minimum_variance_portfolio(cov_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the minimum variance portfolio weights given a covariance matrix.
    In particular the weights are given by:
    w = (Σ^(-1) * 1) / (1^T * Σ^(-1) * 1), i.e. the solution of the optimization problem:
    min_w w^T * Σ * w, subject to 1^T * w = 1.
 
    Parameters:
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
 
    Returns:
        np.ndarray: Weights of the minimum variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )
 
    n = cov_matrix.shape[0]
    ones_vec = np.ones((n, 1))
 
    # Σ^(-1) · 1  — solve the linear system instead of explicitly inverting Σ
    # (numerically more stable than np.linalg.inv)
    cov_inv_ones = np.linalg.solve(cov_matrix, ones_vec)          # shape (n, 1)
 
    # Normalisation scalar: 1^T · Σ^(-1) · 1
    min_var_ptf_numerator = cov_inv_ones                           # Σ^(-1) · 1
    min_var_ptf_weights = min_var_ptf_numerator / (ones_vec.T @ cov_inv_ones)  # shape (n, 1)
 
    return min_var_ptf_weights.flatten()


def mean_variance_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
) -> np.ndarray:
    """
    Calculate the classic mean-variance portfolio weights given expected returns and a
    covariance matrix.

    In particular the weights solve:
    max_w mu^T * w - (gamma / 2) * w^T * Sigma * w, subject to 1^T * w = 1,
    where mu are the expected returns and gamma is the risk-aversion parameter.

    Parameters:
        expected_returns (np.ndarray): Expected returns vector.
        cov_matrix (np.ndarray): Covariance matrix of asset returns.
        risk_aversion (float): Risk-aversion parameter gamma. Must be strictly positive.

    Returns:
        np.ndarray: Weights of the mean-variance portfolio.
    """
    cov_matrix = _validate_covariance_matrix(
        cov_matrix,
        name="cov_matrix",
        require_positive_definite=True,
        positive_definite_message=(
            "cov_matrix must be positive definite (symmetric with positive eigenvalues)"
        ),
    )

    expected_returns = np.asarray(expected_returns, dtype=float)
    if expected_returns.ndim == 2 and 1 in expected_returns.shape:
        expected_returns = expected_returns.reshape(-1)
    elif expected_returns.ndim != 1:
        raise ValueError(
            "expected_returns must be one-dimensional or a single-column vector"
        )

    if expected_returns.shape[0] != cov_matrix.shape[0]:
        raise ValueError(
            "expected_returns and cov_matrix must refer to the same number of assets, "
            f"got {expected_returns.shape[0]} and {cov_matrix.shape[0]}"
        )

    if not np.isfinite(expected_returns).all():
        raise ValueError("expected_returns contains NaN or Inf values")

    if not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be finite")

    if risk_aversion <= 0:
        raise ValueError(
            f"risk_aversion must be strictly positive, got {risk_aversion}"
        )
    n = cov_matrix.shape[0]
    ones_vec = np.ones(n)
 
    # Solve Σ · x = mu  and  Σ · y = 1  simultaneously (avoids two separate solves)
    # np.linalg.solve handles multiple right-hand sides stacked as columns.
    rhs = np.column_stack([expected_returns, ones_vec])       # shape (n, 2)
    sol = np.linalg.solve(cov_matrix, rhs)                    # shape (n, 2)
    cov_inv_mu   = sol[:, 0]                                  # Σ^(-1) · mu
    cov_inv_ones = sol[:, 1]                                  # Σ^(-1) · 1
 
    # Scalar quantities for the Lagrangian solution
    A = ones_vec @ cov_inv_ones   # 1^T Σ^(-1) 1
    B = ones_vec @ cov_inv_mu     # 1^T Σ^(-1) mu
 
    # Lagrange multiplier lambda* so that weights sum to 1:
    #   lambda* = (1 - (1/gamma) * (B - A * lambda*)) ... solved analytically:
    #   lambda* = (gamma - B) / A   ... from the full KKT system
    #
    # Full closed-form:
    #   w* = (1/gamma) * Σ^(-1) * mu + [1 - (B/gamma)] / A * Σ^(-1) * 1
    lam = (1.0 / risk_aversion) * (B / A) - (1.0 / A)   # NOT used directly below;
    # cleaner to write the two-fund separation directly:
    # w* = w_mv  +  (1/gamma) * Σ^(-1) * (mu - mu_mv · 1)
    # where mu_mv = B / A  is the min-var portfolio mean return
    mu_mv = B / A
    w_mv  = cov_inv_ones / A                              # minimum-variance weights
 
    mean_var_ptf_weights = w_mv + (1.0 / risk_aversion) * (cov_inv_mu - mu_mv * cov_inv_ones)
    

    return mean_var_ptf_weights.flatten()


def inverse_volatility_portfolio(covariance: np.ndarray) -> np.ndarray:
    """
    Compute the inverse-volatility (naive risk-parity) portfolio.

    Each weight is proportional to the inverse of the asset's volatility,
    ``w_i ∝ 1 / sigma_i``, with ``sum_i w_i = 1``. Lab notes Question 2 shows
    that this coincides with the ERC solution when the assets are uncorrelated.

    Parameters:
        covariance (np.ndarray): Covariance matrix of asset returns. Only the
            diagonal is used.

    Returns:
        np.ndarray: Inverse-volatility weights summing to 1.
    """

    # Extract per-asset volatilities from the diagonal of the covariance matrix
    vols = np.sqrt(np.diag(covariance))       # σ_i = sqrt(Σ_ii)
 
    if np.any(vols <= 0):
        raise ValueError(
            "All diagonal entries of covariance must be strictly positive "
            "(i.e. every asset must have positive variance)."
        )
 
    inv_vols = 1.0 / vols                     # 1/σ_i
    weights  = inv_vols / inv_vols.sum()      # normalise so Σ w_i = 1
 
    return weights
    


def erc_objective_function(weights: np.ndarray, covariance: np.ndarray) -> float:
    """
    ERC objective based on percentage risk contributions.

    At the ERC solution, each asset should contribute 1/N of total portfolio risk.
    """
    marginal_contrib = covariance @ weights
    portfolio_var = weights @ marginal_contrib

    if portfolio_var <= 0:
        return 1e10

    total_rc = weights * marginal_contrib
    pcr = total_rc / portfolio_var

    target = np.full_like(pcr, 1.0 / len(weights))

    return np.sum((pcr - target) ** 2)


def equal_risk_contribution_portfolio(
    covariance: np.ndarray,
    initial_solution: np.ndarray | None = None,
    options: dict[str, Any] | None = None,
    pcr_tolerance: float = 0.005,
    ignore_objective: bool = False,
) -> np.ndarray:
    """
    Calculate the equal risk contribution portfolio.

    Parameters:
        covariance (np.ndarray): Covariance matrix of assets, must be positive definite.
        initial_solution (np.ndarray | None): Initial solution guess, default to
            None, i.e. to the inverse volatility portfolio.
        options (Dict[str, Any] | None): A dictionary of solver options, see
            scipy.optimize.minimize.
        pcr_tolerance (float): The max allowable tolerance for differences in the percentage
            contribution to risk (pcr) coming from different assets, default to 10bps.

    Returns:
        np.ndarray: Equal risk contribution portfolio.
    """
    covariance = _validate_covariance_matrix(
        covariance,
        name="covariance",
        require_positive_definite=True,
        positive_definite_message=(
            "covariance must be positive definite (symmetric with positive eigenvalues)"
        ),
    )
 
    n = covariance.shape[0]
 
    # Initial guess
    # Default: inverse-volatility portfolio (warm-start close to the ERC solution)
    if initial_solution is None:
        initial_solution = inverse_volatility_portfolio(covariance)
    else:
        initial_solution = np.asarray(initial_solution, dtype=float).flatten()
        if initial_solution.shape[0] != n:
            raise ValueError(
                f"initial_solution length {initial_solution.shape[0]} does not match "
                f"covariance dimension {n}"
            )
        # Normalise user-provided guess to lie on the simplex
        initial_solution = initial_solution / initial_solution.sum()
 
    # Solver options
    default_options = {"ftol": 1e-14, "maxiter": 5_000, "disp": False}
    if options is not None:
        default_options.update(options)
 
    # Constraints and bounds 
    # Weights must sum to 1 (equality constraint) and be non-negative (long-only)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(1e-8, 1.0)] * n   # small positive lower bound for numerical stability
 
    # Objective (partially applied with the covariance matrix) 
    objective = partial(erc_objective_function, covariance=covariance)
 
    # Optimise
    result = minimize(
        objective,
        initial_solution,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options=default_options,
    )
 
    weights = result.x
 
    # Re-normalise to correct for any tiny numerical drift off the simplex
    weights = weights / weights.sum()
 
    # Post-optimisation sanity check
    # Verify that the percentage contribution to risk (PCR) is approximately equal
    # across all assets, up to pcr_tolerance.
    if not ignore_objective:
        rc = risk_contribution(weights, covariance)
        pcr = rc / rc.sum()
        pcr_range = pcr.max() - pcr.min()

        if pcr_range > pcr_tolerance:
            raise RuntimeError(
                f"ERC optimisation did not converge to equal risk contributions. "
                f"Max PCR dispersion: {pcr_range:.6f} (tolerance: {pcr_tolerance}). "
                f"Solver message: '{result.message}'. "
                "Try a different initial_solution or relax pcr_tolerance."
            )
 
    return weights
    