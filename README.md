# Risk-Based Portfolio Allocation with ERC and HRP in Python

This repository contains a Python implementation and comparison of risk-based
portfolio allocation strategies applied to Euro Stoxx 50 equities.

The project focuses on two main allocation methodologies:

- Equal Risk Contribution;
- Hierarchical Risk Parity.

The analysis studies how these strategies behave under different covariance
estimators and correlation-processing techniques.

The project includes:

- sample covariance estimation;
- constant-correlation shrinkage;
- single-factor covariance shrinkage;
- Equal Risk Contribution optimization;
- inverse-volatility allocation;
- correlation detoning;
- hierarchical clustering;
- Hierarchical Risk Parity;
- alternative clustering linkage methods;
- alternative hierarchy traversal rules;
- cluster stability analysis;
- portfolio concentration diagnostics;
- rolling out-of-sample backtests;
- comparison between ERC and HRP.

## Project Overview

The case study is framed from the perspective of a quantitative analyst working
for a pension fund.

The objective is to study allocation strategies that rely primarily on the
covariance and correlation structure of asset returns rather than on expected
return estimates.

The investment universe consists of Euro Stoxx 50 constituents.

The analysis uses:

- total return indices denominated in EUR;
- simple daily returns;
- monthly rebalancing at the end of each month;
- a rolling two-year estimation window;
- no risk-free rate.

All analyses are repeated using three covariance estimators:

1. sample covariance matrix;
2. constant-correlation shrinkage estimator;
3. single-factor market shrinkage estimator.

The equally weighted return of the investment universe is used as the market
proxy for the single-factor estimator.

## Main Objectives

The project addresses three connected questions.

### Equal Risk Contribution

The first objective is to construct portfolios in which all assets contribute
approximately the same amount to total portfolio risk.

### Hierarchical Risk Parity

The second objective is to construct diversified portfolios using hierarchical
clustering and recursive risk allocation.

### Robustness and Comparison

The final objective is to compare ERC and HRP in terms of:

- sensitivity to covariance estimation;
- cumulative performance;
- realized volatility;
- drawdowns;
- turnover;
- concentration;
- cluster stability.

## Data

The project uses the course-provided file:

```text
sx5e_underlyings.csv
```

The dataset contains total return indices for the Euro Stoxx 50 investment
universe.

Course-provided or proprietary data should not be included in a public
repository unless redistribution is explicitly permitted.

## Return Calculation

Simple daily returns are computed from total return indices.

```text
return_t =
    total_return_index_t
    / total_return_index_t_minus_1
    - 1
```

The risk-free rate is ignored.

The data preparation procedure includes:

- chronological sorting;
- duplicate-date checks;
- alignment of securities on common dates;
- missing-value treatment;
- removal of securities with insufficient history;
- prevention of look-ahead bias.

## Rolling Estimation Framework

The portfolios are rebalanced monthly at the end of each month.

At every rebalance date:

1. select the previous two years of daily returns;
2. estimate the required covariance and correlation matrices;
3. calculate target portfolio weights;
4. hold the portfolio until the following rebalance;
5. record out-of-sample returns;
6. advance the estimation window.

A representative approximation is:

```text
estimation_window = 504 trading days
rebalancing_frequency = monthly
```

The exact number of observations may vary because of holidays and missing data.

## Covariance Estimators

Every portfolio strategy is evaluated using three covariance estimators.

### Sample Covariance Matrix

The sample covariance matrix is estimated directly from the return observations
inside the current rolling window.

```text
sample_covariance =
    covariance_matrix(returns_window)
```

The sample estimator is flexible but may be noisy and unstable, particularly
when the number of assets is large relative to the number of observations.

### Constant-Correlation Shrinkage

The constant-correlation estimator combines the sample covariance matrix with a
structured target.

The target preserves each asset's estimated volatility while replacing all
off-diagonal correlations with their average value.

```text
target_covariance_ij =
    average_correlation
    * volatility_i
    * volatility_j
```

For diagonal elements:

```text
target_covariance_ii =
    variance_i
```

The shrunk covariance matrix is:

```text
shrunk_covariance =
    shrinkage_intensity
    * target_covariance
    + (
        1 - shrinkage_intensity
    )
    * sample_covariance
```

### Single-Factor Shrinkage

The single-factor estimator assumes that returns are driven by one common market
factor and asset-specific residuals.

```text
asset_return_i =
    alpha_i
    + beta_i * market_return
    + residual_i
```

The equally weighted return of the current investment universe is used as the
market proxy.

For two different assets:

```text
target_covariance_ij =
    beta_i
    * beta_j
    * market_variance
```

For each asset variance:

```text
target_variance_i =
    beta_i^2
    * market_variance
    + residual_variance_i
```

The target is then combined with the sample covariance matrix using the estimated
shrinkage intensity.

## Why Covariance Estimation Matters

Risk-based allocation strategies do not require expected returns, but they still
depend on estimated variances and correlations.

Estimation error may affect:

- marginal risk contributions;
- cluster formation;
- portfolio weights;
- turnover;
- concentration;
- realized portfolio risk.

Shrinkage estimators reduce sampling noise by moving the sample covariance matrix
toward a more stable structured target.

However, ERC and HRP may respond differently to covariance estimation errors.

ERC directly solves a nonlinear risk-budgeting problem.

HRP uses the covariance structure indirectly through clustering and recursive
allocation.

## Part I — Equal Risk Contribution

Equal Risk Contribution allocates the portfolio so that each asset contributes
the same amount to total portfolio volatility.

## Portfolio Variance

For a weight vector `w` and covariance matrix `Sigma`, portfolio variance is:

```text
portfolio_variance =
    transpose(w)
    @ Sigma
    @ w
```

Portfolio volatility is:

```text
portfolio_volatility =
    sqrt(portfolio_variance)
```

## Marginal Risk Contribution

The marginal contribution of asset `i` to portfolio volatility is:

```text
marginal_risk_contribution_i =
    (
        Sigma
        @ w
    )[i]
    / portfolio_volatility
```

This measures the local change in portfolio volatility associated with a small
change in the weight of asset `i`.

## Total Risk Contribution

The total risk contribution of asset `i` is:

```text
risk_contribution_i =
    weight_i
    * marginal_risk_contribution_i
```

The individual risk contributions must sum to total portfolio volatility.

```text
sum(risk_contributions)
    approximately equals
portfolio_volatility
```

## Relative Risk Contribution

The relative contribution of asset `i` is:

```text
relative_risk_contribution_i =
    risk_contribution_i
    / portfolio_volatility
```

For an equally risk-contributing portfolio with `N` assets:

```text
target_relative_risk_contribution =
    1 / N
```

## ERC Optimization

The ERC portfolio is obtained by finding weights whose risk contributions are as
close as possible to the equal-risk target.

A representative objective is:

```text
minimize:
    sum(
        (
            relative_risk_contribution_i
            - 1 / number_of_assets
        )^2
    )
```

Subject to:

```text
sum(weights) = 1

weights_i >= 0
```

The exact solver and tolerance should be documented in the implementation.

## Equivalent Pairwise Formulation

An alternative ERC objective minimizes differences between pairs of risk
contributions.

```text
minimize:
    sum over i and j of
    (
        risk_contribution_i
        - risk_contribution_j
    )^2
```

Both formulations aim to produce approximately equal asset-level risk
contributions.

## ERC Validation

The optimized portfolio should satisfy:

```text
sum(weights) approximately equals 1
```

and:

```text
all weights are non-negative
```

The relative risk contributions should satisfy:

```text
relative_risk_contribution_i
    approximately equals
1 / number_of_assets
```

for every asset.

## Risk-Contribution Dispersion

The quality of the ERC solution is measured through the dispersion of relative
risk contributions.

Possible diagnostics include:

```text
standard_deviation(
    relative_risk_contributions
)
```

```text
maximum(
    relative_risk_contributions
)
-
minimum(
    relative_risk_contributions
)
```

```text
root_mean_squared_error =
    sqrt(
        mean(
            (
                relative_risk_contribution_i
                - 1 / number_of_assets
            )^2
        )
    )
```

A successful ERC solution should produce values close to zero.

## Inverse-Volatility Portfolio

The inverse-volatility portfolio allocates more weight to lower-volatility
assets.

```text
raw_weight_i =
    1 / volatility_i
```

The normalized weight is:

```text
inverse_volatility_weight_i =
    raw_weight_i
    / sum(raw_weights)
```

## ERC with a Diagonal Covariance Matrix

When the covariance matrix is diagonal, assets have zero pairwise covariance.

In this case:

```text
portfolio_variance =
    sum(
        weight_i^2
        * variance_i
    )
```

Equalizing risk contributions implies:

```text
weight_i
    proportional to
1 / volatility_i
```

Therefore, when the covariance matrix is diagonal:

```text
ERC weights
    equal
inverse-volatility weights
```

The implementation verifies this property numerically.

## Numerical Diagonal Test

A diagonal covariance matrix can be constructed from estimated asset variances.

```text
diagonal_covariance =
    diagonal_matrix(
        asset_variances
    )
```

The following comparison is then performed:

```text
maximum_absolute_difference =
    max(
        abs(
            erc_weights
            - inverse_volatility_weights
        )
    )
```

The difference should be close to zero within numerical tolerance.

## ERC Rolling Backtest

The ERC portfolio is backtested under:

- sample covariance;
- constant-correlation shrinkage;
- single-factor shrinkage.

At every monthly rebalance, three separate ERC weight vectors are calculated.

The weights are then applied to subsequent returns until the next rebalance.

## ERC Performance Measures

The assignment requires the following metrics:

- cumulative return;
- 63-day exponentially weighted realized volatility;
- drawdown;
- turnover.

Additional diagnostics may include:

- maximum drawdown;
- annualized return;
- average concentration;
- weight dispersion;
- average risk-contribution dispersion.

## Cumulative Performance

```text
cumulative_wealth_t =
    product(
        1 + portfolio_return_s
        for all s up to t
    )
```

The cumulative return is:

```text
cumulative_return_t =
    cumulative_wealth_t
    - 1
```

## Exponentially Weighted Realized Volatility

The project reports a 63-day exponentially weighted realized volatility.

A general recursive estimate is:

```text
ewm_variance_t =
    decay_factor
    * ewm_variance_t_minus_1
    + (
        1 - decay_factor
    )
    * portfolio_return_t^2
```

The annualized volatility is:

```text
annualized_ewm_volatility_t =
    sqrt(
        252
        * ewm_variance_t
    )
```

The exact pandas span, half-life, or smoothing parameter should be documented in
the code.

The rolling history used for the reported measure is 63 trading days.

## Drawdown

The running peak is:

```text
running_peak_t =
    maximum cumulative wealth
    observed up to time t
```

Drawdown is:

```text
drawdown_t =
    cumulative_wealth_t
    / running_peak_t
    - 1
```

Maximum drawdown is:

```text
maximum_drawdown =
    minimum(drawdown_series)
```

## Turnover

A simple turnover measure is:

```text
turnover_t =
    sum(
        abs(
            target_weight_i_t
            - previous_weight_i
        )
    )
```

A more precise measure compares the new target weights with the drifted
pre-rebalance weights.

```text
drifted_weight_i =
    previous_weight_i
    * (
        1 + asset_return_i
    )
    / (
        1 + portfolio_return
    )
```

Then:

```text
turnover_t =
    sum(
        abs(
            new_target_weight_i
            - drifted_weight_i
        )
    )
```

The selected turnover convention should be stated explicitly.

## Expected Impact of Covariance Estimation on ERC

Because ERC directly uses the covariance matrix, different estimators may affect:

- asset risk contributions;
- portfolio weights;
- weight stability;
- turnover;
- realized volatility.

Shrinkage may reduce unstable month-to-month changes in the covariance estimate.

However, risk-based portfolios are often less sensitive to estimation error than
unconstrained mean-variance portfolios because they do not invert expected-return
signals.

## Part II — Hierarchical Risk Parity

Hierarchical Risk Parity constructs portfolio weights through three main stages:

1. hierarchical clustering;
2. quasi-diagonalization;
3. recursive risk allocation.

Unlike classical optimization, HRP does not require direct inversion of the
covariance matrix.

## Correlation Matrix

The correlation matrix is obtained from the covariance matrix.

```text
correlation_ij =
    covariance_ij
    / (
        volatility_i
        * volatility_j
    )
```

The diagonal entries equal one.

```text
correlation_ii = 1
```

## Correlation Distance

A common correlation-based distance between assets is:

```text
distance_ij =
    sqrt(
        (
            1 - correlation_ij
        )
        / 2
    )
```

Properties include:

```text
correlation = 1
    gives distance = 0

correlation = -1
    gives distance = 1
```

This distance can be used as the input to hierarchical clustering.

When Ward linkage is used, the implementation must ensure that the chosen input
is compatible with the Euclidean structure required by the method.

## Correlation Detoning

Equity correlation matrices are often dominated by a broad market component.

This common component may cause nearly all stocks to appear strongly related,
making sector and subgroup structure more difficult to detect.

Detoning removes the contribution of the first principal component from the
correlation matrix before clustering.

## PCA Decomposition of Correlation

The correlation matrix can be decomposed as:

```text
correlation_matrix =
    sum over j of
    eigenvalue_j
    * outer_product(
        eigenvector_j,
        eigenvector_j
    )
```

The first component is:

```text
market_component =
    largest_eigenvalue
    * outer_product(
        first_eigenvector,
        first_eigenvector
    )
```

## Detoned Correlation Matrix

The raw detoned matrix is:

```text
raw_detoned_matrix =
    correlation_matrix
    - market_component
```

Removing the first component may alter the diagonal.

The matrix is therefore rescaled back into a valid correlation matrix.

```text
detoned_correlation_ij =
    raw_detoned_matrix_ij
    / sqrt(
        raw_detoned_matrix_ii
        * raw_detoned_matrix_jj
    )
```

The diagonal is then set to one.

## Why Detoning May Improve Clustering

Removing the first principal component may help hierarchical clustering by:

- reducing the dominant market-wide correlation;
- making sector relationships more visible;
- increasing cross-sectional differentiation;
- avoiding one broad undifferentiated cluster;
- improving the economic interpretation of subclusters.

Potential limitations include:

- removing economically relevant systematic information;
- introducing numerical instability;
- producing negative eigenvalues after adjustment;
- increasing sensitivity to smaller and noisier components.

The resulting matrix should therefore be checked carefully.

## Detoning Validation

The implementation should verify:

```text
detoned_correlation_matrix
    approximately equals
transpose(detoned_correlation_matrix)
```

```text
diagonal entries approximately equal 1
```

```text
all entries lie approximately between -1 and 1
```

The eigenvalues should also be inspected.

Small negative eigenvalues caused by floating-point error may be corrected if
necessary, but any correction should be documented.

## Dendrogram Comparison

For the last rebalance date, the project reports dendrograms based on:

- the original correlation matrix;
- the detoned correlation matrix.

The comparison illustrates how removing the market factor changes the hierarchy
of stocks.

Useful observations include:

- number and composition of visible clusters;
- merge distances;
- separation of sector groups;
- concentration of late-stage merges;
- changes in the order of dendrogram leaves.

## Quasi-Diagonalization

After hierarchical clustering, assets are reordered according to the leaves of
the dendrogram.

Applying this order to the correlation matrix produces a quasi-diagonalized
matrix.

Assets belonging to the same cluster should appear in neighboring rows and
columns.

The resulting heatmap should display block-like structures when clustering is
informative.

The project compares quasi-diagonalized matrices before and after detoning at the
last rebalance date.

## Hierarchical Clustering

The HRP implementation is parameterized by the linkage method and distance
metric.

The main linkage methods are:

- single linkage;
- Ward linkage;
- complete linkage;
- average linkage.

## Single Linkage

Single linkage defines the distance between two clusters as the smallest distance
between any pair of observations.

Advantages:

- can identify elongated structures;
- simple interpretation.

Limitations:

- susceptible to chaining;
- may create unstable and unbalanced hierarchies.

## Complete Linkage

Complete linkage uses the largest pairwise distance between the two clusters.

Advantages:

- tends to produce compact clusters;
- reduces chaining.

Limitations:

- sensitive to outliers;
- may create strongly separated clusters.

## Average Linkage

Average linkage uses the average pairwise distance between observations in the
two clusters.

Advantages:

- balances single and complete linkage;
- often produces relatively stable hierarchies.

Limitations:

- may blur sharply separated cluster structures.

## Ward Linkage

Ward linkage merges clusters that produce the smallest increase in within-cluster
dispersion.

Advantages:

- often produces compact and balanced clusters;
- useful for identifying cohesive groups.

Limitations:

- requires a Euclidean-compatible representation;
- can behave poorly when applied incorrectly to arbitrary precomputed distances.

## Cluster Variance

HRP requires a measure of the variance of each cluster.

Within a cluster, inverse-variance weights are commonly used.

```text
inverse_variance_weight_i =
    (
        1 / variance_i
    )
    / sum(
        1 / variance_j
        for j in cluster
    )
```

Cluster variance is then:

```text
cluster_variance =
    transpose(cluster_weights)
    @ cluster_covariance
    @ cluster_weights
```

## Risk Allocation Between Clusters

Suppose a parent cluster is divided into left and right subclusters.

Let:

```text
variance_left =
    estimated variance
    of left cluster

variance_right =
    estimated variance
    of right cluster
```

The allocation fractions are:

```text
allocation_left =
    variance_right
    / (
        variance_left
        + variance_right
    )
```

```text
allocation_right =
    variance_left
    / (
        variance_left
        + variance_right
    )
```

The lower-variance cluster therefore receives the larger capital allocation.

## Recursive Bisection

The traditional HRP recursive-bisection allocator follows the quasi-diagonalized
leaf order.

At every stage:

1. divide the ordered list of assets approximately in half;
2. estimate the variance of both subgroups;
3. allocate capital inversely to subgroup variance;
4. repeat until individual assets are reached.

This method follows the leaf order but does not necessarily split at actual
dendrogram branches.

## Dendrogram Iteration

The alternative allocator follows the actual clustering hierarchy.

At every stage:

1. identify the active cluster with the largest merge distance;
2. split it according to its two actual child clusters;
3. allocate capital inversely to child-cluster variance;
4. continue until all terminal assets are reached.

This method respects the exact dendrogram structure.

## Comparing the Two Allocation Traversals

Recursive bisection and dendrogram iteration may produce similar weights when:

- the dendrogram is balanced;
- the quasi-diagonal order is consistent with natural cluster splits;
- adjacent leaf halves correspond to actual child clusters;
- cluster variances are similar.

They may differ when:

- the hierarchy is strongly unbalanced;
- the dendrogram contains clusters of very different sizes;
- halving the leaf order cuts across natural branches;
- merge distances vary substantially.

## Generic HRP Configuration

The HRP engine should accept parameters such as:

```text
covariance_estimator

detoning_enabled

linkage_method

distance_metric

allocation_traversal
```

This modular structure allows the full sensitivity analysis to be performed
without duplicating the portfolio logic.

## Cluster Stability

The project measures the stability of clusters across consecutive monthly
rebalances.

The baseline clustering configuration is:

```text
linkage_method = ward

distance_metric = euclidean
```

The analysis is repeated across:

```text
covariance estimators:
    sample
    constant-correlation shrinkage
    single-factor shrinkage

detoning:
    disabled
    enabled
```

## Number of Clusters

Cluster stability is evaluated for:

```text
k = 5

k = 8

k = 10

k = 15
```

At each date, the dendrogram is cut to obtain exactly `k` clusters.

## Rand Index

The Rand index compares two cluster assignments by considering whether every
pair of assets is grouped consistently.

```text
rand_index =
    number_of_consistent_asset_pairs
    / total_number_of_asset_pairs
```

The implementation uses:

```text
sklearn.metrics.rand_score
```

The score lies between zero and one.

```text
rand_index close to 1
```

indicates high stability between consecutive rebalances.

```text
rand_index close to 0
```

indicates substantial changes in cluster assignments.

## Universe Consistency in Cluster Comparisons

If the eligible stock universe changes through time, the Rand index should be
computed only on the assets available at both consecutive rebalances.

A consistent ticker ordering must be used before comparing cluster labels.

Cluster label numbers themselves are arbitrary, but the Rand index is invariant
to label permutations.

## Smoothed Cluster Stability

The project reports a 12-month exponentially weighted moving average of the Rand
index.

```text
smoothed_rand_index_t =
    ewm_mean(
        consecutive_rand_indices,
        span corresponding to 12 monthly observations
    )
```

The exact smoothing convention should be stated in the implementation.

## Questions Addressed by the Stability Analysis

The cluster-stability analysis evaluates:

- whether detoning improves cluster persistence;
- whether covariance shrinkage stabilizes the clusters;
- whether the effects depend on the number of clusters;
- whether market conditions influence cluster turnover.

## HRP Weight Sensitivity

The assignment requires a broad sensitivity analysis over:

```text
linkage methods
    x
allocation traversals
    x
covariance estimators
    x
detoning choices
```

The linkage methods are:

```text
single
ward
complete
average
```

The allocation traversals are:

```text
recursive bisection
dendrogram iteration
```

The covariance estimators are:

```text
sample
constant correlation
single factor
```

Detoning is:

```text
on
off
```

## Sensitivity Metrics

Each HRP variant is summarized using:

- mean portfolio turnover;
- mean Herfindahl concentration index.

## Herfindahl Concentration Index

The Herfindahl concentration index is:

```text
herfindahl_index =
    sum(
        weight_i^2
    )
```

For a long-only fully invested portfolio:

```text
1 / number_of_assets
    <= herfindahl_index
    <= 1
```

A lower value indicates a more diversified allocation.

A higher value indicates greater weight concentration.

The effective number of holdings can also be reported as:

```text
effective_number_of_holdings =
    1 / herfindahl_index
```

## Linkage Effects on HRP Weights

The linkage method influences:

- cluster membership;
- dendrogram balance;
- order of risk allocation;
- depth of individual assets in the hierarchy;
- final portfolio concentration.

Single linkage may generate long chains and unbalanced clusters.

Ward linkage often produces more balanced groups.

Complete and average linkage may generate different intermediate structures,
which propagate into cluster-level risk allocations.

## Allocation-Traversal Effects

The allocator determines how the hierarchy is translated into weights.

Recursive bisection depends on the dendrogram leaf order.

Dendrogram iteration follows actual parent-child relationships.

The difference can be evaluated using:

- weight correlations;
- L1 distance;
- concentration;
- turnover;
- cumulative performance.

## Part III — Connecting the Dots

The final section connects covariance estimation, ERC, HRP, and portfolio
performance.

## Weight Robustness Across Estimators

For the same strategy, the average L1 distance between two weight vectors is:

```text
l1_distance =
    sum(
        abs(
            weights_estimator_a
            - weights_estimator_b
        )
    )
```

The rolling average is:

```text
average_l1_distance =
    mean(
        l1_distance_at_each_rebalance
    )
```

The comparison is performed between:

- sample and constant-correlation weights;
- sample and single-factor weights;
- constant-correlation and single-factor weights.

## ERC and HRP Robustness

The L1 analysis evaluates whether ERC and HRP are equally sensitive to the
covariance estimator.

ERC weights depend directly on the full covariance matrix and nonlinear
risk-contribution equations.

HRP weights depend on:

- correlation-based distances;
- discrete cluster assignments;
- cluster variances;
- hierarchy traversal.

HRP may be robust when different covariance estimators generate the same
dendrogram.

However, small correlation changes may also cause discontinuous cluster changes
and produce material weight differences.

The empirical comparison determines which effect dominates.

## ERC and HRP Performance Comparison

One representative ERC strategy and one representative HRP strategy are selected
based on the preceding analysis.

They are compared using:

- cumulative performance;
- 63-day exponentially weighted realized volatility;
- drawdowns;
- turnover.

Additional metrics may include:

- annualized return;
- maximum drawdown;
- average concentration;
- effective number of holdings;
- stability of weights.

## Why Shrinkage May Matter Less Than in Mean-Variance Optimization

In classical mean-variance optimization, the covariance matrix is often inverted.

Small estimation errors can therefore be amplified into large and unstable
portfolio positions.

ERC and HRP are generally less dependent on precise matrix inversion.

ERC uses the covariance matrix to equalize risk contributions.

HRP uses correlations to construct a hierarchy and cluster variances to allocate
capital.

As a result, covariance shrinkage may have a smaller effect on risk-based
portfolio weights than on unconstrained optimization-based strategies.

This does not imply that shrinkage is irrelevant.

It may still improve:

- numerical stability;
- risk-contribution estimates;
- cluster persistence;
- turnover;
- out-of-sample volatility.

The effect must be evaluated empirically.

## Backtest Timing

All portfolio weights must be estimated using only information available at the
rebalance date.

The new weights are applied only to subsequent returns.

A representative process is:

```text
at month-end t:
    estimate covariance
    calculate weights

during the following holding period:
    apply weights to returns
```

This prevents look-ahead bias.

## Weight Drift

Between monthly rebalances, portfolio weights drift because assets earn different
returns.

For accurate turnover calculations, pre-trade weights should be updated before
comparing them with the next target allocation.

```text
drifted_weight_i =
    previous_weight_i
    * (
        1 + cumulative_asset_return_i
    )
    / (
        1 + cumulative_portfolio_return
    )
```

The new target weights are compared with these drifted weights.

## Numerical Validation

The implementation should include the following checks.

### Covariance-Matrix Symmetry

```text
covariance_matrix
    approximately equals
transpose(covariance_matrix)
```

### Covariance Positive Semidefiniteness

Eigenvalues should be non-negative within numerical tolerance.

### Valid Correlation Matrix

```text
correlation_ii approximately equals 1
```

```text
-1 <= correlation_ij <= 1
```

### Portfolio Budget

```text
sum(weights) approximately equals 1
```

### Long-Only Constraint

```text
weights_i >= 0
```

### ERC Risk Contributions

```text
risk_contribution_i
    approximately equals
portfolio_volatility
    / number_of_assets
```

### ERC Diagonal Equivalence

```text
ERC weights
    approximately equal
inverse-volatility weights
```

when the covariance matrix is diagonal.

### HRP Weight Positivity

```text
all HRP weights are non-negative
```

### HRP Allocation Conservation

At each hierarchy split:

```text
allocation_left
    + allocation_right
    = 1
```

### Rand Index Bounds

```text
0 <= rand_index <= 1
```

### Herfindahl Bounds

For a long-only fully invested portfolio:

```text
1 / number_of_assets
    <= herfindahl_index
    <= 1
```

### Out-of-Sample Timing

Portfolio returns must use weights available before the return is observed.

### Reproducibility

All deterministic calculations should produce identical results across runs.

## Common Implementation Risks

Potential implementation errors include:

- applying new weights to returns from the same estimation date;
- failing to account for drifted weights in turnover;
- using covariance matrices with misaligned ticker order;
- computing risk contributions from portfolio variance instead of volatility,
  or vice versa;
- using an unconstrained optimizer that produces negative ERC weights;
- failing to normalize ERC or HRP weights;
- applying Ward linkage to an incompatible precomputed distance matrix;
- forgetting to rescale the detoned matrix into a valid correlation matrix;
- comparing cluster labels without aligning the stock universe;
- treating arbitrary cluster labels as economically ordered;
- confusing the Herfindahl index with its inverse;
- implementing recursive bisection as if it followed actual dendrogram branches;
- introducing look-ahead bias in the rolling backtest.

## Suggested Repository Structure

```text
risk-based-allocation-erc-hrp/
|
|-- README.md
|-- requirements.txt
|
|-- src/
|   |-- data_loader.py
|   |-- returns.py
|   |-- rolling_windows.py
|   |-- covariance_estimators.py
|   |-- constant_correlation_shrinkage.py
|   |-- single_factor_shrinkage.py
|   |-- risk_contributions.py
|   |-- erc.py
|   |-- inverse_volatility.py
|   |-- correlation_detoning.py
|   |-- distance_matrices.py
|   |-- hierarchical_clustering.py
|   |-- quasi_diagonalization.py
|   |-- cluster_variance.py
|   |-- recursive_bisection.py
|   |-- dendrogram_allocator.py
|   |-- hrp.py
|   |-- cluster_stability.py
|   |-- backtest.py
|   |-- turnover.py
|   |-- performance_metrics.py
|   |-- sensitivity_analysis.py
|   `-- validation.py
|
|-- notebooks/
|   `-- risk_based_allocation_analysis.ipynb
|
|-- scripts/
|   `-- run_analysis.py
|
|-- data/
|   `-- README.md
|
|-- results/
|   |-- erc_weights.csv
|   |-- erc_risk_contributions.csv
|   |-- hrp_weights.csv
|   |-- cluster_labels.csv
|   |-- rand_indices.csv
|   |-- weight_sensitivity.csv
|   |-- performance_summary.csv
|   `-- figures/
|       |-- erc_cumulative_performance.png
|       |-- erc_realized_volatility.png
|       |-- erc_drawdowns.png
|       |-- erc_turnover.png
|       |-- dendrogram_original.png
|       |-- dendrogram_detoned.png
|       |-- quasi_diagonal_correlation_original.png
|       |-- quasi_diagonal_correlation_detoned.png
|       |-- cluster_stability.png
|       |-- hrp_turnover_sensitivity.png
|       |-- hrp_concentration_sensitivity.png
|       |-- estimator_weight_distances.png
|       |-- erc_vs_hrp_performance.png
|       |-- erc_vs_hrp_volatility.png
|       |-- erc_vs_hrp_drawdown.png
|       `-- erc_vs_hrp_turnover.png
|
`-- report/
    `-- assignment_report.pdf
```

The file and folder names can be adapted to the structure of the actual Python
implementation.

## Requirements

A representative Python environment may include:

```text
numpy
pandas
scipy
scikit-learn
matplotlib
```

Optional visualization packages may include:

```text
seaborn
```

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Project

A possible execution command is:

```bash
python scripts/run_analysis.py
```

Alternatively, the complete workflow can be executed from:

```text
notebooks/risk_based_allocation_analysis.ipynb
```

## Main Outputs

The project reports:

- rolling sample covariance matrices;
- rolling constant-correlation covariance estimates;
- rolling single-factor covariance estimates;
- ERC portfolio weights;
- ERC risk contributions;
- ERC risk-contribution dispersion;
- inverse-volatility equivalence test;
- ERC cumulative performance;
- ERC realized volatility;
- ERC drawdowns;
- ERC turnover;
- original and detoned correlation matrices;
- original and detoned dendrograms;
- quasi-diagonalized correlation matrices;
- HRP weights;
- cluster assignments;
- consecutive-rebalance Rand indices;
- 12-month smoothed cluster stability;
- HRP sensitivity across linkage methods;
- HRP sensitivity across allocation traversals;
- HRP sensitivity across covariance estimators;
- HRP sensitivity with and without detoning;
- mean turnover by HRP configuration;
- mean Herfindahl concentration index;
- average L1 distance between estimator-specific weights;
- comparative ERC and HRP performance.

## Technologies

- Python
- NumPy
- pandas
- SciPy
- scikit-learn
- Matplotlib
- Equal Risk Contribution
- Hierarchical Risk Parity
- Hierarchical clustering
- Principal Component Analysis
- Covariance shrinkage
- Quantitative asset allocation
- Portfolio backtesting

## Data

The project uses course-provided Euro Stoxx 50 total return data.

Expected input file:

```text
sx5e_underlyings.csv
```

When the original dataset cannot be published, the `data` directory should
contain a description of:

- expected file names;
- required columns;
- date format;
- ticker identifiers;
- total return index units;
- missing-data conventions;
- investment-universe filters.

## Academic Context

This project was developed as part of the Buy Side section of the Financial
Engineering course at Politecnico di Milano.

The repository presents the Python implementation and comparison of Equal Risk
Contribution and Hierarchical Risk Parity portfolios, including covariance
shrinkage, correlation detoning, hierarchical clustering, stability analysis,
and rolling out-of-sample evaluation.
