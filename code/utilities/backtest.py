import pandas as pd


def _align_portfolios_and_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align a portfolio schedule with the return matrix on the common investable universe.

    Parameters:
        portfolios (pd.DataFrame): Portfolio weights indexed by rebalance date.
        returns (pd.DataFrame): Asset returns indexed by trading date.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: The aligned portfolio and return matrices.
    """
    if portfolios.empty or portfolios.shape[1] == 0:
        raise ValueError("portfolios must contain at least one asset")

    overlapping_assets = portfolios.columns.intersection(returns.columns)
    if overlapping_assets.empty:
        raise ValueError("portfolios and returns must share at least one asset")

    aligned_portfolios = (
        portfolios.loc[:, overlapping_assets]
        .reindex(returns.index)
        .ffill()        # Error: it was bfill in the initial script
        .shift()
        .dropna(axis=0, how="all")
    )
    if aligned_portfolios.empty:
        raise ValueError(
            "portfolios and returns must overlap on at least one investable date"
        )

    aligned_returns = returns.loc[aligned_portfolios.index, overlapping_assets]

    return aligned_portfolios.fillna(0.0), aligned_returns


def portfolio_returns(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: float = 0.0,
) -> pd.Series:
    """
    Compute the time series of realized portfolio returns from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).
        transaction_costs (float): All-in transaction cost rate applied to one-way turnover
            (e.g. 0.001 = 10 bps). Defaults to 0.0.

    Returns:
        pd.Series: Series of realized portfolio returns over time.
    """
    # We align the ptfs and the returns in order to be able to do simpler computations
    aligned_portfolios, aligned_returns = _align_portfolios_and_returns(
        portfolios=portfolios,
        returns=returns,
    )

    gross_returns = aligned_portfolios.multiply(aligned_returns).sum(axis=1)

    if transaction_costs != 0.0:
           # Logica dei costi di transazione
        #
        # Al momento del rebalance t, il portafoglio viene riportato ai pesi
        # target w_t. Il giorno precedente al rebalance il portafoglio aveva
        # pesi w_{t-1} che si sono "driftati" con i rendimenti di mercato.
        #
        # Il costo one-way viene applicato alla somma delle variazioni assolute
        # dei pesi tra il portafoglio driftato PRIMA del ribilanciamento e il
        # portafoglio target DOPO:
        #
        #   TC_t = transaction_costs * Σ_i |w_i,t  −  w̃_i,t|  / 2   (one-way)
        #
        # dove w̃_i,t è il peso driftato:
        #
        #   w̃_i,t = w_{i,t-1} * (1 + r_{i,t}) / (1 + r_p,t)
        #
        # I costi vengono applicati solo nelle date in cui avviene effettivamente
        # un ribilanciamento (le righe non-NaN del DataFrame originale dei pesi,
        # dopo il bfill e lo shift fatto in _align_portfolios_and_returns).
        #
        # Nota: aligned_portfolios ha già subito .bfill().shift(), quindi le
        # date di rebalance sono identificabili come quelle in cui il peso
        # cambia rispetto alla riga precedente.
 
        # Pesi target allineati sull'intero calendario di trading
        w_target = aligned_portfolios  # shape (T, N), già bfill+shift
 
        # Rendimento del portafoglio al giorno t con i pesi del giorno precedente
        ptf_ret_t = gross_returns  # già calcolato sopra: Σ w_{t-1,i} * r_{t,i}
 
        # Pesi driftati: dopo aver incassato il rendimento del giorno t,
        # i pesi si spostano prima che avvenga il rebalance
        #   w̃_{i,t} = w_{i,t-1} * (1 + r_{i,t}) / (1 + r_p,t)
        w_prev        = w_target.shift(1).fillna(0.0)          # w_{t-1}
        gross_factor  = (1.0 + aligned_returns)                 # (1 + r_{i,t})
        ptf_factor    = (1.0 + ptf_ret_t).values.reshape(-1, 1)  # (1 + r_p,t)
 
        w_drifted = w_prev.multiply(gross_factor).divide(ptf_factor)
 
        # One-way turnover per ogni giorno di trading: Σ|Δw_i| / 2
        # È zero nei giorni in cui i pesi target non cambiano (no rebalance),
        # positivo nelle date di rebalance effettive.
        one_way_turnover = (w_target - w_drifted).abs().sum(axis=1) / 2.0
 
        # Costo di transazione giornaliero = tc * one_way_turnover
        tc_series = transaction_costs * one_way_turnover
 
        # Rendimento netto = rendimento lordo - costi
        gross_returns = gross_returns - tc_series
 
    

    return gross_returns


def backtest(
    portfolios: pd.DataFrame,
    returns: pd.DataFrame,
    transaction_costs: float = 0.0,
) -> pd.Series:
    """
    Compute cumulative portfolio performance from a rebalance schedule.

    Parameters:
        portfolios (pd.DataFrame): DataFrame where each column represents the weights of a
            portfolio over time (dates as index).
        returns (pd.DataFrame): DataFrame of asset returns (dates as index, assets as columns).
        transaction_costs (float): All-in transaction cost rate applied to one-way turnover
            (e.g. 0.001 = 10 bps). Defaults to 0.0.

    Returns:
        pd.Series: Series representing the cumulative returns of the portfolios over time.
    """
    return (
        portfolio_returns(
            portfolios=portfolios,
            returns=returns,
            transaction_costs=transaction_costs,
        )
        .add(1)
        .cumprod()
    )