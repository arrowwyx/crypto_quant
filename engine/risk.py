import pandas as pd
import numpy as np
from typing import Literal


def compute_risk_scaler(
    opens: pd.Series,
    vol_lookback_days: int = 63,
    vol_target_ann: float = 0.25,
    rebalance: Literal["monthly", "days"] = "monthly",
    rebalance_days: int = 30,
) -> pd.Series:
    """
    Build a time series scaler based on recent annualized volatility.
    - Compute daily close-to-close simple returns from minute opens via daily last.
    - Rolling stdev over `vol_lookback_days` → annualize with sqrt(365).
    - scaler_base = vol_target_ann / sigma_ann (no cap). NaN -> 1.0.
    - Recompute on risk rebalance dates only, forward-fill between dates to minute index.
    """
    if opens.empty:
        return pd.Series(dtype=float)

    daily_price = opens.resample("1D").last().dropna()
    daily_ret = daily_price.pct_change().dropna()

    rolling_sigma = daily_ret.rolling(window=vol_lookback_days, min_periods=vol_lookback_days).std(ddof=0)
    sigma_ann = rolling_sigma * np.sqrt(365.0)

    raw_scaler = (vol_target_ann / sigma_ann).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    # Select rebalance dates
    if rebalance == "monthly":
        dates = raw_scaler.resample("MS").last().index
    else:
        start = raw_scaler.index.min()
        dates = pd.date_range(start=start, end=raw_scaler.index.max(), freq=f"{int(rebalance_days)}D")

    rebal_points = raw_scaler.reindex(dates).dropna()
    if rebal_points.empty:
        scaler_daily = pd.Series(1.0, index=daily_price.index)
    else:
        scaler_daily = rebal_points.reindex(daily_price.index, method="ffill").fillna(1.0)

    scaler_minute = scaler_daily.reindex(opens.index, method="ffill").fillna(1.0)
    scaler_minute.name = "risk_scaler"
    return scaler_minute
