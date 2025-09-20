import pandas as pd
import numpy as np


def _rolling_zscore(x: pd.Series, window: int) -> pd.Series:
    mean = x.rolling(window=window, min_periods=window).mean()
    std = x.rolling(window=window, min_periods=window).std(ddof=0)
    z = (x - mean) / std
    return z


def map_factor_to_target_notional(
    factor: pd.Series,
    capital: float,
    mapper: str = "zscore",
    zscore_window: int = 100,
    clip_abs: float = 1.0,
    allow_short: bool = True,
    long_leverage: float = 1.0,
    short_leverage: float = 1.0,
) -> pd.Series:
    """
    Map factor values to target dollar notional exposures.
    - mapper 'zscore': rolling z-score, clipped to [-clip_abs, clip_abs]
    - mapper 'sign': sign(factor)
    Apply leverage: s>=0 scaled by long_leverage, s<0 scaled by short_leverage.
    If short is not allowed, negatives are set to 0.
    """
    f = factor.copy().astype(float)

    if mapper == "zscore":
        s = _rolling_zscore(f, zscore_window)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    elif mapper == "sign":
        s = np.sign(f)
        s = s.clip(lower=-clip_abs, upper=clip_abs)
    else:
        raise ValueError(f"Unknown mapper: {mapper}")

    if not allow_short:
        s = s.clip(lower=0.0)

    # Apply leverage asymmetrically
    s_levered = s.where(s < 0, s * float(long_leverage))
    s_levered = s_levered.where(s < 0, s_levered)  # keep same for non-negative
    s_levered = s_levered.where(s >= 0, s * float(short_leverage))

    target_notional = capital * s_levered.fillna(0.0)
    target_notional.name = "target_notional"
    return target_notional 