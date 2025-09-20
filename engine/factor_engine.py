import numpy as np
import pandas as pd
from typing import Callable, Iterable


def compute_factors(
    df: pd.DataFrame,
    factor_fn: Callable[[np.ndarray], float],
    lookback_minutes: int,
    eval_times: Iterable[pd.Timestamp],
) -> pd.Series:
    """
    Compute factor values at specified evaluation timestamps using the last `lookback_minutes` of OHLCV data.
    The window includes time t (i.e., up to the evaluation bar).
    """
    cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    ohlcv = df[cols]

    eval_times = pd.DatetimeIndex(eval_times)
    values = []
    index = []

    for t in eval_times:
        if t not in ohlcv.index:
            continue
        start_t = t - pd.Timedelta(minutes=lookback_minutes - 1)
        window = ohlcv.loc[start_t:t]
        if len(window) < lookback_minutes:
            values.append(np.nan)
            index.append(t)
            continue
        window_arr = window.to_numpy(dtype=float)
        try:
            val = factor_fn(window_arr)
        except Exception:
            val = np.nan
        values.append(float(val) if val is not None else np.nan)
        index.append(t)

    factor = pd.Series(values, index=pd.DatetimeIndex(index), name="factor")
    return factor 