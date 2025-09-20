import pandas as pd
from typing import Dict


def _align(factor: pd.Series, label: pd.Series) -> pd.DataFrame:
    df = pd.concat({"factor": factor, "label": label}, axis=1)
    return df.dropna()


def compute_ic_metrics(
    factor: pd.Series,
    label: pd.Series,
    rolling_window: int = 50,
) -> Dict[str, pd.Series]:
    """
    Compute Pearson and Spearman IC time series and their cumulative sums.
    """
    aligned = _align(factor, label)
    f = aligned["factor"]
    y = aligned["label"]

    pearson_ic = f.rolling(window=rolling_window, min_periods=rolling_window).corr(y)
    # Spearman via rank transformation then Pearson
    f_rank = f.rank(method="average")
    y_rank = y.rank(method="average")
    spearman_ic = f_rank.rolling(window=rolling_window, min_periods=rolling_window).corr(y_rank)

    ic_cumsum = pearson_ic.fillna(0.0).cumsum()

    return {
        "pearson_ic": pearson_ic.rename("pearson_ic"),
        "spearman_ic": spearman_ic.rename("spearman_ic"),
        "ic_cumsum": ic_cumsum.rename("ic_cumsum"),
    } 