import pandas as pd
from typing import Iterable


def build_labels(
    df: pd.DataFrame,
    eval_times: Iterable[pd.Timestamp],
    k_minutes: int,
) -> pd.Series:
    """
    For factor at time t, label is simple return from open[t+1] to open[t+1+k].
    """
    opens = df["open"]
    eval_times = pd.DatetimeIndex(eval_times)

    label_vals = []
    idx = []

    for t in eval_times:
        if t not in opens.index:
            continue
        t1 = t + pd.Timedelta(minutes=1)
        tk = t + pd.Timedelta(minutes=1 + k_minutes)
        if t1 not in opens.index or tk not in opens.index:
            label_vals.append(float("nan"))
            idx.append(t)
            continue
        r = (float(opens.loc[tk]) / float(opens.loc[t1])) - 1.0
        label_vals.append(r)
        idx.append(t)

    return pd.Series(label_vals, index=pd.DatetimeIndex(idx), name="label") 