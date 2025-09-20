import pandas as pd
from typing import Optional


def _ensure_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_minute_data(
    csv_path: str,
    tz: str = "UTC",
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
    forward_fill: bool = True,
) -> pd.DataFrame:
    """
    Load 1-minute OHLCV data, set UTC Datetime index, filter by [start, end], and forward-fill gaps.

    Expected columns in CSV: at least [open, high, low, close, volume] and either 'open_time' (ms) or 'Datetime'.
    """
    raw = pd.read_csv(csv_path)

    if "open_time" in raw.columns:
        ts = pd.to_datetime(raw["open_time"], unit="ms", utc=True)
    elif "Datetime" in raw.columns:
        # Assume the Datetime column is UTC or parseable; coerce to UTC
        ts = pd.to_datetime(raw["Datetime"], utc=True)
    else:
        raise ValueError("CSV must include 'open_time' (ms) or 'Datetime' column")

    raw.index = ts
    raw = raw.drop(columns=[c for c in ["Datetime"] if c in raw.columns])
    raw = raw.sort_index()
    raw = raw[~raw.index.duplicated(keep="first")]

    raw = _ensure_numeric(raw)

    # Filter start/end
    if start_ts is not None:
        start_ts = pd.Timestamp(start_ts).tz_convert("UTC") if start_ts.tzinfo else pd.Timestamp(start_ts, tz="UTC")
        raw = raw[raw.index >= start_ts]
    if end_ts is not None:
        end_ts = pd.Timestamp(end_ts).tz_convert("UTC") if end_ts.tzinfo else pd.Timestamp(end_ts, tz="UTC")
        raw = raw[raw.index <= end_ts]

    # Ensure minute frequency and forward-fill if requested
    if forward_fill and not raw.empty:
        full_index = pd.date_range(start=raw.index[0], end=raw.index[-1], freq="min", tz="UTC")
        df = raw.reindex(full_index)
        # Track which were newly inserted
        missing_mask = df["open"].isna()
        # Forward-fill OHLC
        for col in ["open", "high", "low", "close"]:
            if col in df.columns:
                df[col] = df[col].ffill()
        # Volume: set 0 for inserted, keep original otherwise (fillna 0)
        if "volume" in df.columns:
            df.loc[missing_mask, "volume"] = 0.0
            df["volume"] = df["volume"].fillna(0.0)
    else:
        df = raw

    # Keep only standard columns
    keep_cols = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep_cols]

    return df 