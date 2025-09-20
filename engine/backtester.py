import pandas as pd
import numpy as np
from typing import Iterable, Dict, Any


def run_backtest(
    df: pd.DataFrame,
    eval_times: Iterable[pd.Timestamp],
    trade_delay_minutes: int,
    target_notional: pd.Series,
    initial_capital: float,
    cost_bps: float = 0.0,
) -> Dict[str, Any]:
    """
    Single-asset backtest using dollar target notionals.
    - At evaluation time t, the desired target notional is target_notional[t].
    - Execution occurs at trade_time = t + trade_delay_minutes, at open[trade_time].
    - Trading cost is applied as bps of traded notional (cost_bps).

    Returns dict with equity curve, pnl, trades, and positions.
    """
    opens = df["open"].astype(float)
    index = opens.index

    eval_times = pd.DatetimeIndex(eval_times)
    eval_times = eval_times.intersection(index)
    target_notional = target_notional.reindex(eval_times)

    # Plan trade schedule
    trade_times = eval_times + pd.to_timedelta(trade_delay_minutes, unit="min")
    trade_times = trade_times.intersection(index)

    # Map trade_time -> target_notional decided at eval_time = trade_time - delay
    plan = {}
    for t_eval in eval_times:
        t_trade = t_eval + pd.to_timedelta(trade_delay_minutes, unit="min")
        if t_trade in index:
            plan[t_trade] = float(target_notional.loc[t_eval]) if pd.notna(target_notional.loc[t_eval]) else np.nan

    cash = float(initial_capital)
    units = 0.0

    equity = []
    pnl = []
    pos_units = []
    pos_notional = []

    trades = []  # list of dicts

    prev_price = None

    for t in index:
        price = float(opens.loc[t])

        # PnL from previous minute to current open (using units held during that interval)
        if prev_price is not None:
            minute_pnl = units * (price - prev_price)
        else:
            minute_pnl = 0.0

        # Execute trade at time t if scheduled (at the open price)
        if t in plan and pd.notna(plan[t]):
            target_notional_t = float(plan[t])
            target_units = target_notional_t / price if price != 0 else 0.0
            delta_units = target_units - units
            traded_notional = abs(delta_units) * price
            cost = traded_notional * float(cost_bps) / 10000.0

            # Adjust cash for trade and cost
            cash -= (delta_units * price)  # buy reduces cash; sell increases cash
            cash -= cost                   # trading cost

            # Record trade
            trades.append({
                "time": t,
                "price": price,
                "target_notional": target_notional_t,
                "target_units": target_units,
                "delta_units": delta_units,
                "traded_notional": traded_notional,
                "cost": cost,
            })

            # Update position
            units = target_units

        # Equity after possible trade at t
        equity_t = cash + units * price

        # Append series
        equity.append(equity_t)
        pnl.append(minute_pnl)
        pos_units.append(units)
        pos_notional.append(units * price)

        prev_price = price

    equity_series = pd.Series(equity, index=index, name="equity")
    pnl_series = pd.Series(pnl, index=index, name="pnl")
    pos_units_series = pd.Series(pos_units, index=index, name="position_units")
    pos_notional_series = pd.Series(pos_notional, index=index, name="position_notional")

    returns = equity_series.pct_change().fillna(0.0)

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df.set_index("time", inplace=True)
        trades_df.index = pd.DatetimeIndex(trades_df.index)

    results = {
        "equity": equity_series,
        "returns": returns,
        "pnl": pnl_series,
        "position_units": pos_units_series,
        "position_notional": pos_notional_series,
        "trades": trades_df,
        "initial_capital": float(initial_capital),
    }
    return results 