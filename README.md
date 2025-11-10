## Crypto Multi-Factor CTA Backtest (Single-Asset)

This repository contains a modular, config-driven backtesting framework for a single-asset CTA strategy. It now supports:
- Factor profiles (single source of truth for per-factor params)
- Separate runners for single-factor and portfolio (combo) backtests
- Threshold/continuous trade modes, stop-loss, monthly risk control, buy&hold baseline, summaries

### Directory structure
```
crypto_quant/
  engine/
    ...
    profile_loader.py        # Loads factor profiles with validation
    combo.py                 # Combo builders (profiles-based)
    portfolio_backtester.py  # Multi-sleeve execution
    risk.py                  # Volatility-based risk scaler
  factors/
    registry.py
    example_factors.py
    profiles/
      pvcorr.yaml
      double_ma.yaml
  configs/
    single/
      pvcorr.yaml
    portfolio/
      combo_pvcorr_double_ma.yaml
  run.py                    # Unified runner (auto-detects single vs portfolio)
  reports/                  # Outputs
  ...
```

### Factor profiles
- Location: `factors/profiles/{name}.yaml`
- Must include `factor: {name}` matching the registry name
- Example (`factors/profiles/pvcorr.yaml`):

```yaml
factor: pvcorr
lookback_minutes: 5000
k_minutes: 720
rebalance_minutes: 720
mapper: percentile
zscore_window: 100
clip_abs_signal: 1.0
trade_mode: threshold
entry_long_threshold: 0.90
entry_short_threshold: -0.90
allow_short: false
long_leverage: 1.0
short_leverage: 1.0
stop_loss_pct: 0.03
```

### Unified runner (preferred)
- The unified runner auto-detects mode based on the config contents.
  - If `sleeves` is present and non-empty → portfolio run
  - Else if `factor_profile` is present → single-factor run

Usage:
```bash
# Single-factor
python run.py --config configs/single/pvcorr.yaml

# Portfolio
python run.py --config configs/portfolio/combo_pvcorr_double_ma.yaml
```

### Single-factor config
`configs/single/pvcorr.yaml`:
```yaml
type: single

data:
  source_csv: okx_BTCUSDT_1min_data.csv
  timezone: UTC
  start: 2020-01-01 00:00:00
  end: 2025-01-01 00:00:00
  forward_fill: true

factor_profile: pvcorr

execution:
  initial_capital: 100000
  trade_delay_minutes: 1
  cost_bps: 7

reporting:
  out_dir: reports
  plots: [pnl, equity]
  write_results_core: false
  folder: pvcorr
```
Run:
```bash
python run.py --config configs/single/pvcorr.yaml
```

### Portfolio (combo) config
`configs/portfolio/combo_pvcorr_double_ma.yaml`:
```yaml
type: portfolio
name: combo_pvcorr_double_ma

data:
  source_csv: okx_BTCUSDT_1min_data.csv
  timezone: UTC
  start: 2020-01-01 00:00:00
  end: 2025-01-01 00:00:00
  forward_fill: true

sleeves:
  - profile: pvcorr
    weight: 0.5
  - profile: double_ma
    weight: 0.5

execution:
  initial_capital: 100000
  trade_delay_minutes: 1
  cost_bps: 7

risk_control:
  enabled: true
  vol_lookback_days: 63
  vol_target_ann: 0.8
  rebalance: monthly
  rebalance_days: 30

reporting:
  out_dir: reports
  plots: [pnl, equity]
  write_results_core: false
  folder: combo_pvcorr_double_ma
```
Run:
```bash
python run.py --config configs/portfolio/combo_pvcorr_double_ma.yaml
```

### Risk control
- Base scaler: `target_vol / recent_vol`; computed monthly (or every N days) and forward-filled to minute bars.
- Per-sleeve leverage floor: `scaler = max(base_scaler, sleeve_leverage)` with `sleeve_leverage = max(|long_leverage|, |short_leverage|)`.
- Combo uses mapper with leverage=1.0 and applies scaler after mapping to avoid double counting.

### Reporting
- Outputs to `reports/{folder or name}/` with:
  - `summary.json` (annual return, vol, Sharpe, max drawdown, Calmar)
  - Plots: `pnl.png`, `net_value.png` (with buy&hold overlay)
  - `backtest.yaml` (effective config used)
  - IC metrics/plots are skipped when trade_mode is threshold
- Set `reporting.write_results_core: true` to also dump `results_core.csv`.

### Legacy cleanup
- The old entry points and monolithic config have been removed in favor of profiles + unified runner:
  - Removed: `config/backtest.yaml`, `main.py`, `run_single.py`, `run_portfolio.py`.
- Use `run.py` with the configs in `configs/single/` and `configs/portfolio/` instead. 