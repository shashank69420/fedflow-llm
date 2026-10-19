# Final run summary

All numbers below are read from or computed from `outputs/metrics.json` and `outputs/forecasts.csv` by `analyze_peaks.py`.

## Forecast error on held-out windows (from metrics.json)

Windows used: 91. Parse failures: 0. Windows scored: 91.

| Method | MAE | RMSE |
|---|---:|---:|
| Model (federated LoRA) | 10.34 | 15.57 |
| Persistence (repeat last value) | 11.29 | 16.96 |
| Moving average (last 3 values) | 10.86 | 16.97 |

## Peak crossing of 100 (peak of the 6-step horizon, strictly above 100)

Windows in forecasts.csv: 91. Windows without a parsed model prediction (excluded from the model row): 0.

| Method | Windows | True cross, predicted cross | True cross, missed | False alarm | Neither |
|---|---:|---:|---:|---:|---:|
| Model (federated LoRA) | 91 | 46 | 15 | 0 | 30 |
| Persistence baseline | 91 | 51 | 10 | 1 | 29 |

| Method | Recall (hits / true crossings) | Precision (hits / predicted crossings) |
|---|---:|---:|
| Model (federated LoRA) | 46/61 (75.4%) | 46/46 (100.0%) |
| Persistence baseline | 51/61 (83.6%) | 51/52 (98.1%) |
