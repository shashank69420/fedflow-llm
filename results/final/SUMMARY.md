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

## Onset analysis (last history value vs the 100 threshold)

- Onset window: last value <= 100 and true peak > 100.
- Calm window: last value <= 100 and true peak <= 100.
- Already-high window: last value > 100.

Windows without a parsed model prediction (excluded from model counts below): 0.

| Window type | Windows |
|---|---:|
| Onset | 10 |
| Calm | 29 |
| Already-high | 52 |
| Total | 91 |

### Onset windows

| | Count |
|---|---:|
| Onset windows scored | 10 |
| Model flagged (model peak > 100) | 1 |
| Model missed (model peak <= 100) | 9 |
| Model recall on onsets | 1/10 (10.0%) |
| Mean (model peak - true peak) | -29.74 |

Persistence flags 0 onset windows by definition: its predicted peak is the last value, which is <= 100.

### Calm windows

| | Count |
|---|---:|
| Calm windows scored | 29 |
| Model false alarms (model peak > 100) | 0/29 (0.0%) |

### Already-high windows

Already-high windows scored: 52.

| | True peak > 100 | True peak <= 100 | Total |
|---|---:|---:|---:|
| Model peak > 100 | stays high and model agrees: 45 | traffic falls but model still flags high: 0 | still flags high: 45 |
| Model peak <= 100 | wrong drop: model predicts a drop but traffic stays high: 6 | correct drop: 1 | predicts a drop: 7 |
