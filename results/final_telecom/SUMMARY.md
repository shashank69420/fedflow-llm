# Telecom final run: onset recall and false alarms

Source: `results/final_telecom/outputs/forecasts.csv` (366 test windows, 6 cells). Every number below was computed from that file alone. Onset, calm and already-high follow the definitions in `results/final/analyze_peaks.py`. The model flags an onset when its 6-step peak is above 100.

Note on the reporting rule: at these strides only cells 8336 (stride 3, 14 onsets) and 7511 (stride 3, 14 onsets) have at least 10 onset windows. Cells 9721, 6997, 8062 and 7584 have 6, 9, 6 and 2 onsets, so their recall is not reportable.

## Onset windows and model recall (per cell)

Onset window: last history value <= 100 and true peak > 100. The model flags an onset when its peak > 100. Recall is reported only where a cell has at least 10 onset windows.

| cell | square | windows | onset windows | model flagged onsets | model onset recall | persistence flagged onsets |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9721 | 46 | 6 | 0 | not reportable (6 onset windows) | 0 |
| 1 | 6997 | 46 | 9 | 2 | not reportable (9 onset windows) | 0 |
| 2 | 8336 | 91 | 14 | 4 | 4/14 (28.6%) | 0 |
| 3 | 7511 | 91 | 14 | 6 | 6/14 (42.9%) | 0 |
| 4 | 8062 | 46 | 6 | 2 | not reportable (6 onset windows) | 0 |
| 5 | 7584 | 46 | 2 | 0 | not reportable (2 onset windows) | 0 |

## Pooled onset recall

Cells with at least 10 onset windows: [2, 3] (2 of 6).
Pooled over those cells: 10/28 onset windows flagged by the model (35.7%).
The windows overlap: consecutive windows share history and forecast slots (stride 3 for the experiment cells, stride 6 for the others), so these onset windows are not independent samples.

## Persistence onset recall

Persistence repeats the last history value, which is <= 100 in every onset window, so its peak is never above 100. Its onset recall is 0 by definition (flagged onset windows: 0).

## Mean (model peak - true peak) over onset windows, per cell

Negative means the model under-predicts the peak.

| cell | square | onset windows | mean (model peak - true peak) |
|---:|---:|---:|---:|
| 0 | 9721 | 6 | -30.78 |
| 1 | 6997 | 9 | -39.77 |
| 2 | 8336 | 14 | -33.36 |
| 3 | 7511 | 14 | -44.04 |
| 4 | 8062 | 6 | -36.35 |
| 5 | 7584 | 2 | -45.57 |

## Calm and already-high windows: model false alarms, per cell

Calm: last value <= 100 and true peak <= 100. Already-high: last value > 100. A calm false alarm is a calm window where the model peak is > 100. For already-high windows, the model keeps the peak high (> 100) or predicts a drop (<= 100); a wrong drop is a predicted drop where the true peak stays above 100.

| cell | square | calm windows | calm false alarms | already-high windows | model keeps high | model predicts drop | wrong drop |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 9721 | 37 | 2 | 3 | 1 | 2 | 1 |
| 1 | 6997 | 20 | 0 | 17 | 15 | 2 | 1 |
| 2 | 8336 | 33 | 0 | 44 | 44 | 0 | 0 |
| 3 | 7511 | 18 | 0 | 59 | 56 | 3 | 2 |
| 4 | 8062 | 9 | 1 | 31 | 30 | 1 | 1 |
| 5 | 7584 | 9 | 0 | 35 | 35 | 0 | 0 |

