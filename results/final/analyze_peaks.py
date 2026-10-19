"""
FedFlow-LLM — results/final/analyze_peaks.py
---------------------------------------------
Builds results/final/SUMMARY.md from the saved final-run files.

Reads only:   results/final/outputs/metrics.json
              results/final/outputs/forecasts.csv
Writes only:  results/final/SUMMARY.md

Peak check: for each held-out window, take the peak (max) of the 6 true
values and of the 6 predicted values. A "crossing" means peak > 100
(strictly above). Persistence's predicted peak is its repeated value,
which is the last history value.

Run from anywhere:  python3 results/final/analyze_peaks.py
"""
import csv
import json
import os
from collections import defaultdict

THRESHOLD = 100.0
HORIZON = 6

# Paths are relative to this script, so the working directory doesn't matter.
HERE = os.path.dirname(os.path.abspath(__file__))
METRICS_PATH = os.path.join(HERE, "outputs", "metrics.json")
FORECASTS_PATH = os.path.join(HERE, "outputs", "forecasts.csv")
SUMMARY_PATH = os.path.join(HERE, "SUMMARY.md")

NOT_COMPUTED = "not computed"


def fmt(value, digits=2):
    """Format a number, or say "not computed" if it is missing."""
    return NOT_COMPUTED if value is None else f"{value:.{digits}f}"


def ratio(numerator, denominator):
    """numerator/denominator as "a/b (xx.x%)", or "not computed" if the denominator is 0."""
    if denominator == 0:
        return NOT_COMPUTED
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def load_windows():
    """Group forecasts.csv rows by window start. Returns {start: [6 rows ordered by step]}."""
    windows = defaultdict(list)
    with open(FORECASTS_PATH, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            windows[int(row["start"])].append(row)
    for start, rows in windows.items():
        rows.sort(key=lambda r: int(r["step"]))
        if len(rows) != HORIZON:
            raise ValueError(f"Window {start} has {len(rows)} rows, expected {HORIZON}.")
    return dict(sorted(windows.items()))


def confusion(pairs):
    """pairs = [(true_peak, predicted_peak), ...] -> counts of the four outcomes."""
    counts = {"hit": 0, "miss": 0, "false_alarm": 0, "neither": 0}
    for true_peak, pred_peak in pairs:
        true_cross = true_peak > THRESHOLD
        pred_cross = pred_peak > THRESHOLD
        if true_cross and pred_cross:
            counts["hit"] += 1
        elif true_cross:
            counts["miss"] += 1
        elif pred_cross:
            counts["false_alarm"] += 1
        else:
            counts["neither"] += 1
    return counts


def main():
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    windows = load_windows()

    model_pairs, persistence_pairs = [], []
    model_unparsed = 0
    for rows in windows.values():
        true_peak = max(float(r["true"]) for r in rows)
        # Persistence repeats the last history value on every row, so its max is that value.
        persistence_pairs.append((true_peak, max(float(r["persistence"]) for r in rows)))
        if any(r["parsed"] != "True" or r["model"] == "" for r in rows):
            # A parse failure has no model prediction; it is left out of the model's counts.
            model_unparsed += 1
            continue
        model_pairs.append((true_peak, max(float(r["model"]) for r in rows)))

    results = {
        "Model (federated LoRA)": (confusion(model_pairs), len(model_pairs)),
        "Persistence baseline": (confusion(persistence_pairs), len(persistence_pairs)),
    }

    lines = [
        "# Final run summary",
        "",
        "All numbers below are read from or computed from `outputs/metrics.json` and "
        "`outputs/forecasts.csv` by `analyze_peaks.py`.",
        "",
        "## Forecast error on held-out windows (from metrics.json)",
        "",
        f"Windows used: {metrics.get('windows_used', NOT_COMPUTED)}. "
        f"Parse failures: {metrics.get('parse_failures', NOT_COMPUTED)}. "
        f"Windows scored: {metrics.get('windows_scored', NOT_COMPUTED)}.",
        "",
        "| Method | MAE | RMSE |",
        "|---|---:|---:|",
    ]
    for key, label in [("model", "Model (federated LoRA)"),
                       ("persistence", "Persistence (repeat last value)"),
                       ("moving_avg", "Moving average (last 3 values)")]:
        entry = metrics.get(key) or {}
        lines.append(f"| {label} | {fmt(entry.get('mae'))} | {fmt(entry.get('rmse'))} |")

    lines += [
        "",
        f"## Peak crossing of {THRESHOLD:g} (peak of the 6-step horizon, strictly above {THRESHOLD:g})",
        "",
        f"Windows in forecasts.csv: {len(windows)}. "
        f"Windows without a parsed model prediction (excluded from the model row): {model_unparsed}.",
        "",
        "| Method | Windows | True cross, predicted cross | True cross, missed | False alarm | Neither |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for label, (c, n) in results.items():
        lines.append(f"| {label} | {n} | {c['hit']} | {c['miss']} | {c['false_alarm']} | {c['neither']} |")

    lines += [
        "",
        "| Method | Recall (hits / true crossings) | Precision (hits / predicted crossings) |",
        "|---|---:|---:|",
    ]
    for label, (c, _) in results.items():
        recall = ratio(c["hit"], c["hit"] + c["miss"])
        precision = ratio(c["hit"], c["hit"] + c["false_alarm"])
        lines.append(f"| {label} | {recall} | {precision} |")

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
