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
                    python3 results/final/analyze_peaks.py --dir results/final_telecom
--dir is a folder holding outputs/metrics.json and outputs/forecasts.csv; SUMMARY.md is
written into that folder. The default is the folder this script is in.
"""
import argparse
import csv
import json
import os
from collections import defaultdict

THRESHOLD = 100.0
HORIZON = 6

# Default folder is this script's own, so the working directory doesn't matter.
HERE = os.path.dirname(os.path.abspath(__file__))

NOT_COMPUTED = "not computed"


def fmt(value, digits=2):
    """Format a number, or say "not computed" if it is missing."""
    return NOT_COMPUTED if value is None else f"{value:.{digits}f}"


def ratio(numerator, denominator):
    """numerator/denominator as "a/b (xx.x%)", or "not computed" if the denominator is 0."""
    if denominator == 0:
        return NOT_COMPUTED
    return f"{numerator}/{denominator} ({100 * numerator / denominator:.1f}%)"


def load_windows(forecasts_path):
    """Group forecasts.csv rows by window. Returns {(cell, start): [6 rows ordered by step]}.
    Files without a cell column (the synthetic run) get cell "" and group by start alone, as before."""
    windows = defaultdict(list)
    with open(forecasts_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            windows[(row.get("cell", ""), int(row["start"]))].append(row)
    for key, rows in windows.items():
        rows.sort(key=lambda r: int(r["step"]))
        if len(rows) != HORIZON:
            raise ValueError(f"Window {key} has {len(rows)} rows, expected {HORIZON}.")
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


def classify(window):
    """onset: starts low, peaks high. calm: starts low, stays low. already_high: starts high."""
    if window["last_value"] > THRESHOLD:
        return "already_high"
    return "onset" if window["true_peak"] > THRESHOLD else "calm"


def onset_section(records):
    """Markdown lines for the onset analysis (windows split by their last history value)."""
    groups = {"onset": [], "calm": [], "already_high": []}
    for w in records:
        groups[classify(w)].append(w)
    total = sum(len(g) for g in groups.values())
    if total != len(records):
        raise ValueError(f"Window classes sum to {total}, expected {len(records)}.")

    # Model counts only use windows where the model's prediction parsed.
    scored = {name: [w for w in g if w["model_peak"] is not None] for name, g in groups.items()}
    unparsed = len(records) - sum(len(s) for s in scored.values())

    onset = scored["onset"]
    onset_flagged = sum(w["model_peak"] > THRESHOLD for w in onset)
    onset_missed = len(onset) - onset_flagged
    # Mean of (model peak - true peak); negative means the model under-predicts the peak.
    onset_mean_diff = (sum(w["model_peak"] - w["true_peak"] for w in onset) / len(onset)) if onset else None

    calm = scored["calm"]
    calm_false_alarms = sum(w["model_peak"] > THRESHOLD for w in calm)

    high = scored["already_high"]
    model_high = [w for w in high if w["model_peak"] > THRESHOLD]
    model_drop = [w for w in high if w["model_peak"] <= THRESHOLD]
    agrees_high = sum(w["true_peak"] > THRESHOLD for w in model_high)
    falls_but_flagged = len(model_high) - agrees_high
    wrong_drop = sum(w["true_peak"] > THRESHOLD for w in model_drop)
    correct_drop = len(model_drop) - wrong_drop

    t = f"{THRESHOLD:g}"
    return [
        "",
        f"## Onset analysis (last history value vs the {t} threshold)",
        "",
        f"- Onset window: last value <= {t} and true peak > {t}.",
        f"- Calm window: last value <= {t} and true peak <= {t}.",
        f"- Already-high window: last value > {t}.",
        "",
        f"Windows without a parsed model prediction (excluded from model counts below): {unparsed}.",
        "",
        "| Window type | Windows |",
        "|---|---:|",
        f"| Onset | {len(groups['onset'])} |",
        f"| Calm | {len(groups['calm'])} |",
        f"| Already-high | {len(groups['already_high'])} |",
        f"| Total | {total} |",
        "",
        "### Onset windows",
        "",
        "| | Count |",
        "|---|---:|",
        f"| Onset windows scored | {len(onset)} |",
        f"| Model flagged (model peak > {t}) | {onset_flagged} |",
        f"| Model missed (model peak <= {t}) | {onset_missed} |",
        f"| Model recall on onsets | {ratio(onset_flagged, len(onset))} |",
        f"| Mean (model peak - true peak) | {fmt(onset_mean_diff)} |",
        "",
        f"Persistence flags 0 onset windows by definition: its predicted peak is the last value, which is <= {t}.",
        "",
        "### Calm windows",
        "",
        "| | Count |",
        "|---|---:|",
        f"| Calm windows scored | {len(calm)} |",
        f"| Model false alarms (model peak > {t}) | {ratio(calm_false_alarms, len(calm))} |",
        "",
        "### Already-high windows",
        "",
        f"Already-high windows scored: {len(high)}.",
        "",
        f"| | True peak > {t} | True peak <= {t} | Total |",
        "|---|---:|---:|---:|",
        f"| Model peak > {t} | stays high and model agrees: {agrees_high} "
        f"| traffic falls but model still flags high: {falls_but_flagged} "
        f"| still flags high: {len(model_high)} |",
        f"| Model peak <= {t} | wrong drop: model predicts a drop but traffic stays high: {wrong_drop} "
        f"| correct drop: {correct_drop} "
        f"| predicts a drop: {len(model_drop)} |",
    ]


def main():
    parser = argparse.ArgumentParser(description="Peak-crossing summary of a saved final run")
    parser.add_argument("--dir", default=HERE, help="folder with outputs/metrics.json and outputs/forecasts.csv (default: this script's folder)")
    args = parser.parse_args()
    metrics_path = os.path.join(args.dir, "outputs", "metrics.json")
    forecasts_path = os.path.join(args.dir, "outputs", "forecasts.csv")
    summary_path = os.path.join(args.dir, "SUMMARY.md")

    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    windows = load_windows(forecasts_path)

    # One record per window: true peak, last history value, model peak (None = parse failure).
    records = []
    for rows in windows.values():
        parsed = all(r["parsed"] == "True" and r["model"] != "" for r in rows)
        records.append({
            "true_peak": max(float(r["true"]) for r in rows),
            # Persistence repeats the last history value on every row, so its max is that value.
            "last_value": max(float(r["persistence"]) for r in rows),
            "model_peak": max(float(r["model"]) for r in rows) if parsed else None,
        })

    persistence_pairs = [(w["true_peak"], w["last_value"]) for w in records]
    # A parse failure has no model prediction; it is left out of the model's counts.
    model_pairs = [(w["true_peak"], w["model_peak"]) for w in records if w["model_peak"] is not None]
    model_unparsed = len(records) - len(model_pairs)

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

    lines += onset_section(records)

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
