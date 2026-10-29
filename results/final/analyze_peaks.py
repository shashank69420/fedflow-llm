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

--split val reads only outputs/val_forecasts.csv from --dir and writes SUMMARY_val.md there (SUMMARY.md is
never touched). It compares the model with the slope baseline on onset windows, per cell and pooled.
The slope column is optional: files without it (older runs) still work and show "not computed".
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
MIN_ONSET_WINDOWS = 10   # a cell needs at least this many onset windows before its numbers are reported


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


def val_records(windows):
    """One record per window for the validation summary: cell, last history value, true peak, model peak and slope peak.
    A peak is None when it is missing (model parse failure, or a file without a slope column)."""
    records = []
    for (cell, _start), rows in windows.items():
        parsed = all(r["parsed"] == "True" and r["model"] != "" for r in rows)
        has_slope = all(r.get("slope") not in (None, "") for r in rows)   # old files have no slope column
        records.append({
            "cell": cell,
            "true_peak": max(float(r["true"]) for r in rows),
            "last_value": max(float(r["persistence"]) for r in rows),
            "model_peak": max(float(r["model"]) for r in rows) if parsed else None,
            "slope_peak": max(float(r["slope"]) for r in rows) if has_slope else None,
        })
    return records


def onset_row(onsets, peak_key):
    """(recall text, mean error text) for one forecaster over the onset records `onsets`."""
    if not onsets or any(w[peak_key] is None for w in onsets):
        return NOT_COMPUTED, NOT_COMPUTED
    flagged = sum(w[peak_key] > THRESHOLD for w in onsets)
    mean_diff = sum(w[peak_key] - w["true_peak"] for w in onsets) / len(onsets)
    return ratio(flagged, len(onsets)), fmt(mean_diff)


def val_summary(records, source_name):
    """Markdown lines for SUMMARY_val.md: pooled onset result first, then one row per cell."""
    t = f"{THRESHOLD:g}"
    # Only windows the model parsed count, for both the model and slope, so the two are compared on the same windows.
    scored = [w for w in records if w["model_peak"] is not None]
    onsets = [w for w in scored if w["last_value"] <= THRESHOLD and w["true_peak"] > THRESHOLD]
    cells = sorted({w["cell"] for w in records}, key=lambda c: (len(c), c))

    pooled_model = onset_row(onsets, "model_peak")
    pooled_slope = onset_row(onsets, "slope_peak")
    lines = [
        "# Validation run summary",
        "",
        f"Computed by `analyze_peaks.py --split val` from `outputs/{source_name}` only. No test file is read.",
        "",
        f"**Pooled onset windows (all cells): {len(onsets)}**",
        "",
        f"- Onset window: last history value <= {t} and true peak > {t}.",
        f"- Recall = onset windows whose predicted peak is > {t}. Mean error = mean of (predicted peak - true peak) over onset windows; negative means under-prediction.",
        f"- Windows in the file: {len(records)}. Windows without a parsed model prediction (left out for both methods): {len(records) - len(scored)}.",
        "",
        "## Pooled over all cells (the figure to read)",
        "",
        "| Method | Onset windows | Recall | Mean (predicted peak - true peak) |",
        "|---|---:|---:|---:|",
        f"| Model (federated LoRA) | {len(onsets)} | {pooled_model[0]} | {pooled_model[1]} |",
        f"| Slope baseline | {len(onsets)} | {pooled_slope[0]} | {pooled_slope[1]} |",
        "",
        "Validation windows use stride 6 and each holds 12 history + 6 future values, so neighbouring windows overlap "
        "and the onset windows are not independent samples. Read the counts as a rough size, not as that many separate events.",
    ]
    if len(onsets) < MIN_ONSET_WINDOWS:
        lines += ["", f"Even pooled, there are fewer than {MIN_ONSET_WINDOWS} onset windows, so treat these figures as weak evidence."]

    lines += [
        "",
        f"## Per cell (reported only with at least {MIN_ONSET_WINDOWS} onset windows)",
        "",
        "| Cell | Onset windows | Model recall | Model mean error | Slope recall | Slope mean error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for cell in cells:
        cell_onsets = [w for w in onsets if w["cell"] == cell]
        label = cell if cell != "" else "all"
        if len(cell_onsets) < MIN_ONSET_WINDOWS:
            note = f"not reportable (onset windows: {len(cell_onsets)})"
            lines.append(f"| {label} | {len(cell_onsets)} | {note} | | | |")
            continue
        m, sl = onset_row(cell_onsets, "model_peak"), onset_row(cell_onsets, "slope_peak")
        lines.append(f"| {label} | {len(cell_onsets)} | {m[0]} | {m[1]} | {sl[0]} | {sl[1]} |")
    lines.append(f"| Sum of the per-cell counts | {len(onsets)} | | | | |")
    return lines


def run_val(args):
    """--split val: read val_forecasts.csv and write SUMMARY_val.md (never SUMMARY.md)."""
    forecasts_path = os.path.join(args.dir, "outputs", "val_forecasts.csv")
    summary_path = os.path.join(args.dir, "SUMMARY_val.md")
    records = val_records(load_windows(forecasts_path))
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_summary(records, "val_forecasts.csv")) + "\n")
    print(f"Wrote {summary_path}")


def run_selftest():
    """Check the validation summary on tiny made-up windows (no files read or written)."""
    def window(cell, start, last, true_peak, model_peak, slope_peak):
        rows = []
        for step in range(1, HORIZON + 1):
            rows.append({"cell": cell, "start": str(start), "step": str(step),
                         "true": str(true_peak if step == 3 else 10), "persistence": str(last),
                         "model": "" if model_peak is None else str(model_peak if step == 3 else 10),
                         "parsed": str(model_peak is not None), "slope": "" if slope_peak is None else str(slope_peak if step == 3 else 10)})
        return rows
    # 12 onsets in cell 0 (model flags 6 at 120 and misses 6 at 50; slope flags 3), 2 onsets in cell 1, one calm window, one parse failure.
    rows = []
    for i in range(12):
        rows += window("0", i, 50, 150, 120 if i < 6 else 50, 130 if i < 3 else 20)
    rows += window("1", 0, 50, 150, 120, 20) + window("1", 1, 50, 150, 50, 20)
    rows += window("1", 2, 50, 60, 40, 30) + window("1", 3, 50, 150, None, 130)
    windows = defaultdict(list)
    for r in rows:
        windows[(r["cell"], int(r["start"]))].append(r)
    records = val_records(dict(windows))
    text = "\n".join(val_summary(records, "x.csv"))
    assert "**Pooled onset windows (all cells): 14**" in text          # 12 + 2 (the parse failure is left out)
    # Model: 7 flagged of 14; errors are 7 x (120 - 150) and 7 x (50 - 150), so (-210 - 700) / 14 = -65.00.
    assert "| Model (federated LoRA) | 14 | 7/14 (50.0%) | -65.00 |" in text
    # Slope: 3 flagged of 14; errors are 3 x (130 - 150) and 11 x (20 - 150), so (-60 - 1430) / 14 = -106.43.
    assert "| Slope baseline | 14 | 3/14 (21.4%) | -106.43 |" in text
    assert "| 0 | 12 | 6/12 (50.0%)" in text                            # cell 0 has enough onset windows
    assert "| 1 | 2 | not reportable (onset windows: 2) |" in text      # cell 1 does not
    # A file without the slope column still works and says so.
    for r in rows:
        del r["slope"]
    windows = defaultdict(list)
    for r in rows:
        windows[(r["cell"], int(r["start"]))].append(r)
    assert "| Slope baseline | 14 | not computed | not computed |" in "\n".join(val_summary(val_records(dict(windows)), "x.csv"))
    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description="Peak-crossing summary of a saved final run")
    parser.add_argument("--dir", default=HERE, help="folder with outputs/metrics.json and outputs/forecasts.csv (default: this script's folder)")
    parser.add_argument("--split", choices=["test", "val"], default="test",
                        help="test (default): write SUMMARY.md as before. val: read val_forecasts.csv and write SUMMARY_val.md")
    parser.add_argument("--selftest", action="store_true", help="run the built-in checks and exit")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return
    if args.split == "val":
        run_val(args)
        return
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
