"""
FedFlow-LLM — compare_variants.py
----------------------------------
Compares a v1 and a v2 validation run and applies the four adoption rules of PLAN_V2.md:

  1. overall model MAE of v2 is lower than v1's (strictly)
  2. pooled onset recall over all six cells of v2 is higher than v1's (strictly)
  3. v2's false-alarm rate on calm windows is at most 5 percentage points above v1's
  4. v2 has no more parse failures than v1

Reads only the two runs' val_metrics.json and val_forecasts.csv (written by infer.py --eval --split val)
and writes one markdown report that ends with ADOPT or DO NOT ADOPT.

Window classes, as in results/final/analyze_peaks.py (threshold 100):
  onset         last history value <= 100 and true peak (max of the 6 true values) > 100
  calm          last history value <= 100 and true peak <= 100
  already-high  last history value > 100
A window is flagged when the predicted peak (max of the 6 predictions) is > 100. Recall and false-alarm
rate count only windows whose model prediction parsed. The last history value is the persistence column.

Usage (from anywhere; no GPU, no model):
  python3 compare_variants.py --v1 DIR_OR_FILES --v2 DIR_OR_FILES --out report.md
      DIR is a folder holding val_metrics.json and val_forecasts.csv, directly or in outputs/ or training/outputs/
  python3 compare_variants.py --v1-metrics F --v1-forecasts F --v2-metrics F --v2-forecasts F --out report.md
  python3 compare_variants.py --selftest
"""
import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

THRESHOLD = 100.0
HORIZON = 6
MAX_FALSE_ALARM_RISE_PP = 5.0   # rule 3: at most 5 percentage points above v1
METHODS = ("model", "persistence", "moving_avg", "slope")
LABELS = {"model": "Model", "persistence": "Persistence", "moving_avg": "Moving average (last 3)", "slope": "Slope baseline"}


# ---------------------------------------------------------------- reading

def find_files(folder):
    """(metrics path, forecasts path) inside folder, folder/outputs or folder/training/outputs."""
    for sub in ("", "outputs", os.path.join("training", "outputs")):
        m = os.path.join(folder, sub, "val_metrics.json")
        f = os.path.join(folder, sub, "val_forecasts.csv")
        if os.path.isfile(m) and os.path.isfile(f):
            return m, f
    sys.exit(f"ERROR: no val_metrics.json + val_forecasts.csv pair in {folder}, {folder}/outputs or {folder}/training/outputs.")


def load_windows(forecasts_path):
    """{(cell, start): record} from a val_forecasts.csv. Record: last value, true peak, model peak (None = unparsed),
    slope peak (None = no slope column)."""
    rows_by_window = defaultdict(list)
    with open(forecasts_path, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows_by_window[(row.get("cell", ""), int(row["start"]))].append(row)
    windows = {}
    for key, rows in sorted(rows_by_window.items()):
        rows.sort(key=lambda r: int(r["step"]))
        if len(rows) != HORIZON:
            sys.exit(f"ERROR: {forecasts_path}: window {key} has {len(rows)} rows, expected {HORIZON}.")
        parsed = all(r["parsed"] == "True" and r["model"] != "" for r in rows)
        has_slope = all(r.get("slope") not in (None, "") for r in rows)   # older files have no slope column
        windows[key] = {
            "cell": key[0],
            "last": float(rows[0]["persistence"]),     # persistence repeats the last history value
            "true_peak": max(float(r["true"]) for r in rows),
            "model_peak": max(float(r["model"]) for r in rows) if parsed else None,
            "slope_peak": max(float(r["slope"]) for r in rows) if has_slope else None,
        }
    return windows


def load_run(metrics_path, forecasts_path):
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return {"metrics": metrics, "windows": load_windows(forecasts_path),
            "metrics_path": metrics_path, "forecasts_path": forecasts_path}


# ---------------------------------------------------------------- measures

def window_class(w):
    if w["last"] > THRESHOLD:
        return "already_high"
    return "onset" if w["true_peak"] > THRESHOLD else "calm"


def flag_stats(windows, peak_key, kind):
    """(flagged, total) over parsed windows of one class; a window counts only if its model prediction parsed,
    so the model and the slope baseline are measured on the same windows."""
    pool = [w for w in windows if window_class(w) == kind and w["model_peak"] is not None]
    if any(w[peak_key] is None for w in pool):
        return None, len(pool)                    # e.g. no slope column in this file
    return sum(1 for w in pool if w[peak_key] > THRESHOLD), len(pool)


def mean_onset_error(windows, peak_key):
    """Mean of (predicted peak - true peak) over parsed onset windows, or None."""
    pool = [w for w in windows if window_class(w) == "onset" and w["model_peak"] is not None]
    if not pool or any(w[peak_key] is None for w in pool):
        return None
    return sum(w[peak_key] - w["true_peak"] for w in pool) / len(pool)


def rate(flagged, total):
    return None if flagged is None or total == 0 else flagged / total


def summarize(run):
    """Every measure the report and the rules need, for one run."""
    windows = list(run["windows"].values())
    cells = sorted({w["cell"] for w in windows}, key=lambda c: (len(c), c))
    m = run["metrics"]
    out = {
        "variant": m.get("variant", "v1 (no variant field)"),
        "windows_used": m.get("windows_used"),
        "windows_scored": m.get("windows_scored"),
        "parse_failures": m.get("parse_failures"),
        "csv_unparsed": sum(1 for w in windows if w["model_peak"] is None),
        "errors": {k: m.get(k) or {} for k in METHODS},
        "per_cell_errors": m.get("per_cell", {}),
        "classes": {k: sum(1 for w in windows if window_class(w) == k) for k in ("onset", "calm", "already_high")},
        "cells": cells,
        "pooled": {}, "per_cell": {},
    }
    for peak_key, name in (("model_peak", "model"), ("slope_peak", "slope")):
        f_on, n_on = flag_stats(windows, peak_key, "onset")
        f_calm, n_calm = flag_stats(windows, peak_key, "calm")
        out["pooled"][name] = {"onset_flagged": f_on, "onset_n": n_on, "recall": rate(f_on, n_on),
                               "calm_flagged": f_calm, "calm_n": n_calm, "false_alarm": rate(f_calm, n_calm),
                               "mean_onset_error": mean_onset_error(windows, peak_key)}
        for cell in cells:
            ws = [w for w in windows if w["cell"] == cell]
            f_on, n_on = flag_stats(ws, peak_key, "onset")
            f_calm, n_calm = flag_stats(ws, peak_key, "calm")
            out["per_cell"].setdefault(cell, {})[name] = {
                "onset_flagged": f_on, "onset_n": n_on, "recall": rate(f_on, n_on),
                "calm_flagged": f_calm, "calm_n": n_calm, "false_alarm": rate(f_calm, n_calm),
                "mean_onset_error": mean_onset_error(ws, peak_key)}
    return out


# ---------------------------------------------------------------- rules

def apply_rules(s1, s2):
    """[(rule text, v1 value text, v2 value text, passed)] and the overall decision (True = adopt).
    A rule whose inputs are missing fails, so a broken file can never lead to ADOPT."""
    def pct(x):
        return "not computed" if x is None else f"{100 * x:.2f}%"

    rules = []
    mae1, mae2 = s1["errors"]["model"].get("mae"), s2["errors"]["model"].get("mae")
    ok = mae1 is not None and mae2 is not None and mae2 < mae1
    rules.append(("1. Overall model MAE lower than v1's", fmt(mae1, 4), fmt(mae2, 4), ok))

    p1, p2 = s1["pooled"]["model"], s2["pooled"]["model"]
    ok = p1["recall"] is not None and p2["recall"] is not None and p2["recall"] > p1["recall"]
    rules.append(("2. Pooled onset recall (all cells) higher than v1's",
                  f"{counts(p1['onset_flagged'], p1['onset_n'])}", f"{counts(p2['onset_flagged'], p2['onset_n'])}", ok))

    fa1, fa2 = p1["false_alarm"], p2["false_alarm"]
    rise = None if fa1 is None or fa2 is None else 100 * (fa2 - fa1)
    ok = rise is not None and rise <= MAX_FALSE_ALARM_RISE_PP + 1e-9   # tolerance only for float rounding at exactly 5.00
    rules.append((f"3. Calm false-alarm rate at most {MAX_FALSE_ALARM_RISE_PP:g} points above v1's "
                  f"(v2 - v1 = {'not computed' if rise is None else f'{rise:+.2f} points'})",
                  f"{counts(p1['calm_flagged'], p1['calm_n'])}", f"{counts(p2['calm_flagged'], p2['calm_n'])}", ok))

    pf1, pf2 = s1["parse_failures"], s2["parse_failures"]
    ok = pf1 is not None and pf2 is not None and pf2 <= pf1
    rules.append(("4. Parse failures no more than v1's", str(pf1), str(pf2), ok))
    return rules, all(r[3] for r in rules)


def fmt(x, digits=2):
    return "not computed" if x is None else f"{x:.{digits}f}"


def counts(flagged, total):
    if flagged is None:
        return f"not computed (n={total})"
    if total == 0:
        return "no windows (0/0)"
    return f"{flagged}/{total} ({100 * flagged / total:.1f}%)"


# ---------------------------------------------------------------- report

def check_comparable(r1, r2, s2):
    """Both runs must score exactly the same validation windows, and the v2 run must say it is v2."""
    k1, k2 = set(r1["windows"]), set(r2["windows"])
    if k1 != k2:
        sys.exit(f"ERROR: the two runs scored different windows ({len(k1)} vs {len(k2)}; "
                 f"{len(k1 - k2)} only in v1, {len(k2 - k1)} only in v2). Not comparable.")
    if s2["variant"] != "v2":
        sys.exit(f"ERROR: the v2 run's metrics say variant {s2['variant']!r}, not 'v2'. Wrong file?")


def build_report(r1, r2):
    s1, s2 = summarize(r1), summarize(r2)
    check_comparable(r1, r2, s2)
    rules, adopt = apply_rules(s1, s2)
    L = ["# v1 vs v2 on the validation block", "",
         "Written by `training/compare_variants.py` from these files only:", "",
         f"- v1: `{r1['metrics_path']}`, `{r1['forecasts_path']}`",
         f"- v2: `{r2['metrics_path']}`, `{r2['forecasts_path']}`", "",
         "Adoption rules from PLAN_V2.md; all four must hold.", "",
         "| Rule | v1 | v2 | Holds |", "|---|---|---|---|"]
    L += [f"| {text} | {a} | {b} | {'yes' if ok else 'NO'} |" for text, a, b, ok in rules]
    L += ["", "## Runs", "", "| | v1 | v2 |", "|---|---:|---:|",
          f"| Variant (from metrics) | {s1['variant']} | {s2['variant']} |",
          f"| Windows used | {s1['windows_used']} | {s2['windows_used']} |",
          f"| Windows scored | {s1['windows_scored']} | {s2['windows_scored']} |",
          f"| Parse failures (metrics) | {s1['parse_failures']} | {s2['parse_failures']} |",
          f"| Unparsed windows (forecasts CSV) | {s1['csv_unparsed']} | {s2['csv_unparsed']} |",
          f"| Onset / calm / already-high windows | {s1['classes']['onset']} / {s1['classes']['calm']} / {s1['classes']['already_high']} "
          f"| {s2['classes']['onset']} / {s2['classes']['calm']} / {s2['classes']['already_high']} |"]
    L += ["", "## Error over all scored windows (from val_metrics.json)", "",
          "| Method | v1 MAE | v2 MAE | v1 RMSE | v2 RMSE |", "|---|---:|---:|---:|---:|"]
    for k in METHODS:
        e1, e2 = s1["errors"][k], s2["errors"][k]
        L.append(f"| {LABELS[k]} | {fmt(e1.get('mae'))} | {fmt(e2.get('mae'))} | {fmt(e1.get('rmse'))} | {fmt(e2.get('rmse'))} |")
    L += ["", "Baselines depend only on the windows, so they match between the runs when both scored the same windows "
          "(they can differ only through which windows parsed).", "",
          "## Peaks, pooled over all cells (from val_forecasts.csv)", "",
          "| Measure | v1 model | v2 model | v1 slope | v2 slope |", "|---|---:|---:|---:|---:|"]
    p = {(v, n): s["pooled"][n] for v, s in (("v1", s1), ("v2", s2)) for n in ("model", "slope")}
    L.append("| Onset recall | " + " | ".join(counts(p[k]["onset_flagged"], p[k]["onset_n"])
                                            for k in (("v1", "model"), ("v2", "model"), ("v1", "slope"), ("v2", "slope"))) + " |")
    L.append("| Calm false-alarm rate | " + " | ".join(counts(p[k]["calm_flagged"], p[k]["calm_n"])
                                                     for k in (("v1", "model"), ("v2", "model"), ("v1", "slope"), ("v2", "slope"))) + " |")
    L.append("| Mean (predicted peak - true peak) on onsets | " + " | ".join(fmt(p[k]["mean_onset_error"])
                                                     for k in (("v1", "model"), ("v2", "model"), ("v1", "slope"), ("v2", "slope"))) + " |")
    L += ["", "Validation windows start every 6 slots and are 18 slots long, so neighbouring windows overlap; "
          "counts are not independent samples.", "",
          "## Per cell", "",
          "| Cell | Square | v1 MAE | v2 MAE | v1 RMSE | v2 RMSE | v1 onset recall | v2 onset recall | v1 calm false alarms | v2 calm false alarms "
          "| v1 mean onset error | v2 mean onset error | slope MAE (v1 run) | slope onset recall (v1 run) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in s1["cells"]:
        e1 = s1["per_cell_errors"].get(str(cell), {})
        e2 = s2["per_cell_errors"].get(str(cell), {})
        c1, c2 = s1["per_cell"][cell], s2["per_cell"][cell]
        L.append(f"| {cell} | {e1.get('square_id', e2.get('square_id', 'not recorded'))} "
                 f"| {fmt((e1.get('model') or {}).get('mae'))} | {fmt((e2.get('model') or {}).get('mae'))} "
                 f"| {fmt((e1.get('model') or {}).get('rmse'))} | {fmt((e2.get('model') or {}).get('rmse'))} "
                 f"| {counts(c1['model']['onset_flagged'], c1['model']['onset_n'])} | {counts(c2['model']['onset_flagged'], c2['model']['onset_n'])} "
                 f"| {counts(c1['model']['calm_flagged'], c1['model']['calm_n'])} | {counts(c2['model']['calm_flagged'], c2['model']['calm_n'])} "
                 f"| {fmt(c1['model']['mean_onset_error'])} | {fmt(c2['model']['mean_onset_error'])} "
                 f"| {fmt((e1.get('slope') or {}).get('mae'))} | {counts(c1['slope']['onset_flagged'], c1['slope']['onset_n'])} |")
    L += ["", "Per-cell onset counts are small; the pooled figures above are the ones the rules use.", "",
          f"**DECISION: {'ADOPT' if adopt else 'DO NOT ADOPT'}**"]
    return "\n".join(L) + "\n", adopt


# ---------------------------------------------------------------- selftest

def run_selftest():
    """Made-up runs: one that passes all four rules, one that fails rule 3, one exactly at the 5-point limit,
    and two refusals (different windows, v2 file that is not v2)."""
    import tempfile

    def write_run(folder, variant, preds, parse_failures, mae):
        """preds: {(cell, start): (last, true_peak, model_peak or None, slope_peak)}"""
        os.makedirs(folder, exist_ok=True)
        with open(os.path.join(folder, "val_forecasts.csv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "start", "step", "true", "model", "persistence", "moving_avg", "parsed", "raw_output", "slope"])
            for (cell, start), (last, tp, mp, sp) in preds.items():
                for step in range(1, 7):
                    w.writerow([cell, start, step, tp if step == 2 else 10, "" if mp is None else (mp if step == 2 else 10),
                                last, last, mp is not None, "x", sp if step == 2 else 10])
        metrics = {"windows_used": len(preds), "parse_failures": parse_failures, "windows_scored": len(preds) - parse_failures,
                   "model": {"mae": mae, "rmse": mae + 1}, "persistence": {"mae": 9.0, "rmse": 10.0},
                   "moving_avg": {"mae": 8.0, "rmse": 9.0}, "slope": {"mae": 12.0, "rmse": 13.0},
                   "per_cell": {"0": {"square_id": 11, "model": {"mae": mae, "rmse": mae + 1}, "slope": {"mae": 12.0}},
                                "1": {"square_id": 22, "model": {"mae": mae, "rmse": mae + 1}, "slope": {"mae": 12.0}}}}
        if variant:
            metrics["variant"] = variant
        with open(os.path.join(folder, "val_metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f)

    def windows(onset_flags, calm_flags):
        """Two cells. onset_flags / calm_flags: list of predicted peaks for onset / calm windows, spread over the cells."""
        preds, start = {}, 0
        for i, mp in enumerate(onset_flags):
            preds[(str(i % 2), start)] = (50, 150, mp, 130 if i == 0 else 60); start += 6
        for i, mp in enumerate(calm_flags):
            preds[(str(i % 2), start)] = (50, 80, mp, 60); start += 6
        preds[("0", start)] = (150, 160, 140, 150)            # one already-high window, ignored by the rules
        return preds

    with tempfile.TemporaryDirectory() as tmp:
        # v1: 4 onsets, 1 flagged (25%); 20 calm, 2 false alarms (10%); 1 parse failure; MAE 15.
        v1_preds = windows([120, 50, 50, 50], [120, 120] + [50] * 18)
        v1_preds[("1", 999)] = (50, 80, None, 60)           # the parse failure (a calm window)
        write_run(os.path.join(tmp, "v1"), None, v1_preds, 1, 15.0)
        r1 = load_run(*find_files(os.path.join(tmp, "v1")))

        # ADOPT: MAE 14 < 15; recall 3/4 > 1/4; false alarms 3/20 = +5.00 points exactly (allowed); parse failures 1 <= 1.
        good = windows([120, 120, 120, 50], [120, 120, 120] + [50] * 17)
        good[("1", 999)] = (50, 80, None, 60)
        write_run(os.path.join(tmp, "v2a", "outputs"), "v2", good, 1, 14.0)
        text, adopt = build_report(r1, load_run(*find_files(os.path.join(tmp, "v2a"))))   # found in outputs/
        assert adopt and text.rstrip().endswith("**DECISION: ADOPT**"), text
        assert "| 2. Pooled onset recall (all cells) higher than v1's | 1/4 (25.0%) | 3/4 (75.0%) | yes |" in text
        assert "+5.00 points" in text and "| 4. Parse failures no more than v1's | 1 | 1 | yes |" in text
        assert "| Mean (predicted peak - true peak) on onsets | -82.50 | -47.50 | -72.50 | -72.50 |" in text   # v1 (-30-300)/4, v2 (-90-100)/4, slope (-20-270)/4

        # DO NOT ADOPT: same as above but 4 false alarms (20%, +10 points) -> rule 3 fails; everything else holds.
        bad = windows([120, 120, 120, 50], [120, 120, 120, 120] + [50] * 16)
        bad[("1", 999)] = (50, 80, None, 60)
        write_run(os.path.join(tmp, "v2b"), "v2", bad, 1, 14.0)
        text, adopt = build_report(r1, load_run(*find_files(os.path.join(tmp, "v2b"))))
        assert not adopt and text.rstrip().endswith("**DECISION: DO NOT ADOPT**")
        assert "+10.00 points" in text and "| NO |" in text

        # DO NOT ADOPT on MAE alone (equal is not lower) and on more parse failures.
        write_run(os.path.join(tmp, "v2c"), "v2", good, 2, 15.0)
        text, adopt = build_report(r1, load_run(*find_files(os.path.join(tmp, "v2c"))))
        assert not adopt and text.count("| NO |") == 2

        # Refusals: a v2 run that scored other windows, and a "v2" file that does not say v2.
        other = dict(good); other[("1", 12345)] = (50, 80, 50, 60)
        write_run(os.path.join(tmp, "v2d"), "v2", other, 1, 14.0)
        write_run(os.path.join(tmp, "v2e"), None, good, 1, 14.0)
        for folder in ("v2d", "v2e"):
            try:
                build_report(r1, load_run(*find_files(os.path.join(tmp, folder))))
                raise AssertionError(f"{folder} was not refused")
            except SystemExit:
                pass

    # Rule 3 edges, with rules 1, 2 and 4 held true so only rule 3 decides. v1: 10 false alarms on 100 calm windows (10%).
    def rule3_run(folder, variant, n_calm, n_alarms, mae):
        preds = windows([120, 50, 50, 50] if variant is None else [120, 120, 120, 50],
                        [120] * n_alarms + [50] * (n_calm - n_alarms))
        write_run(folder, variant, preds, 0, mae)
        return load_run(*find_files(folder))

    with tempfile.TemporaryDirectory() as tmp:
        base = rule3_run(os.path.join(tmp, "v1"), None, 100, 10, 15.0)
        for name, n_calm, n_alarms, rise_text, expect in [
                ("within", 100, 14, "+4.00 points", True),      # 14%: 4 points above
                ("exactly", 100, 15, "+5.00 points", True),     # 15%: exactly 5 points above, allowed
                ("over", 10000, 1501, "+5.01 points", False)]:  # 15.01%: 5.01 points above, refused
            # The over case needs 10000 calm windows so v2's rate can be 15.01%; v1 must score the same windows,
            # so it gets a matching 10000-window file with 1000 false alarms (still 10%).
            v1 = base if n_calm == 100 else rule3_run(os.path.join(tmp, "v1_big"), None, n_calm, n_calm // 10, 15.0)
            v2 = rule3_run(os.path.join(tmp, name), "v2", n_calm, n_alarms, 14.0)
            text, adopt = build_report(v1, v2)
            assert rise_text in text and adopt == expect, (name, text)
            rule3_row = next(line for line in text.splitlines() if line.startswith("| 3."))
            assert rule3_row.endswith("| yes |" if expect else "| NO |"), rule3_row

        # v2 parsed a window that v1 did not. Both files hold the same windows (cell, start), so they are NOT refused;
        # they are compared, each run counting only its own parsed windows: v1 scores one window fewer than v2.
        # (Behaviour recorded as it is today, not changed.)
        v1_preds = windows([120, 50, 50, 50], [120] * 10 + [50] * 90)
        unparsed_key = max(k for k in v1_preds if v1_preds[k][0] == 50 and v1_preds[k][1] == 80)   # a calm window
        v2_preds = windows([120, 120, 120, 50], [120] * 10 + [50] * 90)
        v1_preds[unparsed_key] = v1_preds[unparsed_key][:2] + (None,) + v1_preds[unparsed_key][3:]
        write_run(os.path.join(tmp, "v1_unparsed"), None, v1_preds, 1, 15.0)
        write_run(os.path.join(tmp, "v2_all_parsed"), "v2", v2_preds, 0, 14.0)
        r1 = load_run(*find_files(os.path.join(tmp, "v1_unparsed")))
        r2 = load_run(*find_files(os.path.join(tmp, "v2_all_parsed")))
        text, adopt = build_report(r1, r2)                      # no SystemExit: the window sets are identical
        s1, s2 = summarize(r1), summarize(r2)
        assert s1["windows_scored"] == len(v1_preds) - 1 and s2["windows_scored"] == len(v2_preds)
        assert (s1["pooled"]["model"]["calm_n"], s2["pooled"]["model"]["calm_n"]) == (99, 100)   # different denominators
        assert "| 3. Calm false-alarm rate" in text and "10/99 (10.1%) | 10/100 (10.0%) | yes |" in text
        assert "| 4. Parse failures no more than v1's | 1 | 0 | yes |" in text and adopt
    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description="Compare v1 and v2 validation runs (PLAN_V2.md adoption rule)")
    parser.add_argument("--v1", help="folder of the v1 run (val_metrics.json + val_forecasts.csv, directly or in outputs/)")
    parser.add_argument("--v2", help="folder of the v2 run")
    parser.add_argument("--v1-metrics"); parser.add_argument("--v1-forecasts")
    parser.add_argument("--v2-metrics"); parser.add_argument("--v2-forecasts")
    parser.add_argument("--out", default="compare_v1_v2_val.md", help="markdown report to write (default compare_v1_v2_val.md)")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return
    runs = []
    for v in ("v1", "v2"):
        folder, m, f = getattr(args, v), getattr(args, f"{v}_metrics"), getattr(args, f"{v}_forecasts")
        if folder:
            m, f = find_files(folder)
        if not (m and f):
            parser.error(f"give --{v} DIR, or both --{v}-metrics and --{v}-forecasts")
        runs.append(load_run(m, f))
    if os.path.exists(args.out):
        sys.exit(f"ERROR: {args.out} already exists; choose another --out so no report is overwritten.")
    text, adopt = build_report(*runs)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text)
    print(f"Wrote {args.out}")
    print("ADOPT" if adopt else "DO NOT ADOPT")


if __name__ == "__main__":
    main()
