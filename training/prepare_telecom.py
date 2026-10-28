"""
FedFlow-LLM — prepare_telecom.py
---------------------------------
Turns the raw Telecom Italia (Milan) day files into 6 scaled "cells" of internet
load, 2016 ten-minute slots each. Only the internet column is used.

Reads  (never copied into the repo):  ~/fedflow_data/telecom_raw/sms-call-internet-mi-2013-11-01..14.txt
Caches (outside the repo):            ~/fedflow_data/telecom_cache/day_01..14.npz
Writes (relative to the repo root):
  datasets/telecom_milan/cells.csv        timestep + 6 scaled columns named by square ID
  datasets/telecom_milan/cells_meta.json  chosen squares, raw means, k, experiment cell, slice
  datasets/telecom_milan/prep_report.txt  the numbers behind every choice

Usage:  python3 training/prepare_telecom.py            full run (days already cached are skipped)
        python3 training/prepare_telecom.py --selftest   check the selection logic on made-up data

All rules below are LOCKED. Do not change them after seeing a model result.
"""
import argparse
import json
import multiprocessing
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.expanduser("~/fedflow_data/telecom_raw")
CACHE_DIR = os.path.expanduser("~/fedflow_data/telecom_cache")
OUT_DIR = os.path.join(REPO, "datasets", "telecom_milan")

N_DAYS = 14
SLOTS_PER_DAY = 144                  # 10-minute slots per day
N_SLOTS = N_DAYS * SLOTS_PER_DAY     # 2016
N_SQUARES = 10000                    # square IDs run 1..10000
BASE_MS = 1383260400000              # 1 Nov 2013 00:00 Milan time (= 31 Oct 23:00 UTC)
SLOT_MS = 600000                     # 10 minutes in milliseconds
DAY_MS = SLOTS_PER_DAY * SLOT_MS
CHUNK_ROWS = 2_000_000               # rows read at a time, so a whole file is never in memory

TRAIN_END, VAL_END = 1440, 1728      # training = slots 0-1439, validation 1440-1727, test 1728-2015
MIN_NONBLANK_PCT = 95                # a square needs >= 95% non-blank slots over all 2016
MIN_PASSING_SQUARES = 100            # abort if fewer squares pass the filter
TARGET_PERCENTILES = [40, 46, 52, 58, 64, 70]
TARGET_MEDIAN = 90.0                 # the median chosen cell's mean training load is scaled to 90
THRESHOLD = 100.0                    # 100 = the 10 Mbps primary link, as in the synthetic data
TRACE_LEN = 36                       # slice length in steps
SLICE_START_MIN, SLICE_START_MAX = 1740, 1980   # inclusive; 1740 = first step with a model forecast
MIN_CROSSINGS = 3
CACHE_VERSION = 1


# ---------------------------------------------------------------- stage 1: one day at a time

def raw_path(day):
    """Path of the raw file for day index 0..13."""
    return os.path.join(RAW_DIR, f"sms-call-internet-mi-2013-11-{day + 1:02d}.txt")


def cache_path(day):
    return os.path.join(CACHE_DIR, f"day_{day + 1:02d}.npz")


def process_day(day):
    """Stream one day file in chunks and return/cache its (10000 x 144) sum and non-blank arrays."""
    src = raw_path(day)
    size = os.path.getsize(src)
    dest = cache_path(day)

    # Reuse the cache only if it was built from a file of the same size and the same code version.
    if os.path.exists(dest):
        with np.load(dest) as z:
            if int(z["version"]) == CACHE_VERSION and int(z["src_size"]) == size:
                return {"day": day, "rows": int(z["rows"]), "numeric_rows": int(z["numeric_rows"]),
                        "min_value": float(z["min_value"]), "cached": True}

    cells = N_SQUARES * SLOTS_PER_DAY
    sums = np.zeros(cells, dtype=np.float64)       # running internet sum per (square, slot-in-day)
    counts = np.zeros(cells, dtype=np.int64)       # how many numeric internet values hit each cell
    rows = numeric_rows = 0
    min_value = np.inf
    reader = pd.read_csv(
        src, sep="\t", header=None, usecols=[0, 1, 7], names=["square", "ts", "internet"],
        dtype={"square": np.int64, "ts": np.int64, "internet": np.float64},  # blank internet -> NaN
        chunksize=CHUNK_ROWS,
    )
    for chunk in reader:
        square = chunk["square"].to_numpy()
        rel = chunk["ts"].to_numpy() - BASE_MS - day * DAY_MS   # ms since this file's own midnight
        value = chunk["internet"].to_numpy()
        # Locked rule: abort if a row is outside this file's 144 slots (or not on the 10-minute grid).
        if (rel < 0).any() or (rel >= DAY_MS).any() or (rel % SLOT_MS != 0).any():
            raise SystemExit(f"ABORT: day {day + 1:02d} has a row outside its own 144 slots or off the 10-minute grid.")
        if (square < 1).any() or (square > N_SQUARES).any():
            raise SystemExit(f"ABORT: day {day + 1:02d} has a square ID outside 1..{N_SQUARES}.")
        keep = ~np.isnan(value)                        # blank internet is not a number: it adds 0 and marks nothing
        if not np.isfinite(value[keep]).all():
            raise SystemExit(f"ABORT: day {day + 1:02d} has an infinite internet value.")
        index = (square[keep] - 1) * SLOTS_PER_DAY + rel[keep] // SLOT_MS   # flat (square, slot) position
        sums += np.bincount(index, weights=value[keep], minlength=cells)    # add values into their cells
        counts += np.bincount(index, minlength=cells)                       # count numeric values per cell
        rows += len(chunk)
        numeric_rows += int(keep.sum())
        if keep.any():
            min_value = min(min_value, float(value[keep].min()))

    tmp = dest[:-4] + ".tmp.npz"                       # write to a temp name first so a timeout never leaves a half file
    np.savez(tmp, sums=sums.reshape(N_SQUARES, SLOTS_PER_DAY), nonblank=(counts > 0).reshape(N_SQUARES, SLOTS_PER_DAY),
             rows=rows, numeric_rows=numeric_rows, min_value=min_value, src_size=size, version=CACHE_VERSION)
    os.replace(tmp, dest)                              # atomic rename: the cache file is either complete or absent
    return {"day": day, "rows": rows, "numeric_rows": numeric_rows, "min_value": min_value, "cached": False}


def build_days(workers):
    """Process every day (skipping cached ones) and return the per-day summaries in day order."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    with multiprocessing.Pool(processes=workers) as pool:   # days are independent, so several run at once
        summaries = []
        for s in pool.imap_unordered(process_day, range(N_DAYS)):
            print(f"day {s['day'] + 1:02d} done: rows={s['rows']} numeric_internet_rows={s['numeric_rows']} "
                  f"({'from cache' if s['cached'] else 'computed'})", flush=True)
            summaries.append(s)
    return sorted(summaries, key=lambda s: s["day"])


def load_all():
    """Stack the 14 cached days into (10000 x 2016) sum and non-blank arrays."""
    sums, nonblank = [], []
    for day in range(N_DAYS):
        with np.load(cache_path(day)) as z:
            sums.append(z["sums"])
            nonblank.append(z["nonblank"])
    return np.concatenate(sums, axis=1), np.concatenate(nonblank, axis=1)


# ---------------------------------------------------------------- stage 2: pure selection logic

def percentile_ranks(means):
    """Percentile of each value = 100 * (number of strictly lower values) / (N - 1)."""
    lower = np.searchsorted(np.sort(means), means, side="left")
    return 100.0 * lower / (len(means) - 1)


def select_cells(ids, means, targets):
    """For each target percentile (in order) take the nearest unused square; lower ID on ties.
    ids must be sorted ascending so that argmin's first hit is the lowest ID."""
    pct = percentile_ranks(means)
    used = np.zeros(len(ids), dtype=bool)
    chosen = []
    for target in targets:
        distance = np.abs(pct - target)
        distance[used] = np.inf                              # squares already taken are not eligible
        best = int(np.argmin(distance + 0.0))                # first minimum = lowest ID among exact ties
        used[best] = True
        chosen.append(best)
    return chosen, pct


def upward_crossings(values):
    """Count i where values[i-1] <= 100 and values[i] > 100."""
    return sum(1 for a, b in zip(values, values[1:]) if a <= THRESHOLD < b)


def best_slice(series):
    """(start, crossings) of the 36-step slice with the most crossings; earliest start on ties.
    Uses the TRUE (scaled) series only, never model output."""
    best_start, best_count = None, -1
    for s in range(SLICE_START_MIN, SLICE_START_MAX + 1):
        count = upward_crossings(series[s:s + TRACE_LEN].tolist())
        if count > best_count:                               # strict ">" keeps the earliest start on ties
            best_start, best_count = s, count
    return best_start, best_count


def block_stats(series):
    """Mean, max and fraction of slots above 100 for the training, validation and test blocks."""
    out = {}
    for name, (a, b) in {"train": (0, TRAIN_END), "val": (TRAIN_END, VAL_END), "test": (VAL_END, N_SLOTS)}.items():
        block = series[a:b]
        out[name] = {"mean": float(block.mean()), "max": float(block.max()), "frac_above_100": float((block > THRESHOLD).mean())}
    return out


def run_selftest():
    # Crossings: 100 -> 101 counts (<=100 then >100); 101 -> 102 does not; 99 -> 100 does not.
    assert upward_crossings([99, 100, 101, 102, 99, 101]) == 2
    assert upward_crossings([99.0, 100.0]) == 0
    # Percentile of 5 distinct values is 0, 25, 50, 75, 100.
    assert list(percentile_ranks(np.array([5.0, 1.0, 3.0, 2.0, 4.0]))) == [100.0, 0.0, 50.0, 25.0, 75.0]
    # Nearest-percentile pick: 101 squares, means = ID, so percentile = ID - 1.
    ids = np.arange(1, 102)
    chosen, _ = select_cells(ids, ids.astype(float), [40, 46, 40])
    assert [int(ids[c]) for c in chosen] == [41, 47, 40]     # third target is taken -> next nearest, lower ID (40 beats 42)
    print("SELFTEST OK")


# ---------------------------------------------------------------- stage 3: report and files

def main():
    parser = argparse.ArgumentParser(description="Prepare the Telecom Italia (Milan) cells")
    parser.add_argument("--selftest", action="store_true", help="test the selection logic on made-up data")
    parser.add_argument("--workers", type=int, default=4, help="days processed in parallel (default 4)")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    summaries = build_days(args.workers)
    sums, nonblank = load_all()
    report = ["Telecom Italia (Milan) prep report", "Source: ~/fedflow_data/telecom_raw (14 day files, internet column only)", ""]

    total_rows = sum(s["rows"] for s in summaries)
    report.append(f"Rows read: {total_rows}; rows with a numeric internet value: {sum(s['numeric_rows'] for s in summaries)}")
    report.append(f"Smallest numeric internet value seen: {min(s['min_value'] for s in summaries)}")
    report.append("")

    # Cell filter: non-blank slots * 100 >= 95 * 2016 (integer maths avoids float rounding at the edge).
    nonblank_count = nonblank.sum(axis=1)
    passing = np.where(nonblank_count * 100 >= MIN_NONBLANK_PCT * N_SLOTS)[0]    # array positions; square ID = position + 1
    report.append(f"FILTER: squares with >= {MIN_NONBLANK_PCT}% non-blank slots over all {N_SLOTS}: {len(passing)} of {N_SQUARES}")
    if len(passing) < MIN_PASSING_SQUARES:
        report.append(f"ABORT: fewer than {MIN_PASSING_SQUARES} squares passed. Nothing else was computed or written.")
        finish(report)
        sys.exit(1)

    ids = passing + 1
    raw_means = sums[passing, :TRAIN_END].mean(axis=1)                         # training slots only, blanks count as 0
    chosen, pct = select_cells(ids, raw_means, TARGET_PERCENTILES)
    squares = [int(ids[c]) for c in chosen]
    means = [float(raw_means[c]) for c in chosen]
    percentiles = [float(pct[c]) for c in chosen]
    k = TARGET_MEDIAN / float(np.median(means))                                # median of 6 = average of the middle two
    scaled = np.round(sums[np.array(squares) - 1, :] * k, 2)                   # shape (6, 2016), rounded to 2 decimals
    blank_share = 1.0 - nonblank[np.array(squares) - 1, :].mean(axis=1)        # share of slots with no numeric value (filled with 0)

    report.append("")
    report.append("CHOSEN CELLS (rank by raw training mean among passing squares; percentile = 100 * lower / (N-1))")
    report.append(f"{'cell':<5}{'square':<8}{'target pct':<12}{'actual pct':<12}{'raw train mean':<18}")
    for i in range(6):
        report.append(f"{i:<5}{squares[i]:<8}{TARGET_PERCENTILES[i]:<12}{percentiles[i]:<12.4f}{means[i]:<18.6f}")
    report.append(f"Median of the 6 raw training means: {float(np.median(means)):.6f}")
    report.append(f"k = {TARGET_MEDIAN:g} / median = {k:.10f}")
    report.append("")
    report.append("SCALED CELLS (values rounded to 2 decimals; unitless load, 100 = the 10 Mbps primary link)")
    report.append(f"{'square':<8}{'block':<7}{'mean':<12}{'max':<12}{'frac > 100':<12}")
    stats = [block_stats(scaled[i]) for i in range(6)]
    for i in range(6):
        for block in ("train", "val", "test"):
            s = stats[i][block]
            report.append(f"{squares[i]:<8}{block:<7}{s['mean']:<12.4f}{s['max']:<12.2f}{s['frac_above_100']:<12.4f}")
    report.append("")
    report.append("BLANK-FILLED SLOTS (no numeric internet value, filled with 0), share of the slots in each span")
    report.append(f"{'square':<8}{'all 2016':<12}{'train':<12}{'val':<12}{'test':<12}")
    for i in range(6):
        row = nonblank[squares[i] - 1]
        spans = [1 - row.mean(), 1 - row[:TRAIN_END].mean(), 1 - row[TRAIN_END:VAL_END].mean(), 1 - row[VAL_END:].mean()]
        report.append(f"{squares[i]:<8}" + "".join(f"{v:<12.4f}" for v in spans))

    # Experiment cell: best slice per cell on TRUE scaled traffic; most crossings, then earliest start, then lower square ID.
    report.append("")
    report.append(f"CROSSINGS: best {TRACE_LEN}-step slice per cell, slice starts {SLICE_START_MIN}-{SLICE_START_MAX} inclusive, "
                  "true scaled traffic only")
    best = []
    for i in range(6):
        start, count = best_slice(scaled[i])
        best.append((start, count))
        report.append(f"square {squares[i]}: best slice start {start}, upward crossings of {THRESHOLD:g}: {count}")
    order = sorted(range(6), key=lambda i: (-best[i][1], best[i][0], squares[i]))
    exp = order[0]
    report.append(f"Experiment cell: square {squares[exp]} (cell {exp}), slice start {best[exp][0]}, crossings {best[exp][1]}")
    if best[exp][1] < MIN_CROSSINGS:
        report.append(f"ABORT: the best slice has only {best[exp][1]} crossings (need {MIN_CROSSINGS}). cells.csv and cells_meta.json were NOT written.")
        finish(report)
        sys.exit(1)

    frame = pd.DataFrame(scaled.T, columns=[str(s) for s in squares])
    frame.insert(0, "timestep", np.arange(N_SLOTS))
    frame.to_csv(os.path.join(OUT_DIR, "cells.csv"), index=False)
    meta = {
        "squares": squares,
        "raw_train_means": means,
        "percentiles": percentiles,
        "target_percentiles": TARGET_PERCENTILES,
        "k": k,
        "squares_passing_filter": int(len(passing)),
        "filter": f">= {MIN_NONBLANK_PCT}% non-blank slots over all {N_SLOTS}",
        "split": {"train": [0, TRAIN_END - 1], "val": [TRAIN_END, VAL_END - 1], "test": [VAL_END, N_SLOTS - 1]},
        "slice_rule": {
            "trace_len": TRACE_LEN, "start_min": SLICE_START_MIN, "start_max": SLICE_START_MAX,
            "threshold": THRESHOLD, "min_crossings": MIN_CROSSINGS,
            "chosen_from": "true scaled traffic only, never from model outputs",
        },
        "best_slice_per_cell": [{"square": squares[i], "start": best[i][0], "crossings": best[i][1]} for i in range(6)],
        "experiment_cell": squares[exp],
        "experiment_cell_index": exp,
        "best_slice_start": best[exp][0],
        "best_slice_crossings": best[exp][1],
    }
    with open(os.path.join(OUT_DIR, "cells_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    report.append("")
    report.append("Wrote datasets/telecom_milan/cells.csv and cells_meta.json")
    finish(report)


def finish(report):
    """Print the report and save it as prep_report.txt."""
    text = "\n".join(report) + "\n"
    print(text)
    with open(os.path.join(OUT_DIR, "prep_report.txt"), "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    main()
