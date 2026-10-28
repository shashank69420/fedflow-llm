"""
FedFlow-LLM — telecom_check.py
-------------------------------
Read-only look at the locked telecom cells. Reads only datasets/telecom_milan/cells.csv and
cells_meta.json, changes nothing, and writes datasets/telecom_milan/check_report.txt.

Covers: the experiment-cell slice values, the size of each upward crossing in that slice,
onset-window counts per cell at stride 3 and stride 6 (test block), mean scaled load by day type, and, for both
experiment cells, how often the oracle and reactive values are above 100 in the 36-step trace.

Standard library only.   Usage:  python3 training/telecom_check.py
"""
import csv
import json
import os
import statistics

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO, "datasets", "telecom_milan")
OUT_PATH = os.path.join(DATA_DIR, "check_report.txt")

THRESHOLD = 100.0
HISTORY, HORIZON = 12, 6
WINDOW = HISTORY + HORIZON
SLOTS_PER_DAY = 144
TRAIN_END, VAL_END, N_SLOTS = 1440, 1728, 2016   # same locked split as prepare_telecom.py
NON_WORKING_DAYS = [0, 1, 2, 8, 9]   # 1, 2, 3, 9, 10 Nov 2013: public holiday (Fri 1 Nov), Sat/Sun, Sat/Sun
WORKING_DAYS = [3, 4, 5, 6, 7]       # 4-8 Nov 2013, Monday to Friday
MIN_ONSETS = 10                      # fewer onset windows than this is too few to report onset recall


def day_mean(series, days):
    """Mean scaled value over every slot of the given training days (day index = slot // 144)."""
    return statistics.fmean(series[d * SLOTS_PER_DAY + k] for d in days for k in range(SLOTS_PER_DAY))


def onset_sizes(series, stride):
    """For test windows (starts 1728..1998 every `stride` slots): the 'max future - 100' of each onset window.
    Onset = last history value <= 100 and max of the 6 future values > 100. Also returns the window count."""
    starts = range(VAL_END, N_SLOTS - WINDOW + 1, stride)
    sizes = []
    for a in starts:
        last_history = series[a + HISTORY - 1]
        peak = max(series[a + HISTORY : a + WINDOW])
        if last_history <= THRESHOLD < peak:
            sizes.append(peak - THRESHOLD)
    return sizes, len(starts)


def trace_values(series, start):
    """Per step of the 36-step slice starting at `start`: (oracle value, reactive value), TRUE data only.
    Same rule as experiments/run_experiment.py: reactive = true value of the previous step; oracle = peak of the
    6 true values of the latest stride-3 test window whose history ends at or before the previous step."""
    out = []
    for t in range(start, start + 36):
        w = max(a for a in range(VAL_END, N_SLOTS - WINDOW + 1, 3) if a + HISTORY - 1 <= t - 1)   # latest usable window
        out.append((max(series[w + HISTORY : w + WINDOW]), series[t - 1]))
    return out


def median_text(values):
    return f"{statistics.median(values):.2f}" if values else "n/a"


def main():
    with open(os.path.join(DATA_DIR, "cells_meta.json"), encoding="utf-8") as f:
        meta = json.load(f)
    with open(os.path.join(DATA_DIR, "cells.csv"), newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    squares = [str(s) for s in meta["squares"]]
    series = {s: [float(r[s]) for r in rows] for s in squares}
    assert [int(r["timestep"]) for r in rows] == list(range(N_SLOTS)), "cells.csv must hold timesteps 0..2015 in order"

    exp = str(meta["experiment_cell"])
    start = meta["best_slice_start"]
    x = series[exp]
    out = [
        "Telecom check report (read-only; computed from datasets/telecom_milan/cells.csv and cells_meta.json)",
        f"Experiment cell: square {exp}, slice start {start}, crossings in cells_meta.json: {meta['best_slice_crossings']}",
        "",
        f"1. SCALED VALUES OF SQUARE {exp}, SLOTS {start}-{start + 35}",
    ]
    out += [f"slot {i}: {x[i]:.2f}" for i in range(start, start + 36)]

    out += ["", "2. UPWARD CROSSINGS IN THAT SLICE (value before <= 100, value after > 100)",
            "'next 6' = slots i..i+5, the 6 forecast steps that follow the 'before' value at slot i-1;",
            "the second reading (slots i+1..i+6) is shown too.",
            "slot  before  after  max(i..i+5)-100  max(i+1..i+6)-100"]
    sizes = []
    for i in range(start + 1, start + 36):
        if x[i - 1] <= THRESHOLD < x[i]:
            size = max(x[i : i + 6]) - THRESHOLD
            sizes.append((i, size))
            out.append(f"{i}  {x[i - 1]:.2f}  {x[i]:.2f}  {size:.2f}  {max(x[i + 1 : i + 7]) - THRESHOLD:.2f}")
    biggest = max(sizes, key=lambda p: p[1])
    others = [s for i, s in sizes if i != biggest[0]]
    out.append(f"Crossings found: {len(sizes)}")
    out.append(f"Largest crossing size: {biggest[1]:.2f} at slot {biggest[0]}; "
               f"largest size among the other {len(others)} crossings: {max(others):.2f}")
    peak_slot = max(range(start, start + 36), key=lambda i: x[i])
    out.append(f"Highest value in the slice: {x[peak_slot]:.2f} at slot {peak_slot}")

    out += ["", f"3. ONSET WINDOWS IN THE TEST BLOCK (slots {VAL_END}-{N_SLOTS - 1}); 12 history + 6 future;",
            f"onset = last history value <= {THRESHOLD:g} and max of the 6 future values > {THRESHOLD:g}",
            f"A count under {MIN_ONSETS} is flagged: onset recall would be too unreliable to report."]
    pooled6 = []
    for stride in (3, 6):
        out += ["", f"Stride {stride}:", "square  windows  onset windows  median(max future - 100)  note"]
        for s in squares:
            sizes_s, n_windows = onset_sizes(series[s], stride)
            note = f"FEWER THAN {MIN_ONSETS}" if len(sizes_s) < MIN_ONSETS else ""
            out.append(f"{s}  {n_windows}  {len(sizes_s)}  {median_text(sizes_s)}  {note}".rstrip())
            if stride == 6:
                pooled6 += sizes_s
        if stride == 6:
            out.append(f"Pooled over all {len(squares)} cells at stride 6: {len(pooled6)} onset windows, "
                       f"median(max future - 100) {median_text(pooled6)}")

    out += ["", "4. MEAN SCALED VALUE BY DAY TYPE",
            "non-working training days: day indexes 0,1,2,8,9 = 1, 2, 3, 9, 10 Nov 2013 (holiday, Sat/Sun, Sat/Sun)",
            "working training days: day indexes 3-7 = 4-8 Nov 2013",
            "square  non-working  working  validation  test"]
    for s in squares:
        out.append(f"{s}  {day_mean(series[s], NON_WORKING_DAYS):.2f}  {day_mean(series[s], WORKING_DAYS):.2f}  "
                   f"{statistics.fmean(series[s][TRAIN_END:VAL_END]):.2f}  {statistics.fmean(series[s][VAL_END:]):.2f}")

    out += ["", f"5. SHARE OF SLOTS ABOVE {THRESHOLD:g}", "square  all 2016 slots  validation  test"]
    for s in squares:
        v = series[s]
        share = lambda block: sum(1 for value in block if value > THRESHOLD) / len(block)
        out.append(f"{s}  {share(v):.4f}  {share(v[TRAIN_END:VAL_END]):.4f}  {share(v[VAL_END:]):.4f}")

    out += ["", f"6. ORACLE AND REACTIVE VALUES IN THE 36-STEP TRACE OF EACH EXPERIMENT CELL (true data only)",
            "oracle = peak of the next 6 true values (latest stride-3 window ending before the step); reactive = previous step's true value.",
            "square  role  slice start  oracle>100  reactive>100  oracle<=100  oracle upward crossing"]
    for e in meta["experiment_cells"]:
        values = trace_values(series[str(e["square"])], e["best_slice_start"])
        oracle = [o for o, _ in values]
        reactive = [r for _, r in values]
        first = next((i for i in range(1, 36) if oracle[i - 1] <= THRESHOLD < oracle[i]), None)
        crossing = "none" if first is None else f"first at step {first}, --quick would run {min(first + 1 + 3, 36)} steps"
        out.append(f"{e['square']}  {e['role'].split(' ')[0]}  {e['best_slice_start']}  {sum(o > THRESHOLD for o in oracle)} of 36  "
                   f"{sum(r > THRESHOLD for r in reactive)} of 36  {sum(o <= THRESHOLD for o in oracle)} of 36  {crossing}")

    text = "\n".join(out) + "\n"
    print(text)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
