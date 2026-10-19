"""
FedFlow-LLM — experiments/summarize.py
---------------------------------------
Reads only the raw files written by run_experiment.py and produces:
  summary.csv   one row per mode and rep
  summary.png   loss % and p95 ping RTT per mode (mean of reps, min-max error bars)

Usage (no sudo):
  python3 experiments/summarize.py                          results/experiment/
  python3 experiments/summarize.py --dir results/experiment/quick
  python3 experiments/summarize.py --selftest               made-up raw text, temp folder

Nothing is estimated: a step without an iperf Server Report is counted as
"no report", and anything that can't be computed is written as "not computed".
"""
import argparse
import csv
import glob
import os
import re
import sys
import tempfile
from datetime import datetime

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(REPO, "results", "experiment")
MODES = ["static", "reactive", "model", "oracle", "alt"]
THRESHOLD = 100.0
MATCH_WINDOW_S = 5.0   # a FLIP must land within one step (5 s) of the write that caused it
NOT_COMPUTED = "not computed"

STEP_MARK = re.compile(r"^### step (\d+)", re.M)
# iperf 2 Server Report data line ends with "lost/total (pct%)", e.g. "0.021 ms 12/4253 (0.28%)".
LOSS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+\(([-\d.eE+]+|nan)%\)")
RTT_RE = re.compile(r"time=([\d.]+) ms")
FLIP_RE = re.compile(r"\[FedFlow\] (\d\d):(\d\d):(\d\d)\.(\d{3}) FLIP (\w+) -> (\w+)")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------- parsers

def parse_iperf(text):
    """Returns (lost_sum, total_sum, steps_with_report, steps_without_report)."""
    lost = total = with_report = without = 0
    marks = list(STEP_MARK.finditer(text))
    for k, mark in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        block = text[mark.start():end]
        report_at = block.find("Server Report")
        match = LOSS_RE.search(block, report_at) if report_at >= 0 else None
        if match is None:
            without += 1
            continue
        lost += int(match.group(1))
        total += int(match.group(2))
        with_report += 1
    return lost, total, with_report, without


def parse_ping(text):
    """All RTTs in ms from reply lines."""
    return [float(x) for x in RTT_RE.findall(text)]


def parse_flips(text):
    """[(seconds_of_day, old_path, new_path)] from POX FLIP lines."""
    flips = []
    for m in FLIP_RE.finditer(text):
        h, mi, s, ms = (int(m.group(i)) for i in range(1, 5))
        flips.append((h * 3600 + mi * 60 + s + ms / 1000.0, m.group(5), m.group(6)))
    return flips


def parse_steps(path):
    """[(value_written, seconds_of_day_written)] in step order."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["time_written"], "%Y-%m-%d %H:%M:%S.%f")
            rows.append((float(r["value_written"]),
                         t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6))
    return rows


def path_for(value):
    return "alternate" if value > THRESHOLD else "primary"


def match_flip_delays(steps, flips):
    """Delay (s) from a step write to the FLIP it caused, only where the match is unambiguous:
    the latest write at or before the FLIP is a write that CHANGES the path to the FLIP's new
    path, the FLIP is within MATCH_WINDOW_S of it, and no other FLIP is matched to that write.
    Times are time-of-day only, so differences are taken modulo 24 h."""
    if not steps:
        return []
    # Path in force before step 0 = step 0's own value (run_experiment writes it before POX starts).
    changes = []  # (write_time, new_path) for writes that change the path
    previous = path_for(steps[0][0])
    for value, t in steps:
        p = path_for(value)
        if p != previous:
            changes.append((t, p))
        previous = p
    write_times = [t for _, t in steps]

    candidates = {}  # change index -> list of delays
    for flip_t, _, new_path in flips:
        # Latest write at or before the flip (modulo 24 h, within MATCH_WINDOW_S).
        gaps = [((flip_t - wt) % 86400, wt) for wt in write_times]
        gaps = [(g, wt) for g, wt in gaps if g <= MATCH_WINDOW_S]
        if not gaps:
            continue
        gap, wt = min(gaps)
        for ci, (ct, cp) in enumerate(changes):
            if ct == wt and cp == new_path:
                candidates.setdefault(ci, []).append(gap)
    return [d[0] for d in candidates.values() if len(d) == 1]


# ---------------------------------------------------------------- summary

def summarize_run(prefix):
    steps_path = prefix + "_steps.csv"
    row = {}
    lost, total, with_report, without = parse_iperf(read(prefix + "_iperf.txt")) \
        if os.path.exists(prefix + "_iperf.txt") else (0, 0, 0, 0)
    row["udp_lost"] = lost
    row["udp_total"] = total
    row["udp_loss_pct"] = round(100.0 * lost / total, 4) if total > 0 else NOT_COMPUTED
    row["steps_with_report"] = with_report
    row["steps_no_report"] = without

    rtts = parse_ping(read(prefix + "_ping.txt")) if os.path.exists(prefix + "_ping.txt") else []
    row["ping_replies"] = len(rtts)
    row["ping_mean_ms"] = round(float(np.mean(rtts)), 4) if rtts else NOT_COMPUTED
    row["ping_p95_ms"] = round(float(np.percentile(rtts, 95)), 4) if rtts else NOT_COMPUTED

    flips = parse_flips(read(prefix + "_pox.txt")) if os.path.exists(prefix + "_pox.txt") else []
    row["flip_lines"] = len(flips)

    steps = parse_steps(steps_path)
    row["steps"] = len(steps)
    row["frac_written_above_100"] = round(sum(v > THRESHOLD for v, _ in steps) / len(steps), 4) \
        if steps else NOT_COMPUTED

    delays = match_flip_delays(steps, flips)
    row["flips_matched"] = len(delays)
    row["median_write_to_flip_s"] = round(float(np.median(delays)), 3) if delays else NOT_COMPUTED
    return row


def find_runs(folder):
    """[(mode, rep, prefix)] from raw_<mode>_<rep>_steps.csv files, in MODES order."""
    runs = []
    for path in glob.glob(os.path.join(folder, "raw_*_steps.csv")):
        m = re.match(r"raw_(\w+?)_(\d+)_steps\.csv$", os.path.basename(path))
        if m:
            runs.append((m.group(1), int(m.group(2)), path[: -len("_steps.csv")]))
    order = {m: i for i, m in enumerate(MODES)}
    return sorted(runs, key=lambda r: (order.get(r[0], len(MODES)), r[0], r[1]))


def plot(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    modes += sorted({r["mode"] for r in rows} - set(modes))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in [(axes[0], "udp_loss_pct", "UDP loss (%)"),
                           (axes[1], "ping_p95_ms", "Ping p95 RTT (ms)")]:
        means, lows, highs, names = [], [], [], []
        for m in modes:
            vals = [r[key] for r in rows if r["mode"] == m and r[key] != NOT_COMPUTED]
            if not vals:
                continue  # nothing measured for this mode: leave the bar out rather than draw 0
            mean = float(np.mean(vals))
            means.append(mean)
            lows.append(mean - min(vals))
            highs.append(max(vals) - mean)
            names.append(f"{m}\n(n={len(vals)})")
        ax.bar(names, means, yerr=[lows, highs], capsize=4, color="#4C78A8")
        ax.set_ylabel(label)
        ax.set_title(f"{label}: mean of reps, min-max bars")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


FIELDS = ["mode", "rep", "steps", "udp_lost", "udp_total", "udp_loss_pct", "steps_with_report",
          "steps_no_report", "ping_replies", "ping_mean_ms", "ping_p95_ms", "flip_lines",
          "frac_written_above_100", "flips_matched", "median_write_to_flip_s"]


def summarize(folder, make_plot=True):
    runs = find_runs(folder)
    if not runs:
        sys.exit(f"ERROR: no raw_*_steps.csv files in {folder}")
    rows = []
    for mode, rep, prefix in runs:
        row = {"mode": mode, "rep": rep}
        row.update(summarize_run(prefix))
        rows.append(row)
    csv_path = os.path.join(folder, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} rows)")
    if make_plot:
        png_path = os.path.join(folder, "summary.png")
        plot(rows, png_path)
        print(f"Wrote {png_path}")
    return rows


# ---------------------------------------------------------------- selftest

def run_selftest():
    iperf_text = """### step 0 timestep 1452 rate 6.895M written 0.00
[  1] local 10.0.0.2 port 40000 connected with 10.0.0.1 port 5001
[  1] Sent 2932 datagrams
[  1] Server Report:
[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams
[  1] 0.0000-5.0012 sec  4.10 MBytes  6.88 Mbits/sec   0.020 ms 0/2932 (0%)
### step 1 timestep 1453 rate 15.000M written 150.00
[  1] Sent 6378 datagrams
[  1] Server Report:
[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams
[  1] 0.0000-5.0010 sec  5.96 MBytes  10.0 Mbits/sec   0.500 ms 2128/6378 (33%)
### step 2 timestep 1454 rate 15.000M written 150.00
[  1] Sent 6378 datagrams
[  1] WARNING: did not receive ack of last datagram after 10 tries.
### step 3 timestep 1455 rate 7.000M written 50.00
[  1] Server Report:
[  1] 0.0000-5.0000 sec  4.17 MBytes  7.00 Mbits/sec   0.010 ms 10/2990 (0.33%)
"""
    ping_text = "\n".join(
        [f"[1700000000.{i:06d}] 64 bytes from 10.0.0.2: icmp_seq={i} ttl=64 time={v} ms"
         for i, v in enumerate([1.0, 2.0, 3.0, 4.0, 100.0], start=1)]
        + ["", "--- 10.0.0.2 ping statistics ---",
           "6 packets transmitted, 5 received, 16.6667% packet loss, time 1000ms",
           "rtt min/avg/max/mdev = 1.000/22.000/100.000/39.000 ms"])
    pox_text = """INFO:reroute_trigger:[FedFlow] Starting on primary path (forecast 0.0, threshold 100.0)
INFO:openflow.of_01:[00-00-00-00-00-01 4] connected
INFO:reroute_trigger:[FedFlow] 10:00:05.600 FLIP primary -> alternate (forecast 150.0)
INFO:reroute_trigger:[FedFlow] Forecast 150.0 > 100.0 on switch 1 — using alt path, port 3
INFO:reroute_trigger:[FedFlow] 10:00:15.900 FLIP alternate -> primary (forecast 50.0)
"""
    steps_text = """step,timestep,true_value,value_written,time_written
0,1452,68.95,0.00,2026-10-20 10:00:00.000
1,1453,150.00,150.00,2026-10-20 10:00:05.000
2,1454,150.00,150.00,2026-10-20 10:00:10.000
3,1455,70.00,50.00,2026-10-20 10:00:15.000
"""
    with tempfile.TemporaryDirectory() as tmp:
        prefix = os.path.join(tmp, "raw_model_1")
        for suffix, text in [("_iperf.txt", iperf_text), ("_ping.txt", ping_text),
                             ("_pox.txt", pox_text), ("_steps.csv", steps_text)]:
            with open(prefix + suffix, "w", encoding="utf-8") as f:
                f.write(text)
        # A second run whose iperf has no reports at all and whose ping and FLIPs are empty.
        prefix2 = os.path.join(tmp, "raw_static_1")
        for suffix, text in [("_iperf.txt", "### step 0 timestep 1452 rate 6.895M written 0.00\nno output\n"),
                             ("_ping.txt", "PING 10.0.0.2\n"), ("_pox.txt", "nothing\n"),
                             ("_steps.csv", "step,timestep,true_value,value_written,time_written\n"
                                            "0,1452,68.95,0.00,2026-10-20 10:00:00.000\n")]:
            with open(prefix2 + suffix, "w", encoding="utf-8") as f:
                f.write(text)

        rows = summarize(tmp, make_plot=True)
        assert os.path.exists(os.path.join(tmp, "summary.png"))
        by_mode = {r["mode"]: r for r in rows}
        assert [r["mode"] for r in rows] == ["static", "model"], "rows should follow MODES order"

        r = by_mode["model"]
        # Steps 0, 1, 3 have reports: lost 0 + 2128 + 10, total 2932 + 6378 + 2990.
        assert (r["udp_lost"], r["udp_total"]) == (2138, 12300), (r["udp_lost"], r["udp_total"])
        assert r["udp_loss_pct"] == round(100 * 2138 / 12300, 4)
        assert (r["steps_with_report"], r["steps_no_report"]) == (3, 1)
        assert r["ping_replies"] == 5 and r["ping_mean_ms"] == 22.0
        assert r["ping_p95_ms"] == round(float(np.percentile([1, 2, 3, 4, 100], 95)), 4)
        assert r["flip_lines"] == 2
        assert r["frac_written_above_100"] == 0.5
        # Write at 10:00:05.000 -> FLIP at 05.600 (0.6 s); write at 15.000 -> FLIP at 15.900 (0.9 s).
        assert r["flips_matched"] == 2 and r["median_write_to_flip_s"] == 0.75, r

        s = by_mode["static"]
        assert s["udp_loss_pct"] == NOT_COMPUTED and s["steps_no_report"] == 1
        assert s["ping_mean_ms"] == NOT_COMPUTED and s["ping_p95_ms"] == NOT_COMPUTED
        assert s["median_write_to_flip_s"] == NOT_COMPUTED

    # Matching rules on their own.
    steps = [(0.0, 0.0), (150.0, 5.0), (150.0, 10.0), (50.0, 15.0)]
    one = match_flip_delays(steps, [(5.6, "primary", "alternate")])
    assert len(one) == 1 and abs(one[0] - 0.6) < 1e-9
    # FLIP after a write that did not change the path (10.0) -> no match.
    assert match_flip_delays(steps, [(10.4, "primary", "alternate")]) == []
    # Wrong direction -> no match.
    assert match_flip_delays(steps, [(5.6, "alternate", "primary")]) == []
    # Two FLIPs for the same write -> ambiguous, no match.
    assert match_flip_delays(steps, [(5.3, "primary", "alternate"), (5.8, "primary", "alternate")]) == []
    # FLIP more than 5 s after the last write -> no match.
    assert match_flip_delays(steps, [(21.0, "alternate", "primary")]) == []
    # Midnight wrap: write at 23:59:59.800, FLIP at 00:00:00.300 -> 0.5 s.
    wrap = match_flip_delays([(0.0, 86390.0), (150.0, 86399.8)], [(0.3, "primary", "alternate")])
    assert len(wrap) == 1 and abs(wrap[0] - 0.5) < 1e-6

    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description="Summarize FedFlow-LLM experiment raw files")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="folder with raw_* files (default results/experiment)")
    parser.add_argument("--selftest", action="store_true", help="test parsers on made-up raw text")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        summarize(args.dir)


if __name__ == "__main__":
    main()
