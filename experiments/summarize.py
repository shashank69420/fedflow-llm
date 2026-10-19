"""
FedFlow-LLM — experiments/summarize.py
---------------------------------------
Reads only the raw files written by run_experiment.py and produces:
  summary.csv   one row per mode and rep
  summary.png   loss % (server side), p95 ping RTT and delivered-vs-alt % per mode
                (mean of reps, min-max error bars)

Usage (no sudo):
  python3 experiments/summarize.py                          results/experiment/
  python3 experiments/summarize.py --dir results/experiment/quick
  python3 experiments/summarize.py --selftest               made-up raw text, temp folder
  python3 experiments/summarize.py --steps-check            per-step static1 vs alt1 table only
                                                            (writes static1_steps_check.txt, not summary.*)

UDP loss comes from the iperf SERVER's own per-connection report
(raw_<mode>_<rep>_server.txt), matched to steps by the "### step i" markers in
that file. The client-side Server Report is kept as a cross-check column; garbled
client report lines (e.g. "4294966796/0 (0%)") are excluded and counted in client_bad_reports.
Nothing is estimated: a step without a report is counted, never filled in, and
anything that can't be computed is written as a reason string instead of a number.
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
NO_SERVER_FILE = "no server file"
MARKER_MISMATCH = "server marker mismatch"
COUNT_MISMATCH = "server report count mismatch"

STEP_MARK = re.compile(r"^### step (\d+)\b", re.M)
# iperf 2 client "Server Report" data line ends with "lost/total (pct%)", e.g. "0.021 ms 12/4253 (0.28%)".
LOSS_RE = re.compile(r"(\d+)\s*/\s*(\d+)\s+\(([-\d.eE+]+|nan)%\)")
# iperf 2 server per-connection report line: "<a>-<b> sec ... lost/total (pct%)".
# Same pattern as SERVER_REPORT_RE in run_experiment.py.
SERVER_REPORT_RE = re.compile(r"\d+(?:\.\d+)?-\s*\d+(?:\.\d+)?\s+sec\s.*?(\d+)\s*/\s*(\d+)\s+\(([^)]*)%\)")
RTT_RE = re.compile(r"time=([\d.]+) ms")
FLIP_RE = re.compile(r"\[FedFlow\] (\d\d):(\d\d):(\d\d)\.(\d{3}) FLIP (\w+) -> (\w+)")


def read(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return f.read()


def split_blocks(text):
    """[(step_number, block_text)] for each "### step i" marker, in file order."""
    marks = list(STEP_MARK.finditer(text))
    blocks = []
    for k, mark in enumerate(marks):
        end = marks[k + 1].start() if k + 1 < len(marks) else len(text)
        blocks.append((int(mark.group(1)), text[mark.end():end]))
    return blocks


# ---------------------------------------------------------------- parsers

def client_report_ok(lost, total, printed_pct):
    """False for a garbled client "Server Report" line, e.g. "4294966796/0 (0%)", where
    lost > total or total is 0, or where lost/total disagrees with the printed (x%)."""
    if total <= 0 or lost > total:
        return False
    try:
        printed = float(printed_pct)
    except ValueError:
        return False  # "nan" or other non-number printed
    # iperf prints the % with 2 significant digits, so rounding moves it by at most ~5% of itself.
    return abs(100.0 * lost / total - printed) <= 0.051 * abs(printed) + 1e-9


def parse_client_iperf(text):
    """Client-side cross-check. Returns (lost_sum, total_sum, steps_with_report,
    steps_without_report, bad_reports). Bad (garbled) report lines are counted, not summed."""
    lost = total = with_report = without = bad = 0
    for _, block in split_blocks(text):
        report_at = block.find("Server Report")
        match = LOSS_RE.search(block, report_at) if report_at >= 0 else None
        if match is None:
            without += 1
            continue
        if not client_report_ok(int(match.group(1)), int(match.group(2)), match.group(3)):
            bad += 1  # excluded from the sums; reported in client_bad_reports
            continue
        lost += int(match.group(1))
        total += int(match.group(2))
        with_report += 1
    return lost, total, with_report, without, bad


def parse_server(text, step_ids):
    """Server-side loss, matched to steps by the file's own "### step i" markers.
    Returns {"status", "lost", "total", "with_report", "without"}. status is "ok",
    MARKER_MISMATCH (a step's marker missing/duplicated/unknown) or COUNT_MISMATCH
    (a step block has more than one report line); on a mismatch nothing is matched."""
    blocks = split_blocks(text)
    numbers = [n for n, _ in blocks]
    if len(numbers) != len(set(numbers)) or set(numbers) != set(step_ids):
        return {"status": MARKER_MISMATCH}
    lost = total = with_report = without = 0
    for _, block in blocks:
        reports = [SERVER_REPORT_RE.search(line) for line in block.splitlines()]
        reports = [m for m in reports if m]
        if len(reports) > 1:
            return {"status": COUNT_MISMATCH}  # one connection per step can't have 2 reports
        if not reports:
            without += 1
            continue
        lost += int(reports[0].group(1))
        total += int(reports[0].group(2))
        with_report += 1
    return {"status": "ok", "lost": lost, "total": total, "with_report": with_report, "without": without}


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
    """[(step_number, value_written, seconds_of_day_written)] in file order."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = datetime.strptime(r["time_written"], "%Y-%m-%d %H:%M:%S.%f")
            rows.append((int(r["step"]), float(r["value_written"]),
                         t.hour * 3600 + t.minute * 60 + t.second + t.microsecond / 1e6))
    return rows


def path_for(value):
    return "alternate" if value > THRESHOLD else "primary"


def match_flip_delays(steps, flips):
    """Delay (s) from a step write to the FLIP it caused, only where the match is unambiguous:
    the latest write at or before the FLIP is a write that CHANGES the path to the FLIP's new
    path, the FLIP is within MATCH_WINDOW_S of it, and no other FLIP is matched to that write.
    steps = [(value_written, seconds_of_day)]. Times are time-of-day, so differences are mod 24 h."""
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
    steps = parse_steps(prefix + "_steps.csv")
    row = {"steps": len(steps)}

    # Server side: the main loss measure.
    if os.path.exists(prefix + "_server.txt"):
        server = parse_server(read(prefix + "_server.txt"), [n for n, _, _ in steps])
    else:
        server = {"status": NO_SERVER_FILE}
    row["server_status"] = server["status"]
    if server["status"] == "ok":
        row["server_udp_lost"] = server["lost"]
        row["server_udp_total"] = server["total"]
        row["server_udp_loss_pct"] = round(100.0 * server["lost"] / server["total"], 4) \
            if server["total"] > 0 else NOT_COMPUTED
        row["server_steps_with_report"] = server["with_report"]
        row["steps_no_server_report"] = server["without"]
    else:
        # Mismatch or missing file: no numbers at all for the server side.
        for key in ("server_udp_lost", "server_udp_total", "server_steps_with_report",
                    "steps_no_server_report"):
            row[key] = NOT_COMPUTED
        row["server_udp_loss_pct"] = server["status"]

    # Client side: cross-check only.
    lost, total, with_report, without, bad = parse_client_iperf(read(prefix + "_iperf.txt")) \
        if os.path.exists(prefix + "_iperf.txt") else (0, 0, 0, 0, 0)
    row["client_udp_loss_pct"] = round(100.0 * lost / total, 4) if total > 0 else NOT_COMPUTED
    row["client_steps_with_report"] = with_report
    row["client_steps_no_report"] = without
    row["client_bad_reports"] = bad

    rtts = parse_ping(read(prefix + "_ping.txt")) if os.path.exists(prefix + "_ping.txt") else []
    row["ping_replies"] = len(rtts)
    row["ping_mean_ms"] = round(float(np.mean(rtts)), 4) if rtts else NOT_COMPUTED
    row["ping_p95_ms"] = round(float(np.percentile(rtts, 95)), 4) if rtts else NOT_COMPUTED

    flips = parse_flips(read(prefix + "_pox.txt")) if os.path.exists(prefix + "_pox.txt") else []
    row["flip_lines"] = len(flips)

    row["frac_written_above_100"] = round(sum(v > THRESHOLD for _, v, _ in steps) / len(steps), 4) \
        if steps else NOT_COMPUTED

    delays = match_flip_delays([(v, t) for _, v, t in steps], flips)
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


def is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def plot(rows, png_path):
    import matplotlib
    matplotlib.use("Agg")  # no display needed
    import matplotlib.pyplot as plt

    modes = [m for m in MODES if any(r["mode"] == m for r in rows)]
    modes += sorted({r["mode"] for r in rows} - set(modes))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key, label in [(axes[0], "server_udp_loss_pct", "UDP loss, server side (%)"),
                           (axes[1], "ping_p95_ms", "Ping p95 RTT (ms)"),
                           (axes[2], "delivered_vs_alt_pct", "Datagrams delivered vs alt (%)")]:
        means, lows, highs, names = [], [], [], []
        for m in modes:
            vals = [r[key] for r in rows if r["mode"] == m and is_number(r[key])]
            if not vals:
                continue  # nothing measured for this mode: leave the bar out rather than draw 0
            mean = float(np.mean(vals))
            means.append(mean)
            lows.append(mean - min(vals))
            highs.append(max(vals) - mean)
            names.append(f"{m}\n(n={len(vals)})")
        ax.bar(names, means, yerr=[lows, highs], capsize=4, color="#4C78A8")
        ax.set_ylabel(label)
        ax.set_title(f"{label}\nmean of reps, min-max bars", fontsize=10)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


FIELDS = ["mode", "rep", "steps", "server_status", "server_udp_lost", "server_udp_total",
          "server_udp_loss_pct", "server_steps_with_report", "steps_no_server_report",
          "client_udp_loss_pct", "client_steps_with_report", "client_steps_no_report",
          "client_bad_reports", "ping_replies", "ping_mean_ms", "ping_p95_ms", "flip_lines",
          "frac_written_above_100", "flips_matched", "median_write_to_flip_s",
          "delivered_vs_alt_pct"]


def add_delivered_vs_alt(rows):
    """delivered_vs_alt_pct = run's server_udp_total / mean server_udp_total of the alt runs x 100.
    "not computed" everywhere if there is no alt run or any alt run has no numeric total."""
    alt_totals = [r["server_udp_total"] for r in rows if r["mode"] == "alt"]
    usable = bool(alt_totals) and all(is_number(t) for t in alt_totals)
    alt_mean = float(np.mean(alt_totals)) if usable else 0.0
    for r in rows:
        if usable and alt_mean > 0 and is_number(r["server_udp_total"]):
            r["delivered_vs_alt_pct"] = round(100.0 * r["server_udp_total"] / alt_mean, 2)
        else:
            r["delivered_vs_alt_pct"] = NOT_COMPUTED


def summarize(folder, make_plot=True):
    runs = find_runs(folder)
    if not runs:
        sys.exit(f"ERROR: no raw_*_steps.csv files in {folder}")
    rows = []
    for mode, rep, prefix in runs:
        row = {"mode": mode, "rep": rep}
        row.update(summarize_run(prefix))
        rows.append(row)
    add_delivered_vs_alt(rows)
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


# ---------------------------------------------------------------- per-step check

DATAGRAM_BYTES = 1470   # iperf UDP datagram size, as printed by the client ("Sending 1470 byte datagrams")
STEP_SECONDS = 5        # iperf -t 5 in run_experiment.py
PRIMARY_MBPS = 10.0     # primary path capacity (CLAUDE.md)
CHECK_RUNS = [("static", 1), ("alt", 1)]
CHECK_FILE = "static1_steps_check.txt"
RATE_MARK = re.compile(r"\brate ([\d.]+)M\b")
SENT_RE = re.compile(r"Sent (\d+) datagrams")
# First interval line of the client block, e.g. "[  1] 0.0000-5.0349 sec  5.90 MBytes  9.83 Mbits/sec".
INTERVAL_RE = re.compile(r"(\d+(?:\.\d+)?)-\s*(\d+(?:\.\d+)?)\s+sec\s")


def step_check_rows(prefix):
    """{step: dict} for one run, straight from its steps.csv, client and server files."""
    true_values = {}
    with open(prefix + "_steps.csv", newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            true_values[int(r["step"])] = r["true_value"]
    server = dict(split_blocks(read(prefix + "_server.txt")))
    rows = {}
    for step, block in split_blocks(read(prefix + "_iperf.txt")):
        # The block starts with the rest of the marker line (" timestep ... rate 6.895M written ...").
        marker_rate = RATE_MARK.search(block.split("\n", 1)[0])
        sent = SENT_RE.search(block)
        sent_at = sent.start() if sent else len(block)
        interval = INTERVAL_RE.search(block, 0, sent_at)  # the client's own line, before "Sent"
        report_at = block.find("Server Report")
        client_report = LOSS_RE.search(block, report_at) if report_at >= 0 else None
        server_reports = [m for m in (SERVER_REPORT_RE.search(line)
                                      for line in server.get(step, "").splitlines()) if m]
        row = {"true": true_values.get(step, "none"),
               "rate": float(marker_rate.group(1)) if marker_rate else None,
               "sent": int(sent.group(1)) if sent else None,
               "duration": float(interval.group(2)) - float(interval.group(1)) if interval else None,
               "client_report": client_report.group(0) if client_report else "no report",
               "srv_lost": server_reports[0].group(1) if len(server_reports) == 1 else "none",
               "srv_total": server_reports[0].group(2) if len(server_reports) == 1 else "none"}
        if step not in server:
            row["srv_lost"] = row["srv_total"] = "no marker"
        elif len(server_reports) > 1:
            row["srv_lost"] = row["srv_total"] = "2+ reports"
        # Offered rate = what the client actually put on the wire over its own printed duration.
        if row["sent"] is not None and row["duration"]:
            row["offered"] = row["sent"] * DATAGRAM_BYTES * 8 / row["duration"] / 1e6
        else:
            row["offered"] = None
        rows[step] = row
    return rows


def fmt(value, spec):
    return "none" if value is None else format(value, spec)


def steps_check(folder):
    """Side-by-side per-step table for CHECK_RUNS. Returns the text."""
    runs = {}
    for mode, rep in CHECK_RUNS:
        prefix = os.path.join(folder, f"raw_{mode}_{rep}")
        if not os.path.exists(prefix + "_server.txt"):
            sys.exit(f"ERROR: {prefix}_server.txt not found")
        runs[(mode, rep)] = step_check_rows(prefix)
    first = runs[CHECK_RUNS[0]]
    steps = sorted(first)
    for key, rows in runs.items():
        if sorted(rows) != steps:
            sys.exit(f"ERROR: {key} has different step numbers from {CHECK_RUNS[0]}")
        for s in steps:
            # Same trace in every mode: true value and target rate must agree, else the table would mislead.
            if (rows[s]["true"], rows[s]["rate"]) != (first[s]["true"], first[s]["rate"]):
                sys.exit(f"ERROR: step {s} true value/rate differ between {CHECK_RUNS[0]} and {key}")

    names = [f"{m}{r}" for m, r in CHECK_RUNS]
    lines = [f"Per-step check: {' vs '.join(names)} (from {os.path.relpath(folder, REPO)}/raw_* files)",
             f"target Mbps = rate in the '### step' marker; expected dg = target x {STEP_SECONDS} s / "
             f"({DATAGRAM_BYTES} B x 8)",
             f"offered Mbps = Sent x {DATAGRAM_BYTES} x 8 / client interval duration / 1e6; "
             f"srv = iperf server file; client report = text as printed after 'Server Report'",
             ""]
    head = f"{'step':>4} {'true':>7} {'target':>7} {'expected':>8}"
    sub = f"{'':>4} {'value':>7} {'Mbps':>7} {'dg':>8}"
    for n in names:
        head += f" | {n + ' offered':>14} {'sent':>5} {'srvLost':>7} {'srvTotal':>8} {'client report':>22}"
        sub += f" | {'Mbps':>14} {'dg':>5} {'':>7} {'':>8} {'':>22}"
    lines += [head, sub, "-" * len(head)]
    for s in steps:
        rate = first[s]["rate"]
        expected = rate * 1e6 * STEP_SECONDS / (DATAGRAM_BYTES * 8) if rate is not None else None
        line = f"{s:>4} {first[s]['true']:>7} {fmt(rate, '.3f'):>7} {fmt(expected, '.1f'):>8}"
        for key in CHECK_RUNS:
            r = runs[key][s]
            line += (f" | {fmt(r['offered'], '.2f'):>14} {fmt(r['sent'], 'd'):>5} {r['srv_lost']:>7}"
                     f" {r['srv_total']:>8} {r['client_report']:>22}")
        lines.append(line)

    lines.append("")
    targets_over = sum(1 for s in steps if first[s]["rate"] is not None and first[s]["rate"] > PRIMARY_MBPS)
    lines.append(f"steps with target > {PRIMARY_MBPS:g} Mbps: {targets_over} of {len(steps)}")
    for name, key in zip(names, CHECK_RUNS):
        rows = runs[key]
        offered = [rows[s]["offered"] for s in steps if rows[s]["offered"] is not None]
        sent = sum(rows[s]["sent"] for s in steps if rows[s]["sent"] is not None)
        srv_total = sum(int(rows[s]["srv_total"]) for s in steps if rows[s]["srv_total"].isdigit())
        srv_lost = sum(int(rows[s]["srv_lost"]) for s in steps if rows[s]["srv_lost"].isdigit())
        over = sum(1 for o in offered if o > PRIMARY_MBPS)
        lines.append(f"{name}: steps with offered > {PRIMARY_MBPS:g} Mbps: {over} of {len(steps)}; "
                     f"max offered {fmt(max(offered) if offered else None, '.2f')} Mbps; "
                     f"sum sent {sent}; sum srvTotal {srv_total}; sum srvLost {srv_lost}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- selftest

STEPS_4 = """step,timestep,true_value,value_written,time_written
0,1452,68.95,0.00,2026-10-20 10:00:00.000
1,1453,150.00,150.00,2026-10-20 10:00:05.000
2,1454,150.00,150.00,2026-10-20 10:00:10.000
3,1455,70.00,50.00,2026-10-20 10:00:15.000
"""

SERVER_HEAD = """------------------------------------------------------------
Server listening on UDP port 5001
UDP buffer size:  208 KByte (default)
------------------------------------------------------------
"""


def server_block(i, lost=None, total=None, conn=1):
    """Made-up server output for step i; lost=None means the server printed no report."""
    text = f"### step {i} timestep {1452 + i} rate 6.895M written 0.00\n" + SERVER_HEAD
    text += f"[  {conn}] local 10.0.0.1 port 5001 connected with 10.0.0.2 port {40000 + i}\n"
    if lost is None:
        return text + f"[  {conn}] WARNING: ack of last datagram failed.\n"
    return (text + "[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams\n"
            f"[  {conn}] 0.0000-5.0012 sec  5.96 MBytes  10.0 Mbits/sec   0.500 ms {lost}/{total} "
            f"({100 * lost / total:.2g}%)\n")


def write_run(folder, name, files):
    prefix = os.path.join(folder, name)
    for suffix, text in files.items():
        with open(prefix + suffix, "w", encoding="utf-8") as f:
            f.write(text)


def run_selftest():
    client_text = """### step 0 timestep 1452 rate 6.895M written 0.00
[  1] local 10.0.0.2 port 40000 connected with 10.0.0.1 port 5001
[  1] Sent 2932 datagrams
[  1] Server Report:
[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams
[  1] 0.0000-5.0012 sec  4.10 MBytes  6.88 Mbits/sec   0.020 ms 0/2932 (0%)
### step 1 timestep 1453 rate 15.000M written 150.00
[  1] Sent 6378 datagrams
[  1] WARNING: did not receive ack of last datagram after 10 tries.
### step 2 timestep 1454 rate 15.000M written 150.00
[  1] Sent 6378 datagrams
[  1] WARNING: did not receive ack of last datagram after 10 tries.
### step 3 timestep 1455 rate 7.000M written 50.00
[  1] Server Report:
[  1] 0.0000-5.0000 sec  4.17 MBytes  7.00 Mbits/sec   0.010 ms 10/2990 (0.33%)
"""
    # Server: steps 0, 1, 3 report; step 2 has no report (counted, not filled in).
    server_ok = (server_block(0, 0, 2932) + server_block(1, 2128, 6378)
                 + server_block(2) + server_block(3, 10, 2990))
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
    base = {"_iperf.txt": client_text, "_ping.txt": ping_text, "_pox.txt": pox_text, "_steps.csv": STEPS_4}

    with tempfile.TemporaryDirectory() as tmp:
        write_run(tmp, "raw_model_1", {**base, "_server.txt": server_ok})
        # Marker for step 2 missing -> marker mismatch.
        write_run(tmp, "raw_model_2", {**base, "_server.txt": server_block(0, 0, 1) + server_block(1, 0, 1)
                                       + server_block(3, 0, 1)})
        # Marker for step 1 duplicated -> marker mismatch.
        write_run(tmp, "raw_model_3", {**base, "_server.txt": server_ok + server_block(1, 5, 10)})
        # Step 1 block has two report lines -> report count mismatch.
        write_run(tmp, "raw_oracle_1", {**base, "_server.txt": server_block(0, 0, 1)
                                        + server_block(1, 1, 10) + server_block(1, 2, 10, conn=2)[
                                            len("### step 1 timestep 1453 rate 6.895M written 0.00\n"):]
                                        + server_block(2, 0, 1) + server_block(3, 0, 1)})
        # A run with no server file, no client reports, and empty ping/FLIPs.
        write_run(tmp, "raw_static_1", {
            "_iperf.txt": "### step 0 timestep 1452 rate 6.895M written 0.00\nno output\n",
            "_ping.txt": "PING 10.0.0.2\n", "_pox.txt": "nothing\n",
            "_steps.csv": "step,timestep,true_value,value_written,time_written\n"
                          "0,1452,68.95,0.00,2026-10-20 10:00:00.000\n"})

        rows = summarize(tmp, make_plot=True)
        assert os.path.exists(os.path.join(tmp, "summary.png"))
        assert [(r["mode"], r["rep"]) for r in rows] == \
            [("static", 1), ("model", 1), ("model", 2), ("model", 3), ("oracle", 1)], "MODES order"
        by = {(r["mode"], r["rep"]): r for r in rows}
        # No alt run in this folder -> delivered_vs_alt_pct can't be computed for anyone.
        assert all(r["delivered_vs_alt_pct"] == NOT_COMPUTED for r in rows)

        r = by[("model", 1)]
        assert r["server_status"] == "ok"
        # Server: steps 0, 1, 3 -> lost 0 + 2128 + 10, total 2932 + 6378 + 2990; step 2 no report.
        assert (r["server_udp_lost"], r["server_udp_total"]) == (2138, 12300), r
        assert r["server_udp_loss_pct"] == round(100 * 2138 / 12300, 4)
        assert (r["server_steps_with_report"], r["steps_no_server_report"]) == (3, 1)
        # Client cross-check: steps 0 and 3 only -> 10 / (2932 + 2990).
        assert r["client_udp_loss_pct"] == round(100 * 10 / 5922, 4)
        assert (r["client_steps_with_report"], r["client_steps_no_report"]) == (2, 2)
        assert r["ping_replies"] == 5 and r["ping_mean_ms"] == 22.0
        assert r["ping_p95_ms"] == round(float(np.percentile([1, 2, 3, 4, 100], 95)), 4)
        assert r["flip_lines"] == 2
        assert r["frac_written_above_100"] == 0.5
        # Write at 10:00:05.000 -> FLIP at 05.600 (0.6 s); write at 15.000 -> FLIP at 15.900 (0.9 s).
        assert r["flips_matched"] == 2 and r["median_write_to_flip_s"] == 0.75, r

        for key in [("model", 2), ("model", 3)]:
            r = by[key]
            assert r["server_status"] == MARKER_MISMATCH and r["server_udp_loss_pct"] == MARKER_MISMATCH, r
            assert r["server_udp_lost"] == NOT_COMPUTED and r["steps_no_server_report"] == NOT_COMPUTED

        r = by[("oracle", 1)]
        assert r["server_status"] == COUNT_MISMATCH and r["server_udp_loss_pct"] == COUNT_MISMATCH, r
        assert r["server_udp_total"] == NOT_COMPUTED

        s = by[("static", 1)]
        assert s["server_status"] == NO_SERVER_FILE and s["server_udp_loss_pct"] == NO_SERVER_FILE
        assert s["client_udp_loss_pct"] == NOT_COMPUTED and s["client_steps_no_report"] == 1
        assert s["ping_mean_ms"] == NOT_COMPUTED and s["ping_p95_ms"] == NOT_COMPUTED
        assert s["median_write_to_flip_s"] == NOT_COMPUTED

    # Marker numbers must match steps.csv exactly (file numbers steps from 0).
    assert parse_server(server_block(1, 0, 1) + server_block(2, 0, 1), [0, 1])["status"] == MARKER_MISMATCH
    assert parse_server(server_block(0, 0, 1) + server_block(1, 3, 9), [0, 1])["lost"] == 3
    # Header and "connected" lines are not reports.
    assert parse_server(server_block(0), [0]) == {"status": "ok", "lost": 0, "total": 0,
                                                  "with_report": 0, "without": 1}

    # delivered_vs_alt_pct: alt mean = (100 + 200 + 300) / 3 = 200.
    made = [{"mode": "alt", "server_udp_total": t} for t in (100, 200, 300)] + \
           [{"mode": "static", "server_udp_total": 150}, {"mode": "model", "server_udp_total": NOT_COMPUTED}]
    add_delivered_vs_alt(made)
    assert [r["delivered_vs_alt_pct"] for r in made] == [50.0, 100.0, 150.0, 75.0, NOT_COMPUTED], made
    made[0]["server_udp_total"] = NOT_COMPUTED  # one alt run unusable -> nothing computed
    add_delivered_vs_alt(made)
    assert all(r["delivered_vs_alt_pct"] == NOT_COMPUTED for r in made)

    # Client report check: garbled line (seen in real runs), printed-% mismatch, and good lines.
    assert not client_report_ok(4294966796, 0, "0")       # lost > total and total 0
    assert not client_report_ok(5, 10, "nan")
    assert not client_report_ok(10, 2990, "5")            # 0.33% computed, 5% printed
    assert client_report_ok(10, 2990, "0.33") and client_report_ok(0, 3077, "0")
    assert not client_report_ok(1, 3077, "0")             # 0.032% computed, iperf would not print 0
    garbled = ("### step 0 timestep 1452 rate 13.161M written 0.00\n[  1] Sent 4209 datagrams\n"
               "[  1] Server Report:\n[  1] 0.0000-0.0000 sec  4.00 GBytes  -nan bits/sec   0.000 ms "
               "4294966796/0 (0%) -/-/-/- ms 0 pps\n"
               "### step 1 timestep 1453 rate 6.771M written 0.00\n[  1] Server Report:\n"
               "[  1] 0.0000-4.9981 sec  4.24 MBytes  7.11 Mbits/sec   0.005 ms 0/3022 (0%)\n"
               "### step 2 timestep 1454 rate 6.771M written 0.00\n[  1] Server Report:\n"
               "[  1] 0.0000-4.9981 sec  4.24 MBytes  7.11 Mbits/sec   0.005 ms 10/2990 (5%)\n")
    assert parse_client_iperf(garbled) == (0, 3022, 1, 0, 2), parse_client_iperf(garbled)

    # Per-step check table on made-up static1/alt1 files.
    client_step = ("### step 0 timestep 1459 rate 13.161M written {w}\n"
                   "[  1] 0.0000-{d} sec  5.90 MBytes  9.83 Mbits/sec\n[  1] Sent {n} datagrams\n"
                   "[  1] Server Report:\n[  1] 0.0000-5.0 sec  1 MBytes  1 Mbits/sec   0.0 ms {rep}\n")
    with tempfile.TemporaryDirectory() as tmp:
        one_step = "step,timestep,true_value,value_written,time_written\n0,1459,131.61,{w},2026-10-20 10:00:00.000\n"
        write_run(tmp, "raw_static_1", {"_steps.csv": one_step.format(w="0.00"),
                                        "_iperf.txt": client_step.format(w="0.00", d="5.0349", n=4209,
                                                                         rep="4294966796/0 (0%)"),
                                        "_server.txt": server_block(0, 0, 4208)})
        write_run(tmp, "raw_alt_1", {"_steps.csv": one_step.format(w="1000.00"),
                                     "_iperf.txt": client_step.format(w="1000.00", d="5.0000", n=5597,
                                                                      rep="0/5596 (0%)"),
                                     "_server.txt": server_block(0, 0, 5596)})
        text = steps_check(tmp)
        row = [ln for ln in text.splitlines() if ln.startswith("   0 ")][0]
        # expected = 13.161e6 * 5 / 11760 = 5595.7; offered static = 4209 * 11760 / 5.0349 / 1e6 = 9.83
        assert "5595.7" in row and "9.83" in row and "4294966796/0 (0%)" in row and "4208" in row, row
        assert f"{5597 * 11760 / 5.0 / 1e6:.2f}" in row, row
        assert "static1: steps with offered > 10 Mbps: 0 of 1" in text, text
        assert "alt1: steps with offered > 10 Mbps: 1 of 1" in text, text

    # FLIP matching rules on their own (steps as (value, time)).
    steps = [(0.0, 0.0), (150.0, 5.0), (150.0, 10.0), (50.0, 15.0)]
    one = match_flip_delays(steps, [(5.6, "primary", "alternate")])
    assert len(one) == 1 and abs(one[0] - 0.6) < 1e-9
    assert match_flip_delays(steps, [(10.4, "primary", "alternate")]) == []   # write didn't change path
    assert match_flip_delays(steps, [(5.6, "alternate", "primary")]) == []    # wrong direction
    assert match_flip_delays(steps, [(5.3, "primary", "alternate"),
                                     (5.8, "primary", "alternate")]) == []    # two FLIPs, one write
    assert match_flip_delays(steps, [(21.0, "alternate", "primary")]) == []   # > 5 s after last write
    wrap = match_flip_delays([(0.0, 86390.0), (150.0, 86399.8)], [(0.3, "primary", "alternate")])
    assert len(wrap) == 1 and abs(wrap[0] - 0.5) < 1e-6                       # midnight wrap

    print("SELFTEST OK")


def main():
    parser = argparse.ArgumentParser(description="Summarize FedFlow-LLM experiment raw files")
    parser.add_argument("--dir", default=DEFAULT_DIR, help="folder with raw_* files (default results/experiment)")
    parser.add_argument("--selftest", action="store_true", help="test parsers on made-up raw text")
    parser.add_argument("--steps-check", action="store_true",
                        help=f"per-step static1 vs alt1 table, printed and saved to <dir>/{CHECK_FILE}")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    elif args.steps_check:
        text = steps_check(args.dir)
        print(text, end="")
        with open(os.path.join(args.dir, CHECK_FILE), "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Wrote {os.path.join(args.dir, CHECK_FILE)}")
    else:
        summarize(args.dir)


if __name__ == "__main__":
    main()
