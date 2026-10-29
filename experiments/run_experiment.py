"""
FedFlow-LLM — experiments/run_experiment.py
--------------------------------------------
Replays one traffic trace through the Mininet + POX dual-path network under
5 routing modes and saves raw logs for experiments/summarize.py.

Usage (from the repo root):
  python3 experiments/run_experiment.py --selftest       no Mininet, no sudo
  sudo python3 experiments/run_experiment.py --quick     oracle mode, 1 rep, short trace -> results/experiment/quick/
  sudo python3 experiments/run_experiment.py --quick --mode reactive
                                                         same quick test for another mode (default oracle)
  sudo python3 experiments/run_experiment.py             5 modes x 3 reps (about 50 minutes)
  add --dataset telecom to any of these to use a Telecom Italia experiment cell; --cell <square_id>
  picks which one (7511 = primary, the default; 8336 = second, declared before any model result):
    true series  datasets/telecom_milan/cells.csv (that square's column)
    forecasts    results/final_telecom/outputs/forecasts.csv, rows of that cell only
    output       results/experiment_telecom/<square_id>/
    quick test   sudo python3 experiments/run_experiment.py --dataset telecom --quick --mode reactive
                 (--cell <square_id> as above). Needs no forecasts.csv. 1 rep, the first N steps of the slice,
                 N = the smallest number of steps that contains an upward crossing of 100 in the WRITTEN values,
                 plus 3 more steps. N is printed. Output: results/experiment_telecom/<square_id>/quick/.
                 The default --mode oracle cannot run on these cells (the oracle never crosses upward).
    slice        same rule, starts 1740-1980 (so every step has a forecast); the recomputed slice
                 must equal the one saved for that cell in cells_meta.json or the run stops
    --forecasts PATH   with --dataset telecom: read the model forecasts from PATH instead of
                       results/final_telecom/outputs/forecasts.csv (e.g. a v2 run, PLAN_V2.md)
    --out-root DIR     with --dataset telecom: write to DIR/<square_id>/ instead of
                       results/experiment_telecom/<square_id>/. A non-default --forecasts needs a
                       non-default --out-root, so v1 results are never mixed with or overwritten by another run.

Trace (synthetic): the 36-step slice of the held-out block (timesteps 1452-1727) with the
most upward crossings of 100 in the TRUE traffic, earliest on ties. Step value
v -> UDP load of v/10 Mbps from h2 to h1 for 5 s.

Modes (value written to forecast_value.txt at the start of each step):
  static    0 (primary path always)
  reactive  true value of the previous step
  model     peak of the model's 6 predictions, most recent window whose history
            ends at or before the previous step
  oracle    peak of the 6 TRUE values over the same window as model
  alt       1000 (alternate path always), reference only

Known effects (identical in every mode, not tuned):
  - The controller polls the file every 1 s, so each step can spend up to ~1 s
    on the previous path before a flip.
  - A flip deletes all flows, including the ping flows; they are reinstalled on
    the next packet.
  - On an overloaded link the iperf Server Report can be lost; that step is
    recorded as "no report", never as a number.
  - iperf 2.1.5's long-running UDP server stops reporting after its 2nd client
    (reproduced on loopback, no network involved). So each step gets a FRESH
    server on h1: start it, wait SERVER_START_WAIT, write the forecast, start the
    client immediately, wait SERVER_STOP_WAIT after the client ends, stop the
    server. This adds about 0.6 s between steps, identical in every mode. The
    server's own per-connection report is saved to raw_<mode>_<rep>_server.txt.
  - A model or oracle window can include up to 2 steps that have already passed
    (windows start every 3 timesteps); this applies equally to both modes.

Only the Python standard library and mininet are imported, because this runs
under sudo.
"""
import sys

sys.dont_write_bytecode = True  # never let root write __pycache__ into the repo

import argparse
import csv
import importlib.util
import json
import os
import pwd
import re
import signal
import socket
import subprocess
import tempfile
import time
from datetime import datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORECASTS_CSV = os.path.join(REPO, "results", "final", "outputs", "forecasts.csv")
TRAFFIC_CSV = os.path.join(REPO, "training", "data", "synthetic_traffic.csv")
TOPO_PY = os.path.join(REPO, "controller", "topo.py")
OUT_DIR = os.path.join(REPO, "results", "experiment")
QUICK_DIR = os.path.join(OUT_DIR, "quick")

THRESHOLD = 100.0
TRACE_LEN = 36
BLOCK_FIRST, BLOCK_LAST = 1452, 1727   # held-out timesteps that have true values in forecasts.csv
MIN_CROSSINGS = 3
STEP_SECONDS = 5
REPS = 3
MODES = ["static", "reactive", "model", "oracle", "alt"]
WINDOW_HISTORY = 12                    # a window starting at w has history w..w+11
POX_UP_TIMEOUT = 20
CONNECT_TIMEOUT = 20
SERVER_START_WAIT = 0.3   # s for a fresh iperf server to start listening
SERVER_STOP_WAIT = 0.3    # s after the client ends, so the server prints its report
SERVER_EXIT_TIMEOUT = 3   # s to wait for the server to exit after SIGINT before SIGKILL
N_SWITCHES = 4
OF_PORT = 6633
H1_IP, H2_IP = "10.0.0.1", "10.0.0.2"

# Telecom paths; only used after use_telecom() has been called (--dataset telecom).
TELECOM_FORECASTS_CSV = os.path.join(REPO, "results", "final_telecom", "outputs", "forecasts.csv")
TELECOM_CELLS_CSV = os.path.join(REPO, "datasets", "telecom_milan", "cells.csv")
TELECOM_META_JSON = os.path.join(REPO, "datasets", "telecom_milan", "cells_meta.json")
TELECOM_OUT_DIR = os.path.join(REPO, "results", "experiment_telecom")
TELECOM = None   # stays None for synthetic; use_telecom() fills it with the experiment cell's saved facts


def use_telecom(square=None, forecasts=None, out_root=None):
    """Point every path and the slice range at one Telecom Italia experiment cell.
    square=None means the primary experiment cell (the one the locked rule picked).
    forecasts / out_root = None keep the v1 defaults (TELECOM_FORECASTS_CSV, TELECOM_OUT_DIR)."""
    global FORECASTS_CSV, OUT_DIR, QUICK_DIR, BLOCK_FIRST, BLOCK_LAST, TELECOM
    with open(TELECOM_META_JSON, encoding="utf-8") as f:
        meta = json.load(f)
    rule = meta["slice_rule"]
    if rule["trace_len"] != TRACE_LEN or rule["threshold"] != THRESHOLD or rule["min_crossings"] != MIN_CROSSINGS:
        raise ValueError("cells_meta.json slice rule differs from this script's TRACE_LEN / THRESHOLD / MIN_CROSSINGS.")
    if square is None:
        square = int(meta["experiment_cell"])
    entry = next((e for e in meta["experiment_cells"] if int(e["square"]) == int(square)), None)
    if entry is None:   # only the declared experiment cells may be used
        allowed = [e["square"] for e in meta["experiment_cells"]]
        raise ValueError(f"Square {square} is not an experiment cell. Choose one of {allowed}.")
    TELECOM = {
        "square": int(entry["square"]),
        "cell": int(entry["cell"]),
        "start": int(entry["best_slice_start"]),
        "crossings": int(entry["best_slice_crossings"]),
    }
    FORECASTS_CSV = TELECOM_FORECASTS_CSV if forecasts is None else os.path.abspath(forecasts)
    root = TELECOM_OUT_DIR if out_root is None else os.path.abspath(out_root)
    OUT_DIR = os.path.join(root, str(TELECOM["square"]))   # one folder per experiment cell
    QUICK_DIR = os.path.join(OUT_DIR, "quick")
    # choose_slice tries starts BLOCK_FIRST .. BLOCK_LAST - TRACE_LEN + 1, so this gives starts 1740-1980.
    BLOCK_FIRST = rule["start_min"]
    BLOCK_LAST = rule["start_max"] + TRACE_LEN - 1


# ---------------------------------------------------------------- trace (pure, no Mininet)

def load_windows():
    """forecasts.csv -> {window_start: {"pred": [6], "true": [6]}} ordered by step."""
    windows = {}
    with open(FORECASTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if TELECOM is not None:
                # The telecom forecasts hold all six cells; keep only the experiment cell's rows.
                if "cell" not in row:
                    raise ValueError(f"{FORECASTS_CSV} has no cell column.")
                if row["cell"] != str(TELECOM["cell"]):
                    continue
            w = windows.setdefault(int(row["start"]), {})
            step = int(row["step"])
            if row["parsed"] != "True" or row["model"] == "":
                raise ValueError(f"Window {row['start']} has no parsed model prediction.")
            w[step] = (float(row["model"]), float(row["true"]))
    out = {}
    for start, steps in windows.items():
        if sorted(steps) != [1, 2, 3, 4, 5, 6]:
            raise ValueError(f"Window {start} does not have steps 1-6.")
        out[start] = {"pred": [steps[s][0] for s in range(1, 7)],
                      "true": [steps[s][1] for s in range(1, 7)]}
    return dict(sorted(out.items()))


def load_traffic():
    """synthetic_traffic.csv -> {timestep: true traffic}. With --dataset telecom: the experiment
    cell's column of cells.csv instead (scaled TRUE traffic)."""
    if TELECOM is not None:
        with open(TELECOM_CELLS_CSV, newline="", encoding="utf-8") as f:
            return {int(r["timestep"]): float(r[str(TELECOM["square"])]) for r in csv.DictReader(f)}
    if not os.path.exists(TRAFFIC_CSV):
        sys.exit(f"ERROR: {TRAFFIC_CSV} not found. Regenerate it with:\n"
                 f"  cd {os.path.dirname(os.path.dirname(TRAFFIC_CSV))} && python3 generate_data.py")
    with open(TRAFFIC_CSV, newline="", encoding="utf-8") as f:
        return {int(r["timestep"]): float(r["traffic"]) for r in csv.DictReader(f)}


def upward_crossings(values):
    """Count i where values[i-1] <= 100 and values[i] > 100."""
    return sum(1 for a, b in zip(values, values[1:]) if a <= THRESHOLD < b)


def choose_slice(traffic):
    """Start of the 36-step slice (TRUE values only) with the most crossings, earliest on ties."""
    best_start, best_count = None, -1
    for s in range(BLOCK_FIRST, BLOCK_LAST - TRACE_LEN + 2):
        count = upward_crossings([traffic[t] for t in range(s, s + TRACE_LEN)])
        if count > best_count:  # strict ">" keeps the earliest start on ties
            best_start, best_count = s, count
    if best_count < MIN_CROSSINGS:
        sys.exit(f"ERROR: best slice has only {best_count} upward crossings (need {MIN_CROSSINGS}).")
    return best_start, best_count


def latest_window(windows, t):
    """Most recent window whose history ends at or before step t-1 (w + 11 <= t - 1)."""
    candidates = [w for w in windows if w + WINDOW_HISTORY - 1 <= t - 1]
    if not candidates:
        raise ValueError(f"No forecast window has history ending at or before timestep {t - 1}.")
    return max(candidates)


def build_trace(needs_forecasts=True):
    """Returns (start, crossings, steps). Each step: timestep, true value, value per mode.
    needs_forecasts=False skips forecasts.csv entirely; the steps then have no model or oracle value
    (only static, reactive and alt), which is all a quick run of those modes needs."""
    windows = load_windows() if needs_forecasts else None
    traffic = load_traffic()
    # The true values in forecasts.csv must be the same series as synthetic_traffic.csv.
    for w, data in (windows or {}).items():
        for k, v in enumerate(data["true"]):
            if abs(traffic[w + WINDOW_HISTORY + k] - v) > 1e-9:
                raise ValueError(f"forecasts.csv true value at {w + WINDOW_HISTORY + k} "
                                 "does not match synthetic_traffic.csv.")

    start, crossings = choose_slice(traffic)
    if TELECOM is not None and (start, crossings) != (TELECOM["start"], TELECOM["crossings"]):
        # The slice must be exactly the one prepare_telecom.py saved; anything else means the data changed.
        raise ValueError(f"Recomputed slice (start {start}, crossings {crossings}) differs from cells_meta.json "
                         f"(start {TELECOM['start']}, crossings {TELECOM['crossings']}).")
    steps = []
    for t in range(start, start + TRACE_LEN):
        w = latest_window(windows, t) if needs_forecasts else None
        true_value = traffic[t]
        if true_value <= 0:
            raise ValueError(f"Timestep {t} has traffic {true_value}; iperf needs a positive rate.")
        written = {"static": 0.0, "reactive": traffic[t - 1], "alt": 1000.0}
        if needs_forecasts:   # model and oracle need the forecast windows
            written = {"static": 0.0, "reactive": traffic[t - 1],
                       "model": max(windows[w]["pred"]), "oracle": max(windows[w]["true"]), "alt": 1000.0}
        steps.append({"timestep": t, "true": true_value, "window": w, "written": written})
    return start, crossings, steps


def quick_length(steps, mode="oracle"):
    """Smallest N whose first N written values contain an upward crossing, plus 3 steps."""
    written = [s["written"][mode] for s in steps]
    for i in range(1, len(written)):
        if written[i - 1] <= THRESHOLD < written[i]:
            return min(i + 1 + 3, len(written))
    sys.exit(f"ERROR: {mode} mode never crosses {THRESHOLD:g} upward in this trace.")


# Same pattern as summarize.py (duplicated because this file must not import numpy):
# an iperf 2 data report line, e.g. "[  1] 0.0000-5.0058 sec  5.96 MBytes ... 0.020 ms 12/4253 (0.28%)".
SERVER_REPORT_RE = re.compile(r"\d+(?:\.\d+)?-\s*\d+(?:\.\d+)?\s+sec\s.*?(\d+)\s*/\s*(\d+)\s+\(([^)]*)%\)")


def count_server_reports(text):
    """Number of per-connection report lines in one step's server output (expected: 1)."""
    return sum(1 for line in text.splitlines() if SERVER_REPORT_RE.search(line))


def selftest_server_counting():
    good = """### step 0 timestep 1452 rate 6.895M written 0.00
------------------------------------------------------------
Server listening on UDP port 5001
------------------------------------------------------------
[  1] local 10.0.0.1 port 5001 connected with 10.0.0.2 port 40000
[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams
[  1] 0.0000-5.0012 sec  4.10 MBytes  6.88 Mbits/sec   0.020 ms 12/2932 (0.41%)
"""
    none = """### step 1 timestep 1453 rate 15.000M written 150.00
[  1] local 10.0.0.1 port 5001 connected with 10.0.0.2 port 40001
[  1] WARNING: ack of last datagram failed.
"""
    double = good + "[  2] 0.0000-5.0012 sec  4.10 MBytes  6.88 Mbits/sec   0.020 ms 0/2932 (0%)\n"
    assert count_server_reports(good) == 1
    assert count_server_reports(none) == 0
    assert count_server_reports(double) == 2   # would show as MISMATCH in the progress line
    # The column header and "connected" lines must not count as reports.
    assert count_server_reports("[ ID] Interval       Transfer     Bandwidth        Jitter   Lost/Total Datagrams\n") == 0


def standin_forecasts_if_missing():
    """Selftest only. If the real telecom forecasts.csv does not exist yet (no model has been run), write a
    temporary file whose 'model' column just repeats the last history value, so the trace code can be
    exercised. These stand-in numbers are never saved and are not results."""
    global FORECASTS_CSV
    if TELECOM is None or os.path.exists(FORECASTS_CSV):
        return
    traffic = load_traffic()
    handle, path = tempfile.mkstemp(prefix="standin_forecasts_", suffix=".csv")
    with os.fdopen(handle, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["cell", "start", "step", "true", "model", "persistence", "moving_avg", "parsed", "raw_output"])
        for w in range(1728, 1999, 3):   # the experiment cell's stride-3 test windows
            for k in range(6):
                last_history = traffic[w + WINDOW_HISTORY - 1]
                writer.writerow([TELECOM["cell"], w, k + 1, traffic[w + WINDOW_HISTORY + k], last_history, last_history, 0, True, ""])
    FORECASTS_CSV = path
    print(f"NOTE: {TELECOM_FORECASTS_CSV} does not exist yet, so this selftest uses stand-in forecasts "
          "(last history value repeated). They are not results and are not saved.")


def run_selftest():
    selftest_server_counting()
    print("Server report counting: OK")
    standin_forecasts_if_missing()
    start, crossings, steps = build_trace()
    print(f"Trace: timesteps {start}-{start + TRACE_LEN - 1}, upward crossings of {THRESHOLD:g} "
          f"in true traffic: {crossings}")
    assert len(steps) == TRACE_LEN, f"trace has {len(steps)} steps, expected {TRACE_LEN}"
    assert crossings >= MIN_CROSSINGS
    print("\nWritten values, first 10 steps:")
    print("step  timestep    true  " + "  ".join(f"{m:>8}" for m in MODES))
    for i, s in enumerate(steps[:10]):
        print(f"{i:>4}  {s['timestep']:>8}  {s['true']:>6.2f}  "
              + "  ".join(f"{s['written'][m]:>8.2f}" for m in MODES))
    print("\nSteps with written value above 100 (of 36):")
    for m in MODES:
        print(f"  {m:<9} {sum(s['written'][m] > THRESHOLD for s in steps)}")
    try:
        n = quick_length(steps)
        print(f"\n--quick would run oracle mode for N = {n} steps")
    except SystemExit as error:   # quick_length stops when oracle never crosses; report it and keep checking
        print(f"\n--quick could NOT run on this trace: {error}")
    # --quick --mode reactive: built WITHOUT forecasts.csv, with the written values for the first N steps.
    _, _, quick_steps = build_trace(needs_forecasts=False)
    assert "model" not in quick_steps[0]["written"] and "oracle" not in quick_steps[0]["written"]
    for full, quick in zip(steps, quick_steps):   # the reactive values must match the full trace exactly
        assert full["written"]["reactive"] == quick["written"]["reactive"] and full["timestep"] == quick["timestep"]
    n = quick_length(quick_steps, "reactive")
    print(f"\n--quick --mode reactive (needs no forecasts.csv): N = {n} steps")
    print("step  timestep    true  reactive_written")
    for i, s in enumerate(quick_steps[:n]):
        print(f"{i:>4}  {s['timestep']:>8}  {s['true']:>6.2f}  {s['written']['reactive']:>16.2f}")
    print(f"Forecast file (as the current user): {forecast_path(pwd.getpwuid(os.getuid()).pw_dir)}")
    print("SELFTEST OK")


# ---------------------------------------------------------------- helpers for the real run

def forecast_path(home):
    """Where reroute_trigger.py reads the forecast: ~/fedflow/controller/forecast_value.txt of the POX user."""
    return os.path.join(home, "fedflow", "controller", "forecast_value.txt")


def write_forecast(path, value, uid, gid):
    """Atomic write (temp file + rename), readable by the non-root POX process."""
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".forecast_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(f"{value:.2f}")
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o644)   # mkstemp creates 0600; POX runs as SUDO_USER and must read it
        os.chown(tmp, uid, gid)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    # Wall-clock time right after the rename, millisecond precision.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def port_free(port):
    """True if nothing is listening on the port (we can bind it ourselves)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind(("0.0.0.0", port))
            return True
        except OSError:
            return False


def wait_for(predicate, timeout, interval=0.2):
    """Poll predicate() until it is true or timeout seconds pass."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def read_text(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def count_connected(pox_log_path):
    """Number of 'connected' lines from openflow.of_01 in the POX log."""
    return sum(1 for line in read_text(pox_log_path).splitlines()
               if "openflow.of_01" in line and line.rstrip().endswith("connected"))


def kill_leftover_pox():
    # Called with an argument list, not a shell, so pkill can't match its own shell's command line.
    subprocess.run(["pkill", "-f", "pox/pox.py"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def mn_cleanup():
    subprocess.run(["mn", "-c"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def load_topo_class():
    """Import DualPathTopo from controller/topo.py by path (import only, nothing is modified)."""
    spec = importlib.util.spec_from_file_location("fedflow_topo", TOPO_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.DualPathTopo


def process_alive(pid):
    """True if pid exists and is not a zombie (an exited child the shell hasn't reaped yet)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            # Field 3 is the state; the command name in field 2 is in parentheses.
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def stop_process(pid, timeout=SERVER_EXIT_TIMEOUT):
    """SIGINT (lets iperf/ping print their final lines), then SIGKILL if it doesn't exit."""
    try:
        os.kill(pid, signal.SIGINT)
    except ProcessLookupError:
        return
    if not wait_for(lambda: not process_alive(pid), timeout, 0.05):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


class RunAborted(Exception):
    pass


def run_once(mode, rep, steps, out_dir, user, home, uid, gid, fpath):
    """One run of one mode. Writes raw_<mode>_<rep>_{iperf,server,ping,pox}.txt and _steps.csv."""
    from mininet.net import Mininet
    from mininet.node import RemoteController
    from mininet.link import TCLink

    prefix = os.path.join(out_dir, f"raw_{mode}_{rep}")
    iperf_path, ping_path = prefix + "_iperf.txt", prefix + "_ping.txt"
    pox_path, steps_path = prefix + "_pox.txt", prefix + "_steps.csv"
    server_path = prefix + "_server.txt"
    tag = f"[{mode} rep {rep}]"

    # ---- pre-flight
    kill_leftover_pox()
    mn_cleanup()
    if not wait_for(lambda: port_free(OF_PORT), 10, 0.5):
        raise RunAborted(f"port {OF_PORT} is still in use after cleanup")
    # Start on this mode's own first value, so step 0 doesn't begin with a flip.
    write_forecast(fpath, steps[0]["written"][mode], uid, gid)

    pox_log = open(pox_path, "w", encoding="utf-8")
    # Run POX as the invoking user so its "~" is that user's home, not /root.
    # POX logs to stderr, so stderr goes into the same file.
    pox = subprocess.Popen(
        ["sudo", "-u", user, "-H", "python3", os.path.join(home, "pox", "pox.py"), "reroute_trigger"],
        cwd=os.path.join(home, "pox"), stdout=pox_log, stderr=subprocess.STDOUT,
    )
    net = None
    ping_pid = None
    server_pid = None
    try:
        if not wait_for(lambda: "is up." in read_text(pox_path), POX_UP_TIMEOUT):
            raise RunAborted(f"POX did not report 'is up.' within {POX_UP_TIMEOUT} s (see {pox_path})")

        topo_cls = load_topo_class()
        net = Mininet(topo=topo_cls(), link=TCLink, controller=None, autoSetMacs=False)
        net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=OF_PORT)
        net.start()

        if not wait_for(lambda: count_connected(pox_path) >= N_SWITCHES, CONNECT_TIMEOUT):
            raise RunAborted(f"only {count_connected(pox_path)}/{N_SWITCHES} switches connected "
                             f"within {CONNECT_TIMEOUT} s (see {pox_path})")

        h1, h2 = net.get("h1"), net.get("h2")
        # Static ARP with the hosts' real MACs; the controller never floods, so ARP can't resolve.
        h1.cmd(f"arp -s {h2.IP()} {h2.MAC()}")
        h2.cmd(f"arp -s {h1.IP()} {h1.MAC()}")

        test = h1.cmd(f"ping -c 3 -W 1 {H2_IP}")
        if " 0 received" in test:
            raise RunAborted(f"test ping h1 -> h2 got no replies:\n{test}")
        print(f"{tag} test ping ok: {[l for l in test.splitlines() if 'received' in l][0].strip()}")

        # Continuous ping for the whole run; -D adds a timestamp to each reply line.
        ping_pid = int(h1.cmd(f"ping -D -i 0.2 {H2_IP} > {ping_path} 2>&1 & echo $!").strip().split()[-1])
        open(server_path, "w").close()  # fresh server file for this run

        run_start = time.time()
        with open(steps_path, "w", newline="", encoding="utf-8") as sf, \
                open(iperf_path, "w", encoding="utf-8") as iperf_file:
            writer = csv.writer(sf)
            writer.writerow(["step", "timestep", "true_value", "value_written", "time_written"])
            for i, s in enumerate(steps):
                value = s["written"][mode]
                rate = s["true"] / 10.0   # v traffic units -> v/10 Mbps
                marker = f"### step {i} timestep {s['timestep']} rate {rate:.3f}M written {value:.2f}\n"

                # Fresh iperf server for this step only (see "Known effects"); its own output
                # is appended under this step's marker. No -i, so it prints one final report.
                with open(server_path, "a", encoding="utf-8") as srv:
                    srv.write(marker)
                server_pid = int(h1.cmd(f"iperf -u -s >> {server_path} 2>&1 & echo $!").strip().split()[-1])
                time.sleep(SERVER_START_WAIT)

                # Write the forecast, then start this step's traffic immediately.
                written_at = write_forecast(fpath, value, uid, gid)
                writer.writerow([i, s["timestep"], f"{s['true']:.2f}", f"{value:.2f}", written_at])
                sf.flush()
                out = h2.cmd(f"iperf -u -c {H1_IP} -b {rate:.3f}M -t {STEP_SECONDS}")
                iperf_file.write(marker)
                iperf_file.write(out)
                if not out.endswith("\n"):
                    iperf_file.write("\n")
                iperf_file.flush()

                time.sleep(SERVER_STOP_WAIT)
                stop_process(server_pid)
                server_pid = None
                # Count report lines in this step's part of the server file (after the last marker).
                server_reports = count_server_reports(read_text(server_path).rsplit(marker, 1)[-1])
                server_flag = {0: "no report", 1: "yes"}.get(server_reports, f"MISMATCH({server_reports})")
                client_flag = "yes" if "Server Report" in out else "no report"
                print(f"{tag} step {i + 1}/{len(steps)} t={s['timestep']} true={s['true']:.2f} "
                      f"wrote={value:.2f} rate={rate:.2f}M server={server_flag} client={client_flag} "
                      f"elapsed={time.time() - run_start:.0f}s")
    finally:
        # ---- teardown, always
        if server_pid:
            stop_process(server_pid)
        if ping_pid:
            stop_process(ping_pid)  # SIGINT makes ping print its statistics
        if net is not None:
            net.get("h1").cmd("kill %iperf 2>/dev/null")
            net.stop()
        pox.terminate()
        try:
            pox.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pox.kill()
            pox.wait()
        kill_leftover_pox()
        pox_log.close()
        mn_cleanup()


def chown_tree(path, uid, gid):
    for root, dirs, files in os.walk(path):
        os.chown(root, uid, gid)
        for name in files:
            os.chown(os.path.join(root, name), uid, gid)


def run_real(quick, mode="oracle"):
    if os.geteuid() != 0:
        sys.exit("ERROR: run this with sudo (only --selftest works without it).")
    user = os.environ.get("SUDO_USER")
    if not user or user == "root":
        sys.exit("ERROR: SUDO_USER is not set to a normal user. Run as: sudo python3 experiments/run_experiment.py")
    pw = pwd.getpwnam(user)
    home, uid, gid = pw.pw_dir, pw.pw_uid, pw.pw_gid
    fpath = forecast_path(home)
    if not os.path.isdir(os.path.dirname(fpath)):
        sys.exit(f"ERROR: {os.path.dirname(fpath)} does not exist; the controller reads {fpath}.")
    if not os.path.exists(os.path.join(home, "pox", "pox.py")):
        sys.exit(f"ERROR: {home}/pox/pox.py not found.")
    print(f"Forecast file used (the controller's ~ for {user}): {fpath}")

    # A quick run of static, reactive or alt needs no forecasts.csv; model and oracle (and the full run) do.
    needs_forecasts = (not quick) or mode in ("model", "oracle")
    start, crossings, steps = build_trace(needs_forecasts)
    print(f"Trace: timesteps {start}-{start + TRACE_LEN - 1}, upward crossings of {THRESHOLD:g}: {crossings}")

    if quick:
        n = quick_length(steps, mode)
        print(f"Quick run: {mode} mode, 1 rep, N = {n} steps")
        out_dir, plan, steps = QUICK_DIR, [(mode, 1)], steps[:n]
    else:
        out_dir = OUT_DIR
        # Round-robin over modes within each rep, so slow drift in the machine affects all modes alike.
        plan = [(m, r) for r in range(1, REPS + 1) for m in MODES]
        existing = [f for f in os.listdir(out_dir) if f.startswith("raw_")] if os.path.isdir(out_dir) else []
        if existing:
            sys.exit(f"ERROR: {out_dir} already has {len(existing)} raw_* files. Move them away first; "
                     "raw results are never overwritten.")
    os.makedirs(out_dir, exist_ok=True)
    est = len(plan) * (len(steps) * (STEP_SECONDS + 1) + 30) / 60
    print(f"Runs: {len(plan)} (about {est:.0f} min). Output: {out_dir}")

    failed = []
    t0 = time.time()
    try:
        for k, (mode, rep) in enumerate(plan, start=1):
            print(f"\n=== run {k}/{len(plan)}: mode={mode} rep={rep} (elapsed {(time.time() - t0) / 60:.1f} min)")
            try:
                run_once(mode, rep, steps, out_dir, user, home, uid, gid, fpath)
            except RunAborted as error:
                print(f"ABORTED run mode={mode} rep={rep}: {error}")
                failed.append((mode, rep, str(error)))
    finally:
        chown_tree(out_dir if quick else OUT_DIR, uid, gid)

    print(f"\nDone in {(time.time() - t0) / 60:.1f} min. Raw files in {out_dir}")
    if failed:
        print(f"{len(failed)} run(s) aborted:")
        for mode, rep, error in failed:
            print(f"  {mode} rep {rep}: {error.splitlines()[0]}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="FedFlow-LLM reactive vs proactive experiment")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--selftest", action="store_true", help="build the trace only; no Mininet, no sudo")
    group.add_argument("--quick", action="store_true",
                       help="1 rep of one mode (--mode, default oracle), short trace -> <output folder>/quick/")
    parser.add_argument("--dataset", choices=["synthetic", "telecom"], default="synthetic",
                        help="synthetic (default, unchanged) or the Telecom Italia experiment cell")
    parser.add_argument("--cell", type=int, default=None,
                        help="with --dataset telecom: experiment cell square ID, 7511 (default, primary) or 8336 (second)")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="with --quick: the mode to run (default oracle, as before)")
    parser.add_argument("--forecasts", default=None,
                        help="with --dataset telecom: forecasts CSV (default results/final_telecom/outputs/forecasts.csv, the v1 run)")
    parser.add_argument("--out-root", default=None,
                        help="with --dataset telecom: output root (default results/experiment_telecom); files go to <root>/<square_id>/")
    args = parser.parse_args()
    if (args.forecasts is not None or args.out_root is not None) and args.dataset != "telecom":
        parser.error("--forecasts and --out-root only apply with --dataset telecom")
    if args.forecasts is not None and not os.path.isfile(args.forecasts):
        parser.error(f"--forecasts {args.forecasts} is not a file")
    # Another run's forecasts must never land in (or overwrite) the v1 results folder.
    if (args.forecasts is not None and os.path.abspath(args.forecasts) != TELECOM_FORECASTS_CSV
            and (args.out_root is None or os.path.abspath(args.out_root) == TELECOM_OUT_DIR)):
        parser.error("a non-default --forecasts needs its own --out-root (not results/experiment_telecom)")
    if args.mode is not None and not args.quick:
        parser.error("--mode only applies with --quick")
    if args.cell is not None and args.dataset != "telecom":
        parser.error("--cell only applies with --dataset telecom")
    if args.dataset == "telecom":
        try:
            use_telecom(args.cell, args.forecasts, args.out_root)
        except ValueError as error:
            parser.error(str(error))
    if args.selftest:
        run_selftest()
    else:
        run_real(args.quick, args.mode or "oracle")


if __name__ == "__main__":
    main()
