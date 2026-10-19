# Experiment: proactive vs reactive routing

The trace is held-out timesteps 1452 to 1487 (36 steps of 5 s), the slice with the most upward crossings of 100 in the true traffic: 4. Each step value v is sent as v/10 Mbps of UDP from h2 to h1, so the threshold of 100 traffic units equals the 10 Mbps primary link. The controller routes on the primary path (via s3) when the written value is at or below 100 and on the 50 Mbps alternate path (via s4) when it is above. Five modes write a different value each step: **static** (0, primary always), **reactive** (true value of the previous step), **model** (peak of the federated model's 6 predictions), **oracle** (peak of the 6 true values over the same window) and **alt** (1000, alternate always, reference only). There are 3 reps per mode.

## Results

Each cell is the mean of the 3 reps, with (min-max).

| Mode (reps) | Ping mean ms | Ping p95 ms | UDP loss % (server side) | delivered_vs_alt_pct | FLIP count | Fraction of steps written above 100 |
|---|---|---|---|---|---|---|
| static (3) | 43.13 (42.22-43.83) | 107.33 (107.00-108.00) | 0.00 (0.00-0.00) | 83.23 (82.99-83.43) | 0 (0-0) | 0.0000 (0.0000-0.0000) |
| reactive (3) | 9.75 (9.71-9.84) | 85.61 (85.30-85.80) | 0.00 (0.00-0.00) | 98.39 (98.36-98.44) | 7 (7-7) | 0.6111 (0.6111-0.6111) |
| model (3) | 24.84 (24.77-24.95) | 103.38 (103.00-104.00) | 0.00 (0.00-0.00) | 88.53 (88.19-89.03) | 3 (3-3) | 0.3333 (0.3333-0.3333) |
| oracle (3) | 0.11 (0.11-0.12) | 0.10 (0.10-0.10) | 0.00 (0.00-0.00) | 100.00 (100.00-100.00) | 1 (1-1) | 0.9167 (0.9167-0.9167) |
| alt (3) | 0.11 (0.11-0.12) | 0.10 (0.10-0.11) | 0.00 (0.00-0.00) | 100.00 (100.00-100.00) | 0 (0-0) | 1.0000 (1.0000-1.0000) |

delivered_vs_alt_pct = 100 × server_udp_total of this run / mean(server_udp_total of alt reps 1, 2 and 3). server_udp_total is the sum over all 36 steps of the Total datagrams in the iperf server's own reports. The alt totals are 176872, 176873 and 176874.

## What this shows

- (a) Delay ordering, by ping mean and by ping p95: oracle and alt lowest (0.11 ms mean, 0.10 ms p95 for both), then reactive (9.75 / 85.61), then model (24.84 / 103.38), then static (43.13 / 107.33).
- (b) The model beat static but not reactive: ping mean 24.84 ms vs 43.13 (static) and 9.75 (reactive); delivered_vs_alt_pct 88.53 vs 83.23 and 98.39.
- (c) Oracle is the ceiling: it matches alt on ping (0.11 ms mean, 0.10 ms p95) and on delivered_vs_alt_pct (100.00). It gets there by sitting on the alternate path for most steps: it wrote a value above 100 on 0.9167 of steps, with 1 FLIP per run.

## What this does not show

- (a) UDP loss is 0% in every run because the 10 Mbps link slows the sender instead of overflowing, mechanism not verified (see static1_steps_check.txt).
- (b) All 3 reps replay the same 36-step trace, so reps show repeatability, not variety.
- (c) Our reactive mode receives the exact previous-step value and the controller polls every 1 s, so it is a faster reactive baseline than a typical real controller.
- (d) The model was trained on synthetic data and flagged 1 of 10 onsets (results/final/SUMMARY.md).
- (e) delivered_vs_alt_pct was added after seeing that loss was 0 in every run, so it is reported alongside the planned metrics, not instead of them.

Files (all in this folder): `summary.csv` (one row per mode and rep, source of every number above), `summary.png` (loss, p95 ping and delivered_vs_alt_pct per mode), `static1_steps_check.txt` (per-step static rep 1 vs alt rep 1 check), and the `raw_<mode>_<rep>_*` files (`_steps.csv`, `_iperf.txt`, `_server.txt`, `_ping.txt`, `_pox.txt`) that summary.csv is built from.
