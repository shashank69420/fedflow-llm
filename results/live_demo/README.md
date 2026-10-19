# Live demo: forecast-driven reroute

This demo fed two forecast values into the POX controller and watched the route change in Mininet. The values are the model peaks of two held-out windows, taken from the final federated model's saved outputs (see `chosen_windows.txt`): 70.35 for the calm window starting at timestep 1440 and 100.84 for the onset window starting at timestep 1608. Each value was written to `forecast_value.txt`. The `reroute_trigger` POX controller polls that file and sends traffic on the primary path (via s3) when the forecast is at or below the threshold of 100.0, and on the alternate path (via s4) when it is above. In Mininet, `h1 ping h2` generated the traffic, and `ovs-ofctl dump-flows` on s3 and s4 showed which path it took.

## Events (in time order)

| Step | Forecast written | POX log line (timestamp) | Switch with flows afterwards | Ping loss |
|---|---|---|---|---|
| 1. Calm | 70.35 | No flip. Startup line: `Starting on primary path (forecast 70.35, threshold 100.0)`. Timestamp: not recorded | s3 | 0% (5/5 received), 0% (3/3 received) |
| 2. Onset | 100.84 | `00:18:56.239 FLIP primary -> alternate (forecast 100.84)` | s4 | 0% (3/3 received) |
| 3. Calm again | 70.35 | `00:21:12.224 FLIP alternate -> primary (forecast 70.35)` | s3 | 0% (3/3 received) |

The Mininet log has no timestamps. Each ping is placed in a step by its order in the log and by which switch's flow dump had entries right after it; the other switch's dump was empty each time.

## What this shows

- The route flips to the alternate path only when the forecast is above 100.
- The route flips back to the primary path when the forecast is below 100.
- There was 0% packet loss in all 4 ping tests.

## What this does not show

- (a) reaction latency was not measured, because the time each value was written was not recorded;
- (b) the forecast values are saved outputs from the Colab run, not live inference on this laptop;
- (c) the 100.84 forecast is only 0.84 above the threshold, and it is the one onset window the model flagged out of 10 (see results/final/SUMMARY.md), so this demonstrates the mechanism, not forecasting skill.
- (d) the ping tests ran without background load, so they show connectivity survives the reroute, not reduced delay or loss under congestion.

Sources: `pox_log.txt`, `mininet_log.txt`, `chosen_windows.txt` (all in this folder).
