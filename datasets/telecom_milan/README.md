# Telecom Italia (Milan) dataset

An alternative to the synthetic traffic series. Six squares of the Milan grid, each turned into a
2016-slot series of internet load (14 days at 10-minute slots), one cell per federated client.

## Source

- Dataset: "Telecommunications - SMS, Call, Internet - MI", Harvard Dataverse,
  DOI [10.7910/DVN/EGZHFV](https://doi.org/10.7910/DVN/EGZHFV). The Dataverse record lists the author as
  "Telecom Italia", a publication date of 2015-05-14, and terms of use that say the data is released under the ODbL 1.0 license.
- Paper: Barlacchi G, De Nadai M, Larcher R, Casella A, Chitic C, Torrisi G, Antonelli F, Vespignani A,
  Pentland A, Lepri B. "A multi-source dataset of urban life in the city of Milan and the Province of Trentino".
  *Scientific Data* 2, 150055 (published 27 October 2015). https://www.nature.com/articles/sdata201555,
  DOI 10.1038/sdata.2015.55. The authors, title, journal and date above were read from that page on the day this
  was written, not typed from memory.
- The data is from **November 2013**: the 14 files `sms-call-internet-mi-2013-11-01.txt` to `...-14.txt`
  (1 to 14 November 2013). Only the internet column is used.
- The raw files (about 4.6 GB) stay in `~/fedflow_data/telecom_raw/` and are never copied into this repository.

## What is in this folder

| File | What it is |
|---|---|
| `cells.csv` | `timestep` (0-2015) plus the 6 scaled series, one column per square ID |
| `cells_meta.json` | chosen squares, raw training means, k, split, slice rule, experiment cell and slice |
| `prep_report.txt` | every number behind the cell choice (written by `training/prepare_telecom.py`) |
| `check_report.txt` | read-only look at the cells (written by `training/telecom_check.py`) |
| `token_audit.txt` | token lengths of the examples (written by `training/token_audit.py`) |

## Locked decisions

These were fixed before any model was trained and are never changed after seeing a model result.

1. **Columns.** Files are tab-separated: square_id, timestamp_ms, country_code, sms_in, sms_out, call_in, call_out, internet. Only `internet` is used.
2. **Timeline.** slot = (timestamp_ms - 1383260400000) / 600000. Slot 0 is 1 Nov 2013 00:00 Milan time; there are 144 slots per file and 2016 in all. The run aborts if a row falls outside its own file's 144 slots (or off the 10-minute grid).
3. **Per square and slot.** Sum the internet values over all country codes. A blank counts as 0. A slot counts as non-blank only if at least one row had a numeric internet value.
4. **Split (chronological).** Training = slots 0-1439 (days 0-9), validation = 1440-1727 (days 10-11), test = 1728-2015 (days 12-13). Day 0 is Friday 1 Nov. The holiday and the weekends stay in training.
5. **Cell filter.** Keep squares with at least 95% non-blank slots over all 2016 slots. Abort if fewer than 100 pass.
6. **Cell choice.** For each passing square, take the mean over training slots only and rank the squares. Percentile of a square = 100 x (passing squares with a strictly lower mean) / (N - 1). For the targets 40, 46, 52, 58, 64 and 70 (in that order) take the nearest unused square; on ties the lower square ID.
7. **Scale.** One factor k = 90 / median of the 6 chosen cells' raw training means, applied to every slot and rounded to 2 decimals. 100 stands for the 10 Mbps primary link, as in the synthetic data.
8. **Experiment cell.** Scaled TRUE traffic only. For each chosen cell, find the 36-step slice with the most upward crossings of 100 (value <= 100 followed by a value > 100). **Slice starts are searched from 1740 to 1980 inclusive, the same range for every cell.** The range starts at 1740 because the first test window starts at slot 1728, so its history ends at 1739 and trace steps before 1740 would have no model forecast. The experiment cell is the one with the most crossings (ties: earliest slice start, then lower square ID). If the best has fewer than 3 crossings the run aborts. k and the cells are never changed to get more crossings.
9. **The slice is chosen from TRUE traffic only, never from model outputs.** The range and rule are saved in `cells_meta.json` (`slice_rule`), and `experiments/run_experiment.py --dataset telecom` recomputes the slice and stops if it differs.

## What was chosen

Numbers below are copied from `prep_report.txt`.

- 9,998 of 10,000 squares passed the 95% filter.
- Chosen squares (cell order 0-5): 9721, 6997, 8336, 7511, 8062, 7584. k = 2.3306204138.
- Experiment cell picked by the locked rule (primary): square 7511, slice start 1837, 8 upward crossings.
- Second experiment cell, declared before any model result: square 8336, slice start 1960, 8 upward crossings.

## Experiment cells

Both **7511** and **8336** are experiment cells, and results are reported for both, whatever they show.
The locked rule (decision 8) still picks 7511 as the primary cell; 8336 is an addition that does not change
any locked rule, cell, k or split. Both are listed in `cells_meta.json` under `experiment_cells`. Each has its own
36-step slice, found by the same rule on true traffic only. Experiment output goes to
`results/experiment_telecom/<square_id>/`, chosen with `--cell <square_id>` (default 7511).

## How the data is used

- `python generate_data.py --dataset telecom` (from `training/`) writes 200 training examples (12 history + 6 future), `val_windows.json` and `test_windows.json` to `training/data/telecom/`, never over the synthetic files. Example j belongs to cell j % 6, so client i (which takes `examples[i::6]`) trains only on cell i.
- Test windows: both experiment cells (7511 and 8336) every 3 slots, the other four cells every 6. Validation windows: every cell every 6 slots.
- `notebooks/fedflow_colab_telecom.ipynb` runs it on Colab.

## Known limits

Numbers are quoted from `check_report.txt` and `prep_report.txt`.

- **One month, one year.** The data covers 1 to 14 November 2013 only, and training uses just the first 10 of those days.
- **Holiday and weekends.** 1 Nov is a public holiday in Italy, and 2-3 and 9-10 Nov are weekends, so 5 of the 10 training days are non-working days. The validation days (11-12 Nov) and test days (13-14 Nov) are working days. For square 7511 the mean scaled value is 85.39 on non-working training days, 108.38 on working training days, 104.63 on validation and 107.16 on test.
- **Onset recall is reportable only with enough onset windows.** An onset window has a last history value <= 100 and a future maximum > 100. At stride 3, squares 9721 (13), 6997 (18), 8336 (14) and 7511 (14) have at least 10 onset windows; squares 8062 (9) and 7584 (5) do not, so onset recall is not reported for those two cells. At stride 6 no single cell reaches 10 (the most is 9, square 6997), so a per-cell recall is not reported there either; pooled over all six cells at stride 6 there are 37 onset windows.
- **Windows overlap.** Windows are 18 slots long and start every 3 or 6 slots, so neighbouring windows share slots and the onset windows are not independent samples.
- **Square 7584 is above 100 most of the time.** It is above 100 in 0.7217 of all 2016 slots (0.7076 of training slots, 0.7604 of validation, 0.7535 of test), about three quarters, so it has few chances to cross upward: 5 onset windows at stride 3.
- **One spike drives one crossing in the experiment slice.** In square 7511, slots 1837-1872, the value at slot 1842 is 214.08. The crossing at slot 1840 has size 114.08 (highest of the next 6 values minus 100) because of that spike; the other 7 crossings have sizes of at most 21.44.
- **The primary cell's oracle is above 100 at every step.** In the 36-step trace of square 7511 the oracle value (the peak of the next 6 true values) is above 100 at 36 of 36 steps (`check_report.txt`, section 6), so the oracle never has an upward crossing and, because the controller sends traffic to the alternate path whenever the written value is above 100, it behaves as always-alternate. The reactive value is above 100 at 22 of 36 steps. The second cell, 8336, shows the same for the oracle (36 of 36 above 100, no upward crossing; reactive 28 of 36), so `run_experiment.py --quick`, which needs an oracle upward crossing, cannot run on either cell.
- **Alternate-path usage is reported next to delay.** For every mode, each row of `summary.csv` holds `frac_written_above_100` (the share of steps whose written value was above 100, so sent to the alternate path) in the same row as `ping_mean_ms` and `ping_p95_ms`, so a mode's delay is never read without how much of the alternate path it used.
- **Scaled values are unitless load, not measured Mbps.** They are the raw internet activity of the dataset multiplied by k. Treating 100 as the 10 Mbps primary link is a modelling choice of this project.
- **Token length.** The longest telecom example is 256 tokens (`token_audit.txt`), exactly `client.py`'s default limit of 256, with no headroom; none is above it.
