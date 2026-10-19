"""
Write a forecast value for reroute_trigger.py to pick up.

Usage:  python3 write_forecast.py 118.4
"""
import math
import os
import sys
import tempfile

# Same absolute path reroute_trigger.py reads, so the folder you run this from doesn't matter.
FORECAST_FILE = os.path.expanduser("~/fedflow/controller/forecast_value.txt")


def parse_value(args):
    """Return the single numeric argument as a float, or exit with an error message."""
    if len(args) != 1:
        sys.exit("ERROR: give exactly one forecast value.\nUsage: python3 write_forecast.py 118.4")
    try:
        value = float(args[0])
    except ValueError:
        sys.exit(f"ERROR: '{args[0]}' is not a number.\nUsage: python3 write_forecast.py 118.4")
    if not math.isfinite(value):  # float() accepts "nan"/"inf"
        sys.exit(f"ERROR: '{args[0]}' is not a finite number.")
    return value


def write_atomically(path, text):
    """Write to a temp file in the same folder, then rename it over the target.
    The rename is atomic, so a reader sees either the old file or the new one, never half."""
    folder = os.path.dirname(path)
    fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=".forecast_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())  # make sure the bytes are on disk before the rename
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)  # don't leave temp files behind on failure
        raise


if __name__ == "__main__":
    forecast = parse_value(sys.argv[1:])
    write_atomically(FORECAST_FILE, str(forecast))
    print(f"[FedFlow] Wrote forecast value {forecast} to {FORECAST_FILE}")
