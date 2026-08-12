"""
scripts/_log_utils.py
=====================
Per-run log file helper. Tees stdout+stderr into a log file inside the
run directory while still printing to the console, so every run leaves a
complete, reviewable log for debugging (including tracebacks).

Usage:
    from scripts._log_utils import start_run_log
    start_run_log(run_dir, "train.log")
    # ... rest of the process output also lands in <run_dir>/train.log
"""

import os
import sys
import time


class _Tee:
    """Duplicate writes to two streams (console + log file)."""

    def __init__(self, stream, log_file):
        self._stream = stream
        self._file = log_file

    def write(self, data):
        self._stream.write(data)
        self._file.write(data)
        self._file.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._file.flush()

    def fileno(self):
        return self._stream.fileno()


def start_run_log(run_dir, name="train.log"):
    """Tee stdout+stderr into ``<run_dir>/<name>`` for the rest of the process.

    Appends a timestamped header (runs sharing a run_id stay distinguishable)
    and keeps echoing to the console. Returns the log file path.
    """
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, name)
    f = open(path, "a")
    f.write("\n" + "=" * 60 + "\n")
    f.write(f"run started {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 60 + "\n")
    f.flush()
    sys.stdout = _Tee(sys.stdout, f)
    sys.stderr = _Tee(sys.stderr, f)
    return path
