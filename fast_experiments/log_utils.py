#!/usr/bin/env python3
"""Lightweight logging utility that writes to both console and a log file.

Mirrors the file-output style used in reproduction_xpu.py (json, print, io).
"""

import io
import json
import os
from datetime import datetime


def setup_logger(log_dir="./logs", name="run"):
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.jsonl")
    log_file = io.open(log_path, "a", encoding="utf-8")
    print(f"[logger] writing -> {log_path}")
    return LogWriter(log_file)


class LogWriter:
    def __init__(self, file_handle):
        self._fh = file_handle

    def write(self, entry):
        line = json.dumps(entry, ensure_ascii=False)
        self._fh.write(line + "\n")
        self._fh.flush()
        print(line)

    def write_separator(self, char="=", width=60):
        sep = char * width
        self._fh.write(f"# {sep}\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
        print("[logger] closed")
