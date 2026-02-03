from __future__ import annotations

import csv
from dataclasses import dataclass
import os
import threading
import time
from typing import Callable


@dataclass
class LogSample:
    ts: float
    vx: float
    vy: float
    yaw_rate: float
    source: str
    left_x: float | None
    left_y: float | None
    right_x: float | None
    right_y: float | None
    left_vis: float | None
    right_vis: float | None


class CsvLogger:
    def __init__(self, log_hz: float, out_dir: str) -> None:
        self._log_hz = log_hz
        self._out_dir = out_dir
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._sample_fn: Callable[[], LogSample] | None = None
        self._file = None
        self._writer = None

    def start(self, sample_fn: Callable[[], LogSample]) -> str:
        os.makedirs(self._out_dir, exist_ok=True)
        path = os.path.join(self._out_dir, f"session_{int(time.time())}.csv")
        self._file = open(path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "ts",
                "vx",
                "vy",
                "yaw_rate",
                "source",
                "left_x",
                "left_y",
                "right_x",
                "right_y",
                "left_vis",
                "right_vis",
            ]
        )
        self._sample_fn = sample_fn
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return path

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        if self._file:
            self._file.flush()
            self._file.close()

    def _run(self) -> None:
        assert self._sample_fn is not None
        interval = 0.0 if self._log_hz <= 0 else 1.0 / self._log_hz
        next_ts = time.time()
        while not self._stop.is_set():
            sample = self._sample_fn()
            if self._writer:
                self._writer.writerow(
                    [
                        f"{sample.ts:.6f}",
                        sample.vx,
                        sample.vy,
                        sample.yaw_rate,
                        sample.source,
                        sample.left_x,
                        sample.left_y,
                        sample.right_x,
                        sample.right_y,
                        sample.left_vis,
                        sample.right_vis,
                    ]
                )
            if interval <= 0:
                continue
            next_ts += interval
            sleep_s = next_ts - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_ts = time.time()

