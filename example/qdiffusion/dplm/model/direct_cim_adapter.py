"""Lightweight direct-CIM optimizer adapter for DPLM Q-Diffusion examples."""

from __future__ import annotations

import datetime
import time
from typing import Any

import numpy as np


class DirectCIMOptimizer:
    """Small solve(ising_matrix) adapter around ``kaiwu.cim.CIMOptimizer``.

    The Q-Diffusion BM path only needs one method: ``solve(ising_matrix)``. This
    adapter keeps direct-hardware retries and token-refresh handling separate
    from the old standalone DVAE/RBM training script.
    """

    def __init__(
        self,
        *,
        task_name: str = "qdiffusion_direct_cim",
        wait: bool = True,
        interval: float = 0.005,
        task_mode: str = "sample",
        sample_number: int = 10,
        tmp_dir: str = "./tmp",
        refresh_hours: float = 20.0,
        **optimizer_kwargs: Any,
    ) -> None:
        import kaiwu as kw

        self.kw = kw
        self.task_name = task_name
        self.wait = wait
        self.interval = interval
        self.task_mode = task_mode
        self.sample_number = sample_number
        self.tmp_dir = tmp_dir
        self.refresh_seconds = refresh_hours * 3600
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.worker = None
        self.worker_created_ts: float | None = None
        self.worker_created_at: datetime.datetime | None = None
        self._refresh_worker(force=True, reason="init")

    def _create_worker(self):
        self.kw.common.CheckpointManager.save_dir = self.tmp_dir
        kwargs = {
            "task_name": self.task_name,
            "wait": self.wait,
            "interval": self.interval,
            "task_mode": self.task_mode,
            "sample_number": self.sample_number,
        }
        kwargs.update(
            {
                key: value
                for key, value in self.optimizer_kwargs.items()
                if value is not None
            }
        )
        return self.kw.cim.CIMOptimizer(**kwargs)

    def _refresh_worker(self, *, force: bool = False, reason: str = "scheduled") -> None:
        now_ts = time.monotonic()
        need_refresh = (
            force
            or self.worker is None
            or self.worker_created_ts is None
            or (
                self.refresh_seconds > 0
                and now_ts - self.worker_created_ts >= self.refresh_seconds
            )
        )
        if not need_refresh:
            return

        old_age = "none"
        if self.worker_created_ts is not None:
            old_age = f"{(now_ts - self.worker_created_ts) / 3600:.2f}h"
        self.worker = self._create_worker()
        self.worker_created_ts = time.monotonic()
        self.worker_created_at = datetime.datetime.now()
        print(
            "[DIRECT_CIM_REFRESH] "
            f"reason={reason} old_worker_age={old_age} "
            f"new_created_at={self.worker_created_at:%Y-%m-%d %H:%M:%S}",
            flush=True,
        )

    def solve(self, ising_matrix):
        """Solves one Ising matrix on direct CIM and returns solver samples."""
        output = None
        while output is None:
            self._refresh_worker()
            try:
                if hasattr(self.worker, "task_name"):
                    self.worker.task_name = f"{self.task_name}_{int(time.time())}"
                output = self.worker.solve(ising_matrix)
            except Exception as exc:
                message = repr(exc)
                lower_message = message.lower()
                if "token" in lower_message and "expir" in lower_message:
                    self._refresh_worker(force=True, reason="token_expired")
                    time.sleep(1)
                    continue
                if "nonetype" in lower_message or "failed to retrieve task" in lower_message:
                    time.sleep(1)
                    continue
                print(f"[DIRECT_CIM_RETRY] {message}", flush=True)
                time.sleep(5)

        return np.asarray(list(output), dtype=np.float32)
