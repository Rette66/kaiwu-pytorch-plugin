"""Lightweight direct-CIM optimizer adapter for DPLM Q-Diffusion examples."""

from __future__ import annotations

import datetime
import importlib
import os
import time
from typing import Any

import numpy as np


DEFAULT_USER_ID = "157297654894690562"
DEFAULT_SDK_CODE = "vbaJxXUrdv0Kiqlc6ytoza9fyiZhk6"
DEFAULT_PROJECT_NO = "26078472"


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
        refresh_task_name: bool = False,
        user_id: str | None = None,
        sdk_code: str | None = None,
        project_no: str | None = None,
        init_license: bool = False,
        pass_credentials: bool = False,
        **optimizer_kwargs: Any,
    ) -> None:
        import kaiwu as kw

        self.kw = kw
        self.cim = importlib.import_module("kaiwu.cim")
        self.user_id = (
            user_id
            or os.getenv("DPLM_DIRECT_CIM_USER_ID")
            or DEFAULT_USER_ID
        )
        self.sdk_code = (
            sdk_code
            or os.getenv("DPLM_DIRECT_CIM_SDK_CODE")
            or DEFAULT_SDK_CODE
        )
        self.project_no = (
            project_no
            or os.getenv("DPLM_DIRECT_CIM_PROJECT_NO")
            or DEFAULT_PROJECT_NO
        )
        self.init_license = init_license
        self.pass_credentials = pass_credentials
        self.task_name = task_name
        self.wait = wait
        self.interval = interval
        self.task_mode = task_mode
        self.sample_number = sample_number
        self.tmp_dir = tmp_dir
        self.refresh_seconds = refresh_hours * 3600
        self.refresh_task_name = refresh_task_name
        self.optimizer_kwargs = dict(optimizer_kwargs)
        self.worker = None
        self.worker_created_ts: float | None = None
        self.worker_created_at: datetime.datetime | None = None
        self._refresh_worker(force=True, reason="init")

    def _init_license_if_available(self) -> None:
        """Initializes Kaiwu license when the runtime exposes that helper."""
        if not self.init_license or not self.user_id or not self.sdk_code:
            return
        license_module = getattr(self.kw, "license", None)
        init = getattr(license_module, "init", None)
        if callable(init):
            init(self.user_id, self.sdk_code)

    def _create_worker(self):
        self.kw.common.CheckpointManager.save_dir = self.tmp_dir
        # The direct-hardware path follows the standalone direct_cim.py script:
        # credentials are passed to CIMOptimizer, not pre-validated via license.
        self._init_license_if_available()
        kwargs = {
            "task_name": self.task_name,
            "wait": self.wait,
            "interval": self.interval,
            "task_mode": self.task_mode,
            "sample_number": self.sample_number,
            "project_no": self.project_no,
        }
        if self.pass_credentials:
            kwargs["user_id"] = self.user_id
            kwargs["sdk_code"] = self.sdk_code
        extra_kwargs = {
            key: value
            for key, value in self.optimizer_kwargs.items()
            if value is not None
        }
        if not self.pass_credentials:
            extra_kwargs.pop("user_id", None)
            extra_kwargs.pop("sdk_code", None)
        kwargs.update(extra_kwargs)
        logged_keys = sorted(key for key in kwargs if key != "sdk_code")
        print(
            "[DIRECT_CIM_CONFIG] "
            f"project_no={kwargs.get('project_no')} "
            f"task_mode={kwargs.get('task_mode')} sample_number={kwargs.get('sample_number')} "
            f"wait={kwargs.get('wait')} interval={kwargs.get('interval')} "
            f"pass_credentials={self.pass_credentials} keys={logged_keys}",
            flush=True,
        )
        return self.cim.CIMOptimizer(**kwargs)

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
                if self.refresh_task_name and hasattr(self.worker, "task_name"):
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
