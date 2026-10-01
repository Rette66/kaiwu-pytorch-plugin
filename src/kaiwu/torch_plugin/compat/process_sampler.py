"""Run a Kaiwu SDK sampler in a separate Python environment."""

import json
import subprocess
from pathlib import Path

import numpy as np

from ._solver_protocol import receive_array, send_array
from ..usage_stats import is_usage_stats_enabled


class KaiwuProcessSampler:
    """Expose a Python 3.10 Kaiwu SA solver through the usual ``solve`` method.

    The model process may use Python 3.12 and FlagOS. Only NumPy Ising matrices
    and solutions cross the process boundary.
    """

    def __init__(self, python_executable: str, **solver_kwargs) -> None:
        worker = Path(__file__).with_name("_kaiwu_solver_worker.py")
        self._process = subprocess.Popen(  # pylint: disable=consider-using-with
            [python_executable, "-u", str(worker), json.dumps(solver_kwargs)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
        )

    def solve(self, ising_matrix: np.ndarray) -> np.ndarray:
        """Return Kaiwu SA solutions for an Ising matrix."""
        try:
            self._process.stdin.write(b"1" if is_usage_stats_enabled() else b"0")
            send_array(self._process.stdin, ising_matrix)
            return receive_array(self._process.stdout)
        except (OSError, EOFError, ValueError) as exc:
            raise RuntimeError("Kaiwu solver process failed; see its stderr") from exc

    def close(self) -> None:
        """Stop the solver process after its current request."""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        self._process.wait()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
