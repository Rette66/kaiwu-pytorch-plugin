"""Python 3.10 worker for :class:`KaiwuProcessSampler`."""

import json
import sys
from contextlib import redirect_stdout

# The worker runs as a script, which places this sibling module on sys.path.
from _solver_protocol import receive_array, send_array  # pylint: disable=import-error
from kaiwu.classical import SimulatedAnnealingOptimizer

try:
    from kaiwu.license._track_data import _caller_context
except ImportError:
    _caller_context = None


def main() -> None:
    """Read Ising matrices and return Kaiwu solutions on standard streams."""
    solver = SimulatedAnnealingOptimizer(**json.loads(sys.argv[1]))
    while enabled := sys.stdin.buffer.read(1):
        matrix = receive_array(sys.stdin.buffer)
        previous_source = getattr(_caller_context, "source", None)
        if _caller_context is not None:
            _caller_context.source = "kpp" if enabled == b"1" else None
        try:
            with redirect_stdout(sys.stderr):
                solution = solver.solve(matrix)
        finally:
            if _caller_context is not None:
                _caller_context.source = previous_source
        send_array(sys.stdout.buffer, solution)


if __name__ == "__main__":
    main()
