"""Entrypoint for DPLM Q-Diffusion ESM2 distance evaluation."""

if __package__:
    from .downstream.pipeline import main
else:  # pragma: no cover - direct script-path compatibility
    from downstream.pipeline import main


if __name__ == "__main__":
    main()
