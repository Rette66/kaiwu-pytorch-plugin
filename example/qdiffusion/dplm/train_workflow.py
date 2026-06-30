"""Entrypoint for the DPLM Q-Diffusion training workflow."""

if __package__:
    from .trainer.trainer import main
else:  # pragma: no cover - direct script-path compatibility
    from trainer.trainer import main


if __name__ == "__main__":
    main()
