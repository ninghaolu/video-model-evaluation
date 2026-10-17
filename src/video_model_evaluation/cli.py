"""Hydra entry point for video model evaluation."""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .evaluator import evaluate_from_config


@hydra.main(version_base="1.3", config_path="configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    evaluate_from_config(cfg)


if __name__ == "__main__":
    main()

