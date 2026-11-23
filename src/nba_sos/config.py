"""Project paths and YAML config loading."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def interim(self) -> Path:
        return self.root / "data" / "interim"

    @property
    def processed(self) -> Path:
        return self.root / "data" / "processed"

    @property
    def reports(self) -> Path:
        return self.root / "reports"


def load_yaml(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_configs(paths: Paths) -> dict:
    return {
        "data_sources": load_yaml(paths.configs / "data_sources.yaml"),
        "features": load_yaml(paths.configs / "feature_config.yaml"),
        "model": load_yaml(paths.configs / "model_config.yaml"),
    }


def season_start_year(season: str) -> int:
    """'2021-22' -> 2021."""
    return int(season[:4])


def previous_season(season: str) -> str:
    y = season_start_year(season) - 1
    return f"{y}-{str(y + 1)[-2:]}"
