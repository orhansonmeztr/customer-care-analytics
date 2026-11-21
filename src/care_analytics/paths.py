"""Filesystem layout. Everything generated lives under one data directory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def truth(self) -> Path:
        """Ground truth of what the generator injected. Only used to score the pipeline."""
        return self.root / "raw" / "_truth"

    @property
    def warehouse(self) -> Path:
        return self.root / "warehouse.duckdb"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def exports(self) -> Path:
        """CSV extracts for Tableau / Power BI."""
        return self.root / "exports"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def ensure(self) -> Paths:
        for p in (self.raw, self.truth, self.outputs, self.exports, self.reports):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(root: str | Path | None = None) -> Paths:
    root = root or os.environ.get("CARE_DATA_DIR", "data")
    return Paths(Path(root)).ensure()
