"""Filesystem paths used throughout the project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    data: Path
    raw: Path
    interim: Path
    processed: Path
    artifacts: Path
    baseline_artifacts: Path
    metadata: Path
    reports: Path

    @classmethod
    def discover(cls) -> "ProjectPaths":
        root = Path(__file__).resolve().parents[2]
        data = root / "data"
        artifacts = root / "artifacts"
        return cls(
            root=root,
            data=data,
            raw=data / "raw",
            interim=data / "interim",
            processed=data / "processed",
            artifacts=artifacts,
            baseline_artifacts=artifacts / "baseline",
            metadata=artifacts / "metadata",
            reports=root / "reports",
        )

    def ensure(self) -> list[Path]:
        created: list[Path] = []
        for path in (
            self.data,
            self.raw,
            self.interim,
            self.processed,
            self.artifacts,
            self.baseline_artifacts,
            self.metadata,
            self.reports,
        ):
            path.mkdir(parents=True, exist_ok=True)
            created.append(path)
        return created

