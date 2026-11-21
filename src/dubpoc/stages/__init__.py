"""Pipeline aşamaları. Her modül `run(ctx)` fonksiyonu sağlar."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from rich.console import Console

from ..manifest import Manifest
from ..project import Project

console = Console()


@dataclass
class Ctx:
    project: Project
    manifest: Manifest
    force: bool = False
    only: str | None = None
    path: str = "both"          # A | B | both
    notes: list[str] = field(default_factory=list)

    @property
    def cfg(self) -> dict:
        return self.project.cfg

    def want(self, p: str) -> bool:
        return self.path in ("both", p)

    def save(self) -> None:
        self.manifest.save()
        if self.notes:
            path = self.project.wdir() / "notes.json"
            old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
            merged = list(dict.fromkeys(old + self.notes))
            path.write_text(json.dumps(merged, ensure_ascii=False, indent=1), encoding="utf-8")

    def all_notes(self) -> list[str]:
        path = self.project.work / "notes.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        return list(dict.fromkeys(old + self.notes))
