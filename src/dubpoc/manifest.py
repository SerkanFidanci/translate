"""work/manifest.jsonl — her satır bir segment (replik).

Kurallar (spec 4):
- Her stage sadece kendi alanlarını yazar.
- Her stage sadece status[stage] "bitmiş" olmayan satırları işler; --force hepsini işler.
- Yazma atomiktir (temp dosya + os.replace).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Iterable

from .project import STAGES

# Bir stage için "tamamlandı" sayılan durumlar. "error" ve "pending" tekrar işlenir.
DONE = {"ok", "review", "fail", "skip", "keep_original", "na"}
# Segmenti pipeline'dan çıkaran durumlar (mix orijinal sesi kullanır)
INACTIVE = {"skip", "keep_original"}


def new_status() -> dict[str, str]:
    return {s: "pending" for s in STAGES}


class Manifest:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.rows: list[dict] = []
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.rows.append(json.loads(line))

    # ---- kalıcılık ---------------------------------------------------------
    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows.sort(key=lambda r: (r["file_id"], r.get("start", 0.0), r["seg_id"]))
        tmp = self.path.with_suffix(f".jsonl.tmp{os.getpid()}")
        with tmp.open("w", encoding="utf-8", newline="\n") as f:
            for r in self.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for _ in range(5):  # Windows'ta dosya kısa süre kilitli olabilir
            try:
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.2)
        os.replace(tmp, self.path)

    # ---- sorgular ----------------------------------------------------------
    def get(self, seg_id: str) -> dict | None:
        return next((r for r in self.rows if r["seg_id"] == seg_id), None)

    def file_ids(self) -> list[str]:
        return sorted({r["file_id"] for r in self.rows})

    def rows_of_file(self, file_id: str) -> list[dict]:
        return sorted((r for r in self.rows if r["file_id"] == file_id), key=lambda r: r.get("start", 0.0))

    def replace_file_rows(self, file_id: str, new_rows: Iterable[dict]) -> None:
        self.rows = [r for r in self.rows if r["file_id"] != file_id] + list(new_rows)

    def select(self, stage: str, force: bool = False, only: str | None = None,
               include_inactive: bool = False) -> list[dict]:
        out = []
        ids = {x.strip() for x in only.split(",")} if only else None  # "f001_s08,f001_s22" de olur
        for r in self.rows:
            if ids and r["seg_id"] not in ids and r["file_id"] not in ids:
                continue
            if not include_inactive and is_inactive(r):
                continue
            if force or r["status"].get(stage, "pending") not in DONE:
                out.append(r)
        return out


# ---- satır yardımcıları -------------------------------------------------------
def is_inactive(row: dict) -> bool:
    return any(v in INACTIVE for v in row.get("status", {}).values())


def set_status(row: dict, stage: str, value: str, invalidate_downstream: bool = True) -> None:
    row["status"][stage] = value
    if invalidate_downstream and value in ("ok", "review", "fail", "shorten"):
        idx = STAGES.index(stage)
        for s in STAGES[idx + 1:]:
            if row["status"].get(s) not in INACTIVE:
                row["status"][s] = "pending"


def add_error(row: dict, stage: str, msg: str) -> None:
    row.setdefault("errors", []).append({"stage": stage, "msg": str(msg)[:2000], "t": time.strftime("%Y-%m-%d %H:%M:%S")})
    row["status"][stage] = "error"


def clear_errors(row: dict, stage: str) -> None:
    row["errors"] = [e for e in row.get("errors", []) if e.get("stage") != stage]
