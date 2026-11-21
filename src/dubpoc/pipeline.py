"""Aşamaları sırayla çalıştırma ve kısaltma döngüsü."""
from __future__ import annotations

import importlib

import yaml

from .manifest import Manifest
from .project import ALL_STEPS, Project
from .stages import Ctx, console

LOOP_STAGES = ["translate", "tts", "vc", "fit"]


def stage_module(name: str):
    return importlib.import_module(f".stages.{name}", package=__package__)


def make_ctx(project: Project, force: bool = False, only: str | None = None, path: str = "both") -> Ctx:
    return Ctx(project=project, manifest=Manifest(project.manifest_path), force=force, only=only, path=path)


def run_stage(ctx: Ctx, name: str) -> None:
    console.rule(f"[bold]{name}")
    stage_module(name).run(ctx)


def _warn_unnamed(ctx: Ctx) -> None:
    y = ctx.project.speakers_yaml
    if not y.exists():
        return
    data = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
    unnamed = [k for k, v in data.items() if k != "overrides" and not (v or {}).get("name")]
    if unnamed:
        console.print(f"[yellow]Uyarı: speakers.yaml'da isimsiz kümeler var ({', '.join(unnamed)}). "
                      "Devam ediliyor; isim verip `dubpoc run --from speakers` ile yeniden çalıştırabilirsiniz.[/]")


def run(project: Project, frm: str | None = None, to: str | None = None, only: str | None = None,
        force: bool = False, path: str = "both") -> Ctx:
    steps = ALL_STEPS
    a = steps.index(frm) if frm else 0
    b = steps.index(to) if to else len(steps) - 1
    selected = steps[a: b + 1]
    ctx = make_ctx(project, force, only, path)
    for name in selected:
        run_stage(ctx, name)
        if name == "speakers":
            _warn_unnamed(ctx)
        if name == "fit" and all(s in selected for s in LOOP_STAGES):
            _shorten_loop(ctx)
        # --force sadece ilk geçişte "hepsini işle" anlamına gelir; döngüler bekleyenleri işler
    return ctx


def _shorten_loop(ctx: Ctx) -> None:
    from .stages.fit import pending_shorten

    max_rounds = int(ctx.cfg["translate"]["max_shorten_rounds"])
    force = ctx.force
    for i in range(max_rounds):
        todo = pending_shorten(ctx)
        if not todo:
            break
        console.rule(f"[bold]kısaltma döngüsü {i + 1} ({len(todo)} segment)")
        ctx.force = False
        for name in LOOP_STAGES:
            run_stage(ctx, name)
    ctx.force = force
