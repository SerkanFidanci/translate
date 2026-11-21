"""dubpoc CLI (typer)."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from . import doctor as doctor_mod
from . import pipeline
from .manifest import Manifest
from .project import ALL_STEPS, STAGES, Project, find_root

if sys.platform == "win32":  # Windows konsolunda Türkçe karakterler
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
        except Exception:  # noqa: BLE001
            pass

app = typer.Typer(add_completion=False, no_args_is_help=True,
                  help="Oyun replikleri için Türkçe AI dublaj POC'si.")
console = Console()
STATE: dict = {}


def project() -> Project:
    if "project" not in STATE:
        STATE["project"] = Project(find_root(STATE.get("root")))
    return STATE["project"]


@app.callback()
def main(root: Optional[Path] = typer.Option(None, "--root", help="Proje kökü (config.yaml'ın olduğu klasör)")):
    STATE["root"] = root


@app.command()
def doctor():
    """Ortam / binary / GPU / HF token / API key kontrolü."""
    raise typer.Exit(doctor_mod.run(project()))


def _make_stage_cmd(name: str):
    def cmd(force: bool = typer.Option(False, "--force", help="Bekleyenler değil, tüm satırları işle"),
            only: Optional[str] = typer.Option(None, "--only", help="Sadece bu seg_id ya da file_id"),
            path: str = typer.Option("both", "--path", help="A | B | both")):
        ctx = pipeline.make_ctx(project(), force, only, path)
        pipeline.run_stage(ctx, name)
    cmd.__doc__ = f"Sadece '{name}' aşamasını çalıştır."
    return cmd


for _name in ALL_STEPS + ["export"]:
    app.command(name=_name)(_make_stage_cmd(_name))


@app.command()
def run(
    frm: Optional[str] = typer.Option(None, "--from", help=f"Başlangıç aşaması ({', '.join(ALL_STEPS)})"),
    to: Optional[str] = typer.Option(None, "--to", help="Bitiş aşaması"),
    only: Optional[str] = typer.Option(None, "--only", help="Sadece bu seg_id ya da file_id"),
    force: bool = typer.Option(False, "--force", help="Tüm satırları yeniden işle"),
    path: str = typer.Option("both", "--path", help="A | B | both (B sadece record/ dosyası olan segmentler)"),
):
    """Aşamaları sırayla çalıştır (varsayılan: hepsi)."""
    for v in (frm, to):
        if v and v not in ALL_STEPS:
            raise typer.BadParameter(f"Bilinmeyen aşama: {v}")
    if path not in ("A", "B", "both"):
        raise typer.BadParameter("--path A | B | both")
    pipeline.run(project(), frm, to, only, force, path)
    status(segments=False)


@app.command()
def serve(port: int = typer.Option(8765, "--port", help="Yerel port"),
          open_browser: bool = typer.Option(True, "--open/--no-open", help="Tarayıcıda aç")):
    """Raporu yerel sunucudan aç (sesler göreli yolla work/ ve out/ altından çalar)."""
    import functools
    import http.server
    import webbrowser

    p = project()
    url = f"http://localhost:{port}/out/report.html"
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(p.root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", port), handler) as httpd:
        console.print(f"Rapor: [bold]{url}[/]  (durdurmak için Ctrl+C)")
        if open_browser:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


@app.command()
def clean(yes: bool = typer.Option(False, "--yes", "-y", help="Onay sorma")):
    """work/, out/ ve speakers.yaml'ı siler (input/ ve record/ asla silinmez)."""
    import shutil

    p = project()
    targets = [x for x in (p.work, p.out, p.speakers_yaml) if x.exists()]
    if not targets:
        console.print("Silinecek bir şey yok.")
        return
    console.print("Silinecek: " + ", ".join(str(t.relative_to(p.root)) for t in targets))
    if not yes and not typer.confirm("Devam edilsin mi?"):
        raise typer.Abort()
    for t in targets:
        shutil.rmtree(t) if t.is_dir() else t.unlink()
    console.print("[green]Temizlendi.[/] (input/, record/ ve config dosyaları korundu; indirilen modeller "
                  "~/.cache/huggingface altında duruyor)")


@app.command()
def status(segments: bool = typer.Option(False, "--segments", "-s", help="Segment bazında tablo")):
    """Manifest özet tablosu."""
    p = project()
    m = Manifest(p.manifest_path)
    if not m.rows:
        console.print("Manifest boş. Önce: dubpoc ingest")
        return
    t = Table(title=f"{len(m.rows)} segment, {len(m.file_ids())} dosya")
    t.add_column("aşama")
    keys = ["ok", "review", "fail", "error", "pending", "shorten", "skip", "keep_original"]
    for k in keys:
        t.add_column(k, justify="right")
    for s in STAGES:
        c = Counter(r["status"].get(s, "pending") for r in m.rows)
        t.add_row(s, *[str(c.get(k, "")) if c.get(k) else "" for k in keys])
    console.print(t)
    if segments:
        st = Table()
        for col in ("seg_id", "char", "emotion", "dur", "TR", "qa", "best A", "hata"):
            st.add_column(col, overflow="fold")
        for r in m.rows:
            a = r.get("candidates", {}).get("A") or []
            bi = (r.get("best") or {}).get("A")
            ba = a[bi] if bi is not None and bi < len(a) else {}
            st.add_row(r["seg_id"], str(r.get("char") or ""), str(r.get("emotion") or ""),
                       f"{r.get('dur_speech') or 0:.2f}", (r.get("text_tr") or "")[:60],
                       r["status"].get("qa", ""),
                       f"{ba.get('id', '')} wer={ba.get('wer')} spk={ba.get('spk_sim_rel')}" if ba else "",
                       (r.get("errors") or [{}])[-1].get("msg", "")[:80] if r.get("errors") else "")
        console.print(st)


if __name__ == "__main__":
    app()
