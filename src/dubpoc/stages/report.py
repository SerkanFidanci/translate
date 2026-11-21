"""5.13 report (env: core) — out/report.html (tek dosya, sesler göreli yolla)."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..manifest import is_inactive
from . import Ctx, console


def _rel(ctx: Ctx, rel_path: str | None) -> str | None:
    if not rel_path:
        return None
    a = ctx.project.abs(rel_path)
    if not a.exists():
        return None
    return Path(os.path.relpath(a, ctx.project.out)).as_posix()


def _avg(vals):
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 3) if vals else None


def run(ctx: Ctx) -> None:
    p = ctx.project
    rows = sorted(ctx.manifest.rows, key=lambda r: (r["file_id"], r.get("start", 0)))
    files = []
    for fid in ctx.manifest.file_ids():
        fr = ctx.manifest.rows_of_file(fid)
        files.append({
            "file_id": fid, "src_path": fr[0].get("src_path"), "dur": fr[0].get("src_dur"),
            "separation": fr[0].get("separation"),
            "orig": _rel(ctx, fr[0].get("ingest_path")),
            "A": _rel(ctx, p.rel(p.out / "A" / f"{fid}.wav")),
            "B": _rel(ctx, p.rel(p.out / "B" / f"{fid}.wav")),
            "n_seg": len(fr),
        })
    segs = []
    for r in rows:
        cands = r.get("candidates", {})
        best = r.get("best", {})
        a_list = []
        for i, c in enumerate(cands.get("A", [])):
            a_list.append({**c, "audio": _rel(ctx, c.get("fit")), "raw": _rel(ctx, c.get("src")),
                           "is_best": best.get("A") == i})
        b = None
        if cands.get("B"):
            bi = best.get("B", 0) or 0
            b = {**cands["B"][bi], "audio": _rel(ctx, cands["B"][bi].get("fit"))}
        a_best = a_list[best["A"]] if best.get("A") is not None and best["A"] < len(a_list) else None
        state = r["status"].get("qa")
        if is_inactive(r):
            state = next(v for v in r["status"].values() if v in ("skip", "keep_original"))
        elif any(v == "error" for v in r["status"].values()):
            state = "error"
        segs.append({
            "seg_id": r["seg_id"], "file_id": r["file_id"], "char": r.get("char"),
            "emotion": r.get("emotion"), "intensity": r.get("intensity"),
            "text_en": r.get("text_en"), "text_tr": r.get("text_tr"),
            "tr_syllables": r.get("tr_syllables"), "syll_target": r.get("syll_target"),
            "dur_speech": r.get("dur_speech"), "start": r.get("start"),
            "orig": _rel(ctx, r.get("voice_path")), "a_best": a_best, "a_list": a_list, "b": b,
            "state": state, "errors": r.get("errors", []), "notes": r.get("mix_notes", {}),
            "shorten_round": r.get("shorten_round", 0),
        })
    active = [s for s in segs if s["state"] not in ("skip", "keep_original")]
    dist = {k: sum(1 for s in active if s["state"] == k) for k in ("ok", "review", "fail", "error")}
    dist["pending"] = sum(1 for s in active if s["state"] not in ("ok", "review", "fail", "error"))

    def path_avgs(get):
        cs = [get(s) for s in active]
        cs = [c for c in cs if c]
        return {k: _avg([c.get(k) for c in cs]) for k in ("wer", "cer", "spk_sim_rel", "dur_ratio", "emo_sim", "score")} | {"n": len(cs)}

    summary = {
        "n_seg": len(segs), "n_active": len(active), "dist": dist,
        "keep_original": sum(1 for s in segs if s["state"] == "keep_original"),
        "skip": sum(1 for s in segs if s["state"] == "skip"),
        "A": path_avgs(lambda s: s["a_best"]), "B": path_avgs(lambda s: s["b"]),
    }
    chars = []
    for bdir in sorted((p.work / "bank").glob("*")) if (p.work / "bank").exists() else []:
        base = {}
        if (bdir / "baseline.json").exists():
            base = json.loads((bdir / "baseline.json").read_text(encoding="utf-8"))
        chars.append({"name": bdir.name, "ref": _rel(ctx, p.rel(bdir / "ref.wav")), **base})
    q = ctx.cfg["qa"]
    env = Environment(loader=FileSystemLoader(Path(__file__).resolve().parent.parent / "templates"),
                      autoescape=select_autoescape(["html", "j2"]))
    html = env.get_template("report.html.j2").render(
        files=files, segs=segs, summary=summary, chars=chars, q=q, cfg=ctx.cfg,
        generated=time.strftime("%Y-%m-%d %H:%M"), notes=ctx.all_notes(),
        project_key=str(p.root.name),
    )
    out = p.odir() / "report.html"
    out.write_text(html, encoding="utf-8")
    console.print(f"report → {out}")
