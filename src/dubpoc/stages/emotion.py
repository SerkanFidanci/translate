"""5.6 emotion (env: emo) — emotion2vec_plus_large + yoğunluk (intensity).

intensity = 0.5*A + 0.25*norm(RMS) + 0.25*norm(pitch_std), normalizasyon karakter bazında min-max.

Araştırma notu: spec'teki A = "emotion_score (nötr değilse)" sınıflandırıcı güvenidir; emin bir
"neutral" da yüksek skor alır. Burada A = 1 − p(neutral) kullanılır (duygunun ne kadar belirgin
olduğu). Spec davranışı için config: emotion.intensity_source = top_score.

Efor/sessiz segmentler: ASR boş ya da sadece ünlem/efor açıklaması ("Mm-hmm.", "[grunts]") ise
status=keep_original (mix orijinal sesi kullanır). Tek kelimelik gerçek sözcükler ("Fuck.", "Okay.",
"Surprise!") dublajlanır: duygu modeli bunlara sık "other" diyordu ve orijinal İngilizce ses kalıyordu.
"""
from __future__ import annotations

import numpy as np

from ..manifest import add_error, clear_errors, set_status
from ..textutil import is_nonverbal_text
from ..workers import WorkerError, run_worker
from . import Ctx, console



def emo_items(ctx: Ctx, pairs: list[tuple[str, str]], emb_dir_name: str) -> list[dict]:
    d = ctx.project.wdir(emb_dir_name)
    return [{"id": i, "path": path, "emb_out": str(d / f"{i}.npy")} for i, path in pairs]


def run_emo_worker(ctx: Ctx, items: list[dict], label: str) -> dict:
    return run_worker(ctx.project, "emo", "emo_worker.py",
                      {"items": items, "options": {"model": ctx.cfg["emotion"]["model"]}}, label=label)


def _minmax(vals: list[float]) -> list[float]:
    if not vals:
        return []
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return [0.5] * len(vals)
    return [(v - lo) / (hi - lo) for v in vals]


def run(ctx: Ctx) -> None:
    p = ctx.project
    rows = ctx.manifest.select("emotion", ctx.force, ctx.only)
    rows = [r for r in rows if r["status"].get("asr") == "ok"]
    if not rows:
        return
    items = emo_items(ctx, [(r["seg_id"], str(p.abs(r["voice_path"]))) for r in rows], "emo_src")
    try:
        res = run_emo_worker(ctx, items, "emotion")
    except WorkerError as e:
        for r in rows:
            add_error(r, "emotion", e)
        ctx.save()
        console.print(f"[red]emotion başarısız:[/] {e}")
        return
    for r in rows:
        out = res.get("results", {}).get(r["seg_id"])
        if out is None:
            add_error(r, "emotion", res.get("errors", {}).get(r["seg_id"], "çıktı yok"))
            continue
        r["emotion"] = out["emotion"]
        r["emotion_score"] = round(float(out["score"]), 4)
        r["emotion_probs"] = {k: round(float(v), 4) for k, v in out["probs"].items()}
        r["rms_db"] = round(float(out["rms_db"]), 2)
        r["pitch_std"] = round(float(out["pitch_std"]), 3) if out.get("pitch_std") is not None else None
        r["emo_emb"] = p.rel(out["emb"]) if out.get("emb") else None
        clear_errors(r, "emotion")

    # karakter bazında yoğunluk (tüm aktif satırlar üzerinden min-max)
    src = ctx.cfg["emotion"].get("intensity_source", "one_minus_neutral")
    all_rows = [r for r in ctx.manifest.select("emotion", force=True) if "rms_db" in r]
    for ch in sorted({r.get("char") or "?" for r in all_rows}):
        group = [r for r in all_rows if (r.get("char") or "?") == ch]
        rms_n = _minmax([r["rms_db"] for r in group])
        pitches = [r["pitch_std"] if r.get("pitch_std") is not None else np.nan for r in group]
        fill = float(np.nanmedian(pitches)) if not all(np.isnan(pitches)) else 0.0
        pit_n = _minmax([fill if np.isnan(x) else x for x in pitches])
        for r, a, b in zip(group, rms_n, pit_n):
            if src == "top_score":
                emo_term = r["emotion_score"] if r["emotion"] != "neutral" else 0.0
            else:
                emo_term = 1.0 - float(r.get("emotion_probs", {}).get("neutral", 0.0))
            r["intensity"] = round(float(np.clip(0.5 * emo_term + 0.25 * a + 0.25 * b, 0, 1)), 3)

    for r in rows:
        if "emotion" not in r or r["status"]["emotion"] == "error":
            continue
        text = r.get("text_en", "")
        nonverbal = is_nonverbal_text(text)
        if nonverbal:
            r["status"]["emotion"] = "keep_original"
            console.print(f"  {r['seg_id']}: efor/sessiz segment → keep_original ({text!r})")
        else:
            set_status(r, "emotion", "ok")
    ctx.save()
