"""5.2 separate — konuşmayı arka plandan (müzik/efekt) ayırma.

Motorlar (config separation.engine):
- dnr (env: dnr, varsayılan): Bandit v2, çok dilli DnR v3 üzerinde eğitilmiş sinematik 3-stem
  (speech / music / sfx) model. Ara sahneler için doğru araç: müzik vocals modelleri konuşmayı
  "enstrüman" sanabiliyor (test setinde ölçüldü). voice = speech, bg = music + sfx.
  separation.dnr_refine_vocals: true → speech stem ayrıca vocals RoFormer'dan geçirilir, artığı bg'ye eklenir.
- audio_separator (env: sep): en yüksek SDR'li BS/Mel-Band RoFormer vocals modeli (spec'teki yol).

auto modda:
- ses stem'i orijinalden voice_missing_db'den fazla düşükse ayrıştırma başarısız sayılır → orijinal kullanılır;
- arka plan RMS'i clean_bg_threshold_dbfs altındaysa dosya "temiz" sayılır → bg atılır, orijinal kullanılır.
"""
from __future__ import annotations

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..workers import WorkerError, run_worker
from . import Ctx, console


def _audio_separator(ctx: Ctx, items: list[dict], label: str = "separate") -> dict:
    p, cfg = ctx.project, ctx.cfg["separation"]
    res = run_worker(p, "sep", "sep_worker.py", {
        "items": items,
        "options": {"model": cfg.get("model") or "", "output_dir": str(p.wdir("sep"))},
    }, label=label)
    chosen = res.get("model")
    if chosen and not cfg.get("model"):
        p.set_config_value("separation", "model", chosen)
        console.print(f"separation.model → config.yaml: [bold]{chosen}[/]")
    return res


def _dnr(ctx: Ctx, items: list[dict]) -> dict:
    p, cfg = ctx.project, ctx.cfg["separation"]
    sep_dir = p.wdir("sep")
    for it in items:
        it["music_out"] = str(sep_dir / f"{it['id']}_music.wav")
        it["sfx_out"] = str(sep_dir / f"{it['id']}_sfx.wav")
    d = cfg.get("dnr") or {}
    res = run_worker(p, "dnr", "dnr_worker.py", {"items": items, "options": {
        "msst_repo": str(p.abs(ctx.cfg["paths"]["msst_repo"])),
        "checkpoint": str(p.abs(d.get("checkpoint", "tools/models/bandit_v2/checkpoint-multi.ckpt"))),
        "config": str(p.abs(d.get("config", "tools/msst/configs/config_dnr_bandit_v2_mus64.yaml"))),
        "batch_size": int(d.get("batch_size", 2)), "num_overlap": int(d.get("num_overlap", 4)),
        "fp16": bool(d.get("fp16", True)),
    }}, label="separate_dnr")
    if not cfg.get("dnr_refine_vocals"):
        return res
    # Hibrit: speech stem'ini vocals modelinden geçir; artık (residual) arka plana eklenir
    ref_items = [{"id": fid, "path": out["voice"],
                  "voice_out": str(sep_dir / f"{fid}_voice_refined.wav"),
                  "bg_out": str(sep_dir / f"{fid}_resid.wav")}
                 for fid, out in res.get("results", {}).items()]
    try:
        ref = _audio_separator(ctx, ref_items, label="separate_refine")
    except WorkerError as e:
        console.print(f"[yellow]vocals inceltme atlandı: {e}[/]")
        return res
    for fid, out in ref.get("results", {}).items():
        bg, sr = audio.read(res["results"][fid]["bg"])
        resid, rsr = audio.read(out["bg"])
        resid = audio.to_channels(audio.resample(resid, rsr, sr), bg.shape[1])
        n = min(len(bg), len(resid))
        mixed = bg.copy()
        mixed[:n] += resid[:n]
        audio.write(res["results"][fid]["bg"], mixed, sr)
        res["results"][fid]["voice"] = out["voice"]
    return res


def run(ctx: Ctx) -> None:
    p, cfg = ctx.project, ctx.cfg["separation"]
    rows = ctx.manifest.select("separate", ctx.force, ctx.only)
    fids = sorted({r["file_id"] for r in rows})
    if not fids:
        return
    if cfg["mode"] == "never":
        for fid in fids:
            for r in ctx.manifest.rows_of_file(fid):
                r["file_voice_path"], r["bg_path"] = r["ingest_path"], None
                r["bg_rms_dbfs"], r["separation"] = None, "never"
                set_status(r, "separate", "ok")
        ctx.save()
        return

    sep_dir = p.wdir("sep")
    items = [{"id": fid, "path": str(p.abs(ctx.manifest.rows_of_file(fid)[0]["ingest_path"])),
              "voice_out": str(sep_dir / f"{fid}_voice.wav"), "bg_out": str(sep_dir / f"{fid}_bg.wav")}
             for fid in fids]
    engine = cfg.get("engine", "dnr")
    try:
        res = _dnr(ctx, items) if engine == "dnr" else _audio_separator(ctx, items)
    except WorkerError as e:
        for fid in fids:
            for r in ctx.manifest.rows_of_file(fid):
                add_error(r, "separate", e)
        ctx.save()
        console.print(f"[red]separate ({engine}) başarısız:[/] {e}")
        return

    thr = float(cfg["clean_bg_threshold_dbfs"])
    for fid in fids:
        out = res.get("results", {}).get(fid)
        err = res.get("errors", {}).get(fid)
        rows_f = ctx.manifest.rows_of_file(fid)
        if err or not out:
            for r in rows_f:
                add_error(r, "separate", err or "çıktı yok")
            continue
        orig_db = audio.rms_dbfs(audio.read(p.abs(rows_f[0]["ingest_path"]), mono=True)[0])
        voice_db = audio.rms_dbfs(audio.read(out["voice"], mono=True)[0])
        bg_db = audio.rms_dbfs(audio.read(out["bg"], mono=True)[0])
        if cfg["mode"] == "auto" and voice_db - orig_db < -float(cfg.get("voice_missing_db", 15)):
            # Model konuşmayı yakalayamadı (ses stem'i boş): orijinali kullan, uyar.
            state, voice, bg = "voice_missing", rows_f[0]["ingest_path"], None
            msg = (f"{fid}: ayrıştırma ({engine}) konuşmayı yakalayamadı (ses stem'i orijinalden "
                   f"{orig_db - voice_db:.0f} dB düşük) → orijinal kullanıldı, arka plan atıldı")
            if msg not in ctx.notes:
                ctx.notes.append(msg)
        elif cfg["mode"] == "auto" and bg_db < thr:
            state, voice, bg = "clean", rows_f[0]["ingest_path"], None
        else:
            state, voice, bg = "separated", p.rel(out["voice"]), p.rel(out["bg"])
        for r in rows_f:
            r.update({"file_voice_path": voice, "bg_path": bg, "separation": state, "sep_engine": engine,
                      "bg_rms_dbfs": round(bg_db, 1), "voice_rel_db": round(voice_db - orig_db, 1),
                      "music_path": p.rel(out["music"]) if out.get("music") else None,
                      "sfx_path": p.rel(out["sfx"]) if out.get("sfx") else None})
            clear_errors(r, "separate")
            set_status(r, "separate", "ok")
        console.print(f"  {fid}: ses {voice_db - orig_db:+.1f} dB, arka plan {bg_db:.1f} dBFS → {state}")
    ctx.save()
