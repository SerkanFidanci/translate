"""5.9 vc — Yol A ve B: karakter tonuna ses dönüştürme.

- Kaynak: Yol A için her TTS adayı, Yol B için record/<seg_id>.(wav|flac|mp3|ogg|m4a).
- Hedef: bank/<char>/ref.wav. Backend: seed_vc (env vc) | chatterbox_vc (env tts); arayüz aynı.
- vc.length_adjust: Seed-VC'nin length_adjust parametresiyle kaynak konuşma süresi hedef süreye
  (dur_speech) yaklaştırılır (±max_length_adjust); fit'te kalan küçük fark esnetilir.
- vc.also_score_without_vc: VC Türkçe anlaşılırlığı düşürebildiği için (Seed-VC EVAL: WER 8→12)
  ham TTS adayları da QA'ya girer; en iyi aday metriklerle seçilir.
"""
from __future__ import annotations

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..workers import WorkerError, run_worker
from . import Ctx, console

REC_EXT = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")
# +vc adayına taşınan alanlar (fit/QA metrikleri taşınmaz; yeniden hesaplanır)
TTS_KEYS = {"id", "backend", "seed", "tts", "tts_dur", "exaggeration", "cfg_weight", "duration_target", "ref_kind", "xtts_speed", "style", "prompt_seg"}


def find_recording(ctx: Ctx, seg_id: str):
    for ext in REC_EXT:
        f = ctx.project.record_dir / f"{seg_id}{ext}"
        if f.exists():
            return f
    return None


def _length_adjust(ctx: Ctx, src_path, target: float | None) -> float:
    c = ctx.cfg["vc"]
    if not c.get("length_adjust", True) or not target:
        return 1.0
    x, sr = audio.read(src_path, mono=True)
    s0, s1 = audio.speech_bounds(x, sr, float(ctx.cfg["fit"]["silence_db"]))
    if s1 <= s0:
        return 1.0
    m = float(c.get("max_length_adjust", 0.2))
    return round(min(max(target / (s1 - s0), 1 - m), 1 + m), 4)


def _skip_vc(ctx: Ctx, rows: list[dict]) -> None:
    """vc.enabled=false ("doğal mod"): ses klonlayan TTS'lerde (VoxCPM2/XTTS) ikinci bir dönüştürme
    tınıyı az iyileştirip metalik/robotik artefakt ekliyor (ölçüldü: tam GTA videosunda seçilenlerin
    20/42'si +vc idi, kullanıcı "robotikleşiyor" dedi). Ham TTS adayları olduğu gibi fit'e gider;
    Yol B kayıtları dönüştürülmeden aday olur."""
    p = ctx.project
    for r in rows:
        if ctx.want("A") and r["status"].get("tts") == "ok":
            r["candidates"]["A"] = [{k: v for k, v in cand.items() if k in TTS_KEYS | {"src"}}
                                    for cand in r.get("candidates", {}).get("A", []) if not cand["id"].endswith("+vc")]
            for cand in r["candidates"]["A"]:
                cand["src"] = cand.get("tts") or cand.get("src")
        if ctx.want("B"):
            rec = find_recording(ctx, r["seg_id"])
            if rec is not None:
                rec_wav = p.wdir("rec") / f"{r['seg_id']}.wav"
                audio.convert_to_wav(rec, rec_wav, p.tool("ffmpeg"))
                r.setdefault("candidates", {})["B"] = [{"id": "rec", "rec": p.rel(rec), "rec_wav": p.rel(rec_wav),
                                                        "src": p.rel(rec_wav)}]
                r.setdefault("best", {}).pop("B", None)
        r.pop("vc_errors", None)
        clear_errors(r, "vc")
        set_status(r, "vc", "ok")
    ctx.save()
    console.print(f"vc: kapalı (vc.enabled=false) — {len(rows)} segment ham TTS/kayıt ile devam ediyor")


def run(ctx: Ctx) -> None:
    p, c = ctx.project, ctx.cfg["vc"]
    rows = [r for r in ctx.manifest.select("vc", ctx.force, ctx.only)
            if r["status"].get("translate") == "ok"]
    if not rows:
        return
    if not c.get("enabled", True):
        _skip_vc(ctx, rows)
        return
    items, meta = [], {}
    for r in rows:
        target_ref = p.work / "bank" / (r.get("char") or "") / "ref.wav"
        if not target_ref.exists():
            add_error(r, "vc", f"ses bankası yok: {target_ref}")
            continue
        target_dur = r.get("dur_speech") or r.get("dur_src")
        d = p.wdir("vc", r["seg_id"])
        if ctx.want("A") and r["status"].get("tts") == "ok":
            raw = [{k: v for k, v in cand.items() if k in TTS_KEYS | {"src"}}
                   for cand in r.get("candidates", {}).get("A", []) if not cand["id"].endswith("+vc")]
            r["candidates"]["A"] = raw  # eski +vc girdilerini at
            for cand in raw:
                src = p.abs(cand["tts"])
                # hedef sürede üretilmiş adaylar (OmniVoice duration) zaten doğru uzunlukta
                la = 1.0 if (cand.get("duration_target") or cand.get("xtts_speed", 1.0) != 1.0)                     else _length_adjust(ctx, src, target_dur)
                key = f"{r['seg_id']}|{cand['id']}"
                out = d / f"A_{cand['id']}.wav"
                items.append({"id": key, "source": str(src), "target": str(target_ref), "out": str(out),
                              "length_adjust": la})
                meta[key] = (r, "A", cand, la)
        if ctx.want("B"):
            rec = find_recording(ctx, r["seg_id"])
            if rec is not None and r.get("text_tr_time") and rec.stat().st_mtime < float(r["text_tr_time"]):
                msg = (f"{r['seg_id']}: record/ kaydı çeviri değişmeden ÖNCE yapılmış; eski metni okuyor olabilir "
                       f"(güncel: {r['text_tr']!r}). out/script.md'ye göre yeniden kaydedin.")
                r["rec_stale"] = True
                if msg not in ctx.notes:
                    ctx.notes.append(msg)
                console.print(f"[yellow]{msg}[/]")
            if rec is not None:
                rec_wav = p.wdir("rec") / f"{r['seg_id']}.wav"
                audio.convert_to_wav(rec, rec_wav, p.tool("ffmpeg"))
                la = _length_adjust(ctx, rec_wav, target_dur)
                key = f"{r['seg_id']}|B"
                items.append({"id": key, "source": str(rec_wav), "target": str(target_ref),
                              "out": str(d / "B.wav"), "length_adjust": la})
                meta[key] = (r, "B", {"rec": p.rel(rec), "rec_wav": p.rel(rec_wav)}, la)
    if not items:
        ctx.save()
        return
    backend = c.get("backend", "seed_vc")
    env = "vc" if backend == "seed_vc" else "tts"
    try:
        res = run_worker(p, env, "vc_worker.py", {"items": items, "options": {
            "backend": backend, "diffusion_steps": int(c["diffusion_steps"]),
            "inference_cfg_rate": float(c["inference_cfg_rate"]), "fp16": bool(c.get("fp16", True)),
            "seed_vc_repo": str(p.abs(ctx.cfg["paths"]["seed_vc_repo"])),
        }}, label="vc")
    except WorkerError as e:
        for r in rows:
            add_error(r, "vc", e)
        ctx.save()
        console.print(f"[red]vc başarısız:[/] {e}")
        return

    ok_rows: set[str] = set()
    b_rows: dict[str, list] = {}
    for key, (r, path, cand, la) in meta.items():
        got = res.get("results", {}).get(key)
        if not got:
            r.setdefault("vc_errors", {})[key] = res.get("errors", {}).get(key, "çıktı yok")
            continue
        if path == "A":
            r["candidates"]["A"].append({**{k: v for k, v in cand.items() if k in TTS_KEYS},
                                         "id": cand["id"] + "+vc", "vc": p.rel(got["path"]),
                                         "src": p.rel(got["path"]), "length_adjust": la})
        else:
            b_rows.setdefault(r["seg_id"], []).append({"id": "rec+vc", **cand, "vc": p.rel(got["path"]),
                                                        "src": p.rel(got["path"]), "length_adjust": la})
        ok_rows.add(r["seg_id"])
    had_items = {m[0]["seg_id"] for m in meta.values()}
    for r in rows:
        if r["seg_id"] not in had_items:
            continue  # bu yol için kaynak yok (örn. Yol B kaydı yok) – hata sayılmaz
        if r["seg_id"] in b_rows:
            r.setdefault("candidates", {})["B"] = b_rows[r["seg_id"]]
            r.setdefault("best", {}).pop("B", None)
        if not c.get("also_score_without_vc", True) and r.get("candidates", {}).get("A"):
            vc_only = [x for x in r["candidates"]["A"] if x["id"].endswith("+vc")]
            if vc_only:
                r["candidates"]["A"] = vc_only
        if r["seg_id"] in ok_rows:
            r.pop("vc_errors", None)
            clear_errors(r, "vc")
            set_status(r, "vc", "ok")
        elif r["status"]["vc"] != "error":
            add_error(r, "vc", str(r.get("vc_errors") or "dönüştürülecek kaynak yok"))
    ctx.save()
