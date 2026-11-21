"""5.8 tts — Yol A: Türkçe metin → aday sesler.

Backend'ler (config tts.backends, sırayla çalışır, her biri kendi venv'inde):
- chatterbox_mtl (env: tts): ChatterboxMultilingualTTS, language_id="tr", V3 checkpoint.
  exaggeration = lerp(exaggeration_range, intensity); referans bank ya da orijinal replik.
- omnivoice (env: omni): k2-fsa OmniVoice; `duration=` ile HEDEF SÜREDE üretir (araştırma
  önerisi: sonradan esnetmek yerine doğru uzunlukta üretmek prozodiyi korur).

Her aday manifest'te candidates.A listesine {id, backend, seed, tts, src} olarak yazılır.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..textutil import speakable_tr
from ..workers import WorkerError, run_worker
from . import Ctx, console

SHORT = {"chatterbox_mtl": "cb", "omnivoice": "ov", "xtts": "xt", "voxcpm": "vx"}
MIN_ORIG_REF_S = 3.0


def lerp(rng: list[float], t: float) -> float:
    t = min(max(float(t or 0.0), 0.0), 1.0)
    return float(rng[0]) + (float(rng[1]) - float(rng[0])) * t


def bank_ref(ctx: Ctx, char: str):
    return ctx.project.work / "bank" / char / "ref.wav"


def bank_ref_text(ctx: Ctx, char: str, name: str = "clips.json") -> str | None:
    clips_path = ctx.project.work / "bank" / char / name
    if not clips_path.exists():
        return None
    texts = []
    for sid in json.loads(clips_path.read_text(encoding="utf-8")):
        r = ctx.manifest.get(sid)
        if not r or not r.get("text_en"):
            return None
        texts.append(r["text_en"].strip())
    return " ".join(texts) or None


def _chatterbox_items(ctx: Ctx, rows: list[dict]) -> list[dict]:
    c = ctx.cfg["tts"]
    items = []
    for r in rows:
        exag = lerp(c["exaggeration_range"], r.get("intensity", 0.5))
        mode = c["reference_mode"]
        ref = bank_ref(ctx, r["char"])
        if mode == "original_line" and (r.get("dur_speech") or 0) >= MIN_ORIG_REF_S:
            ref = ctx.project.abs(r["voice_path"])
            cfgw = float(c["cfg_weight_original_line"])
        else:
            cfgw = float(c["cfg_weight_bank"])
            if exag >= 0.7:
                cfgw = float(c["high_exaggeration_cfg"])
        seeds = list(c["seeds"])[: int(c["n_candidates"])]
        d = ctx.project.wdir("tts", r["seg_id"])
        outs = [{"cid": f"cb{k + 1}", "seed": int(s), "path": str(d / f"cb{k + 1}.wav")} for k, s in enumerate(seeds)]
        # Duygu referansı: son N aday, önce repliğin KENDİ orijinal sesi + sonra karakter bankası olan birleşik
        # referansla üretilir. Chatterbox prozodi/üslubu referansın başından (ilk ~6 sn) aldığı için orijinal
        # satırın heyecanı taşınır; tını banka kısmından gelir. İngilizce aksan sızmasın diye cfg düşük.
        n_emo = min(int(c.get("emotion_ref_candidates", 2)), len(outs)) if mode == "bank" else 0
        if n_emo:
            line_ref = _line_plus_bank_ref(ctx, r, ref, d / "ref_line.wav")
            for o in outs[len(outs) - n_emo:]:
                o.update({"ref": str(line_ref), "cfg_weight": float(c.get("cfg_weight_emotion_ref", 0.3)),
                          "ref_kind": "line+bank"})
        items.append({
            "id": r["seg_id"], "text": speakable_tr(r["text_tr"]), "ref": str(ref),
            "exaggeration": round(exag, 3), "cfg_weight": cfgw, "temperature": float(c.get("temperature", 0.8)),
            "outs": outs,
        })
    return items


def _line_plus_bank_ref(ctx: Ctx, r: dict, bank_path, out_path):
    x, sr = audio.read(ctx.project.abs(r["voice_path"]), mono=True)
    line = audio.trim(x, sr, float(ctx.cfg["fit"]["silence_db"]))
    b, bsr = audio.read(bank_path, mono=True)
    line = audio.resample(line, sr, bsr)
    ref = np.concatenate([line, np.zeros(int(0.25 * bsr), dtype=np.float32), b])
    audio.write(out_path, ref, bsr, subtype="PCM_16")
    return out_path


def _omnivoice_items(ctx: Ctx, rows: list[dict]) -> list[dict]:
    c = ctx.cfg["tts"].get("omnivoice", {}) or {}
    items = []
    for r in rows:
        n = int(c.get("n_candidates", 2))
        seeds = list(ctx.cfg["tts"]["seeds"])[:n] or [11]
        d = ctx.project.wdir("tts", r["seg_id"])
        target = r.get("dur_speech") or r.get("dur_src")
        short = ctx.project.work / "bank" / r["char"] / "ref_short.wav"
        if short.exists():
            ref, ref_text = short, bank_ref_text(ctx, r["char"], "clips_short.json")
        else:
            ref, ref_text = bank_ref(ctx, r["char"]), None
        # Aday başına süre modu: "target" = hedef konuşma süresi, "natural" = modelin kendi tahmini.
        # Sıkı hedef Türkçe anlaşılırlığı bazen bozuyor (ölçüldü); karışık adaylar + QA seçimi daha sağlam.
        modes = list(c.get("duration_modes") or ["target", "natural"])
        outs = []
        for k, s in enumerate(seeds):
            mode = modes[k % len(modes)] if c.get("use_duration", True) else "natural"
            if target and float(target) < float(c.get("min_target_s", 1.0)):
                mode = "natural"  # <1 sn hedefte OmniVoice anlamsız ses üretiyor (ölçüldü: "Fast" 0,58 sn)
            dur = round(float(target), 3) if mode == "target" and target else None
            outs.append({"cid": f"ov{k + 1}", "seed": int(s), "path": str(d / f"ov{k + 1}.wav"), "duration": dur})
        items.append({"id": r["seg_id"], "text": speakable_tr(r["text_tr"]), "ref": str(ref), "ref_text": ref_text,
                      "outs": outs})
    return items


def _xtts_items(ctx: Ctx, rows: list[dict]) -> list[dict]:
    """XTTS-v2: aday 1 = karakter bankası, aday 2 = orijinal replik + banka (duygu/üslup referansı)."""
    c = ctx.cfg["tts"].get("xtts", {}) or {}
    n = int(c.get("n_candidates", 2))
    seeds = list(ctx.cfg["tts"]["seeds"])[:n] or [11]
    items = []
    for r in rows:
        d = ctx.project.wdir("tts", r["seg_id"])
        bank = bank_ref(ctx, r["char"])
        line = _line_plus_bank_ref(ctx, r, bank, d / "ref_line.wav") if c.get("emotion_ref", True) else None
        target = r.get("dur_speech") or r.get("dur_src")
        outs = []
        for k, s in enumerate(seeds):
            use_line = line is not None and k % 2 == 1
            outs.append({"cid": f"xt{k + 1}", "seed": int(s), "path": str(d / f"xt{k + 1}.wav"),
                         "refs": [str(line if use_line else bank)],
                         "ref_kind": "line+bank" if use_line else "bank",
                         "duration": round(float(target), 3) if target else None})
        items.append({"id": r["seg_id"], "text": speakable_tr(r["text_tr"]), "outs": outs})
    return items


EMO_STYLE = {"angry": "angry", "happy": "cheerful, excited", "sad": "sad, quiet", "fearful": "scared, panicked",
             "surprised": "surprised, alarmed", "disgusted": "disgusted, annoyed", "neutral": "calm, natural"}


def auto_style(r: dict) -> str:
    """Yönetmen notu yoksa duygu etiketinden ve yoğunluktan talimat üret."""
    base = EMO_STYLE.get(r.get("emotion") or "", "natural")
    t = float(r.get("intensity") or 0.5)
    return (f"very {base}, intense" if t >= 0.75 else base if t >= 0.35 else f"subtly {base}") + ", dramatic game dialogue"


def _voxcpm_items(ctx: Ctx, rows: list[dict]) -> list[dict]:
    """VoxCPM2: karakter bankasıyla klonlama + "(talimat)" ile duygu (translations.csv style sütunu ya da otomatik)."""
    c = ctx.cfg["tts"].get("voxcpm", {}) or {}
    seeds = list(ctx.cfg["tts"]["seeds"])[: int(c.get("n_candidates", 1))] or [11]
    items = []
    for r in rows:
        d = ctx.project.wdir("tts", r["seg_id"])
        style = r.get("tr_style") or (auto_style(r) if c.get("auto_style", True) else "")
        # 3. aday talimatsız: tek kelimelik repliklerde talimat bazen telaffuzu bozuyor ("Gidelim" → "Kitilim")
        plain_k = 2 if c.get("plain_candidate", True) else -1
        outs = [{"cid": f"vx{k + 1}", "seed": int(s), "path": str(d / f"vx{k + 1}.wav"),
                 "ref": str(bank_ref(ctx, r["char"])), "style": "" if k == plain_k else style}
                for k, s in enumerate(seeds)]
        # Türkçe devam istemi ("vt" adayları): aynı karakterin daha önce iyi çıkmış bir Türkçe repliği
        # prompt_wav+prompt_text olarak verilir, tını yine bankadan (reference_wav). Model İngilizce referansın
        # vurgu/tonlamasını değil Türkçe konuşmanın akışını sürdürür (kullanıcı: "vurgu tonlama bozuk").
        nt = int(c.get("tr_prompt_candidates", 0))
        all_seeds = list(ctx.cfg["tts"]["seeds"])
        for j, (pr, pc) in enumerate(_pick_tr_prompts(ctx, r, nt)):
            outs.append({"cid": f"vt{j + 1}", "seed": int(all_seeds[j % len(all_seeds)]),
                         "path": str(d / f"vt{j + 1}.wav"), "ref": str(bank_ref(ctx, r["char"])), "style": "",
                         "prompt_wav": str(ctx.project.abs(pc["tts"])), "prompt_text": speakable_tr(pr["text_tr"]) + " ",
                         "prompt_seg": pr["seg_id"]})
        items.append({"id": r["seg_id"], "text": speakable_tr(r["text_tr"]), "outs": outs})
    return items


def _pick_tr_prompts(ctx: Ctx, r: dict, k: int) -> list[tuple[dict, dict]]:
    """Aynı karakterin QA'dan geçmiş (CER <= %5), 1-6 sn'lik seçilmiş Türkçe adayları; duygu ve yoğunluğu
    hedef repliğe yakın olanlar önce. İstem zinciri oluşmasın diye kendisi istemle üretilmiş adaylar alınmaz."""
    if k <= 0:
        return []
    pool = []
    for o, x in _PROMPT_POOL:
        if o["seg_id"] == r["seg_id"] or o.get("char") != r.get("char"):
            continue
        dur = float(x.get("dub_speech_dur") or 0)
        cer = x.get("cer") if x.get("cer") is not None else 1.0
        if (x.get("qa") == "fail" or x.get("prompt_seg") or x["id"].endswith("+vc") or not x.get("tts")
                or cer > 0.05 or not 1.0 <= dur <= 6.0):
            continue
        sc = (float(x.get("score") or 0) + (0.3 if o.get("emotion") == r.get("emotion") else 0.0)
              - 0.3 * abs(float(o.get("intensity") or 0.5) - float(r.get("intensity") or 0.5)) + 0.05 * min(dur, 3.0))
        pool.append((sc, o, x))
    pool.sort(key=lambda t: t[0], reverse=True)
    return [(o, x) for _, o, x in pool[:k]]


_PROMPT_POOL: list[tuple[dict, dict]] = []


def _snapshot_prompt_pool(ctx: Ctx) -> None:
    """Her satırın seçili Yol A adayını (satır, aday kopyası) olarak sakla; run() adayları sıfırlamadan önce çağrılır."""
    _PROMPT_POOL.clear()
    for o in ctx.manifest.rows:
        A, b = o.get("candidates", {}).get("A", []), (o.get("best") or {}).get("A")
        if isinstance(b, int) and b < len(A):
            _PROMPT_POOL.append(({k: o.get(k) for k in ("seg_id", "char", "text_tr", "emotion", "intensity")}, dict(A[b])))


def _same_out(cand: dict, o: dict) -> bool:
    return (cand.get("seed") == o.get("seed") and (cand.get("style") or "") == (o.get("style") or "")
            and cand.get("prompt_seg") == o.get("prompt_seg"))


BACKENDS = {
    "chatterbox_mtl": ("tts", "tts_worker.py", _chatterbox_items),
    "omnivoice": ("omni", "omni_worker.py", _omnivoice_items),
    "xtts": ("xtts", "xtts_worker.py", _xtts_items),
    "voxcpm": ("voxcpm", "voxcpm_worker.py", _voxcpm_items),
}


def run(ctx: Ctx) -> None:
    if not ctx.want("A"):
        return
    c = ctx.cfg["tts"]
    rows = [r for r in ctx.manifest.select("tts", ctx.force, ctx.only)
            if r["status"].get("translate") == "ok" and r.get("text_tr")]
    if not rows:
        return
    missing_bank = [r for r in rows if not bank_ref(ctx, r.get("char") or "").exists()]
    for r in missing_bank:
        add_error(r, "tts", f"ses bankası yok: bank/{r.get('char')}/ref.wav (önce speakers)")
    rows = [r for r in rows if r not in missing_bank]
    backends = c.get("backends") or [c.get("backend", "chatterbox_mtl")]
    # tts.reuse_existing: dosyası duran ve aynı ayarla (seed/talimat/istem) üretilmiş adaylar yeniden
    # üretilmez; sadece yeni adaylar üretilir (VoxCPM2 bu kartta aday başına ~15 sn).
    reuse = bool(c.get("reuse_existing", False))
    keep = {"id", "backend", "seed", "tts", "src", "tts_dur", "exaggeration", "cfg_weight", "duration_target",
            "ref_kind", "xtts_speed", "style", "prompt_seg"}
    _snapshot_prompt_pool(ctx)
    any_ok: dict[str, bool] = {r["seg_id"]: False for r in rows}
    for r in rows:
        # çeviri (text_tr_time) adaydan sonra değiştiyse aday eski metni okuyor: yeniden üretilir
        tt = float(r.get("text_tr_time") or 0)
        old = [x for x in r.get("candidates", {}).get("A", []) if not x["id"].endswith("+vc")
               and x.get("tts") and ctx.project.abs(x["tts"]).exists()
               and ctx.project.abs(x["tts"]).stat().st_mtime >= tt] if reuse else []
        r.setdefault("candidates", {})["A"] = [{**{k: v for k, v in x.items() if k in keep}, "src": x["tts"]} for x in old]
        any_ok[r["seg_id"]] = bool(old)
        r.setdefault("best", {}).pop("A", None)

    for be in backends:
        if be not in BACKENDS:
            console.print(f"[red]Bilinmeyen tts backend: {be}[/]")
            continue
        env, script, builder = BACKENDS[be]
        items = builder(ctx, rows)
        if reuse:
            for it in items:
                r = ctx.manifest.get(it["id"])
                have = {x["id"]: x for x in r["candidates"]["A"] if x.get("backend") == be}
                todo = []
                for o in it["outs"]:
                    x = have.get(o["cid"])
                    if x is not None and _same_out(x, o) and Path(o["path"]).exists():
                        continue  # aynı aday zaten var
                    if x is not None:  # ayar değişmiş: eskisini at, yeniden üret
                        r["candidates"]["A"] = [y for y in r["candidates"]["A"] if y is not x]
                    todo.append(o)
                it["outs"] = todo
            items = [it for it in items if it["outs"]]
            console.print(f"tts {be}: {sum(len(it['outs']) for it in items)} yeni aday üretilecek (mevcutlar korunuyor)")
            if not items:
                continue
        opts = {"language_id": c["language_id"], "max_chars": int(c.get("max_chars_per_chunk", 250)),
                "t3_model": c.get("chatterbox_t3_model", "v3"),
                **((c.get("omnivoice") or {}) if be == "omnivoice" else {}),
                **((c.get("xtts") or {}) if be == "xtts" else {}),
                **((c.get("voxcpm") or {}) if be == "voxcpm" else {})}
        try:
            res = run_worker(ctx.project, env, script, {"items": items, "options": opts}, label=f"tts_{be}")
        except WorkerError as e:
            console.print(f"[red]tts {be} başarısız:[/] {e}")
            for r in rows:
                r.setdefault("tts_backend_errors", {})[be] = str(e)[:500]
            continue
        for it in items:
            r = ctx.manifest.get(it["id"])
            out = res.get("results", {}).get(it["id"]) or {}
            for o in it["outs"]:
                got = out.get(o["cid"])
                if not got:
                    continue
                r["candidates"]["A"].append({
                    "id": o["cid"], "backend": be, "seed": o["seed"],
                    "tts": ctx.project.rel(got["path"]), "src": ctx.project.rel(got["path"]),
                    "tts_dur": round(float(got["dur"]), 3),
                    **({"exaggeration": it["exaggeration"], "cfg_weight": o.get("cfg_weight", it["cfg_weight"]),
                        "ref_kind": o.get("ref_kind", "bank")} if be == "chatterbox_mtl" else {}),
                    **({"duration_target": o.get("duration")} if be == "omnivoice" else {}),
                    **({"ref_kind": o.get("ref_kind"), "xtts_speed": got.get("speed")} if be == "xtts" else {}),
                    **({"style": o.get("style"), "prompt_seg": o.get("prompt_seg")} if be == "voxcpm" else {}),
                })
                any_ok[it["id"]] = True
            err = res.get("errors", {}).get(it["id"])
            if err:
                r.setdefault("tts_backend_errors", {})[be] = str(err)[:500]
    for r in rows:
        if any_ok[r["seg_id"]]:
            clear_errors(r, "tts")
            set_status(r, "tts", "ok")
        else:
            add_error(r, "tts", json.dumps(r.get("tts_backend_errors", {}), ensure_ascii=False) or "aday üretilemedi")
    ctx.save()
