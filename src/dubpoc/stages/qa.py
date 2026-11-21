"""5.11 qa (env: core + diar + emo) — otomatik kalite ölçümü ve aday seçimi.

Metrikler (aday başına):
- wer / cer: faster-whisper (language=tr) çıktısı ile text_tr arasında, Türkçe normalizasyonla.
  Türkçe eklemeli bir dil olduğundan WER ek hatalarını iki kez cezalandırır; CER de raporlanır,
  durum kararı qa.text_metric ile seçilir (spec varsayılanı: wer).
- spk_sim: çıktı embedding'i ile bank ref_emb kosinüsü; raporlanan = spk_sim / baseline
  (baseline: karakterin bank DIŞI orijinal kliplerinin ref'e benzerliği → aynı ölçek).
- dur_ratio: fit sonrası konuşma süresi / orijinal konuşma süresi.
- emo_sim (ek): orijinal ve dublaj emotion2vec sınıf dağılımları arası 1 − JS ıraksaması
  (other/unknown çıkarılıp yeniden normalize). Basit embedding kosinüsü akustik benzerliği ödüllendirdiği
  için (False Resonance, 2026) dağılım karşılaştırması tercih edildi. Varsayılan ağırlık 0 (sadece rapor).
"""
from __future__ import annotations

import json

import jiwer
import numpy as np

from ..manifest import add_error, clear_errors, set_status
from ..textutil import normalize_tr
from ..workers import WorkerError, run_worker
from . import Ctx, console
from .asr import asr_options
from .emotion import emo_items, run_emo_worker

DROP_EMO = {"other", "unknown", "<unk>"}


def js_similarity(p: dict, q: dict) -> float | None:
    keys = sorted((set(p) | set(q)) - DROP_EMO)
    a = np.array([max(float(p.get(k, 0.0)), 0.0) for k in keys])
    b = np.array([max(float(q.get(k, 0.0)), 0.0) for k in keys])
    if a.sum() <= 0 or b.sum() <= 0:
        return None
    a, b = a / a.sum(), b / b.sum()
    m = 0.5 * (a + b)

    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))

    return round(1.0 - 0.5 * (kl(a, m) + kl(b, m)), 4)


def text_errors(hyp: str, ref: str) -> tuple[float, float]:
    h, r = normalize_tr(hyp), normalize_tr(ref)
    if not r:
        return (0.0, 0.0) if not h else (1.0, 1.0)
    if not h:
        return 1.0, 1.0
    return float(jiwer.wer(r, h)), float(jiwer.cer(r, h))


def _clamp(v: float) -> float:
    return float(min(max(v, 0.0), 1.0))


def prosody_score(dub_std: float | None, src_std: float | None) -> float | None:
    """Perde canlılığı: dublajın perde sapması (yarım ton) / orijinalinki. Tekdüze (robotik) okuma
    düşük puan alır; 0,7-1,6 kat aralığı tam puan, abartılı dalgalanma da hafif cezalanır."""
    if dub_std is None or not src_std or src_std <= 0:
        return None
    ratio = dub_std / src_std
    if ratio < 0.7:
        return round(max(0.0, ratio / 0.7), 4)
    if ratio > 1.6:
        return round(max(0.0, 1.6 / ratio), 4)
    return 1.0


def score_candidate(cand: dict, q: dict, ref_words: int | None = None) -> tuple[float, str]:
    w = q["weights"]
    metric = q.get("text_metric", "wer")
    # Tek/iki kelimelik replikte WER ikili (0 ya da 1): "Biri mi?" doğru okunup "BİRİMİ" yazılınca fail oluyordu.
    if ref_words is not None and ref_words <= int(q.get("short_ref_words", 2)):
        metric = "cer"
    m = cand.get(metric)
    fail_thr = float(q["fail_wer"]) if metric == "wer" else float(q.get("fail_cer", 0.25))
    parts = []
    if m is not None:
        parts.append((float(w.get("wer", 0.5)), _clamp(1 - m / fail_thr)))
    if cand.get("spk_sim_rel") is not None:
        parts.append((float(w.get("spk_sim", 0.3)), _clamp(cand["spk_sim_rel"])))
    # fit.use_slot: sonraki repliğe çarpmayan taşma/kısalık yarı ağırlıkla cezalanır (dur_score_ratio)
    dr = cand.get("dur_score_ratio", cand.get("dur_ratio"))
    if dr is not None:
        parts.append((float(w.get("dur", 0.2)),
                      _clamp(1 - abs(dr - 1) / (2 * float(q["dur_tolerance"])))))
    if cand.get("lang_tr") is not None and float(w.get("lang", 0)) > 0:
        parts.append((float(w["lang"]), _clamp(cand["lang_tr"])))
    if cand.get("prosody") is not None and float(w.get("prosody", 0)) > 0:
        parts.append((float(w["prosody"]), _clamp(cand["prosody"])))
    if cand.get("emo_sim") is not None and float(w.get("emo", 0)) > 0:
        parts.append((float(w["emo"]), _clamp(cand["emo_sim"])))
    tot = sum(pw for pw, _ in parts)
    score = round(sum(pw * v for pw, v in parts) / tot, 4) if tot else 0.0
    # sonraki repliğin üstüne binen aday: iki Türkçe replik aynı anda duyulur. Süre ağırlığı (0,1) bunu
    # engellemeye yetmiyordu ("Nihayet." 1,75 sn'lik uzatılmış okuma 0,62 sn'lik boşluğa seçildi).
    ov = float(cand.get("slot_overflow_s") or 0.0)
    if ov > float(q.get("overflow_tolerance_s", 0.1)):
        score = round(score * (1 - 0.5 * min(1.0, ov / 0.5)), 4)

    if not cand.get("fit") or m is None:
        return score, "fail"
    if metric == "wer" and (cand.get("wer") or 0) > float(q["fail_wer"]):
        return score, "fail"
    if metric == "cer" and (cand.get("cer") or 0) > float(q.get("fail_cer", 0.25)):
        return score, "fail"
    text_ok = m <= (float(q["wer_max"]) if metric == "wer" else float(q["cer_max"]))
    spk_ok = cand.get("spk_sim_rel") is not None and cand["spk_sim_rel"] >= float(q["spk_sim_min_relative"])
    dur_ok = dr is not None and abs(dr - 1) <= float(q["dur_tolerance"])
    return score, "ok" if (text_ok and spk_ok and dur_ok) else "review"


def run(ctx: Ctx) -> None:
    p, q = ctx.project, ctx.cfg["qa"]
    rows = [r for r in ctx.manifest.select("qa", ctx.force, ctx.only) if r["status"].get("fit") == "ok"]
    if not rows:
        return
    targets: dict[str, tuple[dict, dict]] = {}
    for r in rows:
        for path in ("A", "B"):
            if not ctx.want(path):
                continue
            for cand in r.get("candidates", {}).get(path, []):
                if cand.get("fit"):
                    targets[f"{r['seg_id']}|{path}|{cand['id']}"] = (r, cand)
    if not targets:
        return
    # kanal efektli (telefon/megafon) adaylarda ölçüm efektsiz kopyada: efekt tüm adaylarda aynı, ama Whisper ve
    # konuşmacı benzerliği dar bant/distorsiyonda düşüp iyi okumayı da "review"a atıyordu
    items = [{"id": k, "path": str(p.abs(c.get("fit_dry") or c["fit"])),
              **({"lid_path": str(p.abs(c["tts"]))} if c.get("tts") else {})}
             for k, (_, c) in targets.items()]
    src_l, tgt_l = ctx.cfg["source_lang"], ctx.cfg["target_lang"]

    # 1) Türkçe ASR → WER/CER
    try:
        res = run_worker(p, "core", "asr_worker.py",
                         {"items": items, "options": {**asr_options(ctx, tgt_l, "qa"), "detect_lang": [src_l, tgt_l]}},
                         label="qa_asr")
        for k, (r, cand) in targets.items():
            out = res.get("results", {}).get(k)
            if out is None:
                cand["qa_error"] = res.get("errors", {}).get(k)
                continue
            cand["asr_tr"] = out["text"].strip()
            cand["wer"], cand["cer"] = (round(v, 4) for v in text_errors(cand["asr_tr"], r["text_tr"]))
            lp = out.get("lang_probs") or {}
            if lp.get(src_l) is not None and lp.get(tgt_l) is not None:
                # Türkçelik: p(tr) / (p(tr) + p(en)). Kısa ünlemlerde gürültülü ama aynı metnin adayları arasında
                # İngilizce aksanlı okumayı ayırıyor (ölçüldü: "Herkes yapıyor" seçilen %1, alternatifler %96-98)
                cand["lang_tr"] = round(lp[tgt_l] / max(lp[tgt_l] + lp[src_l], 1e-6), 4)
    except WorkerError as e:
        console.print(f"[red]qa asr başarısız:[/] {e}")

    # 2) konuşmacı benzerliği
    try:
        emb_dir = p.wdir("qa", "emb")
        spk_items = [{"id": k, "path": it["path"], "out": str(emb_dir / (k.replace("|", "__").replace("+", "_") + ".npy"))}
                     for k, it in zip(targets, items)]
        res = run_worker(p, "diar", "diar_worker.py", {
            "action": "embed", "items": spk_items,
            "options": {"token_env": ctx.cfg["diarization"]["hf_token_env"],
                        "model": "pyannote/wespeaker-voxceleb-resnet34-LM"}}, label="qa_spk")
        cache: dict[str, tuple[np.ndarray, float | None] | None] = {}
        for it in spk_items:
            r, cand = targets[it["id"]]
            ch = r.get("char")
            if ch not in cache:
                bdir = p.work / "bank" / (ch or "")
                if (bdir / "ref_emb.npy").exists():
                    base = json.loads((bdir / "baseline.json").read_text(encoding="utf-8")) if (bdir / "baseline.json").exists() else {}
                    ref = np.load(bdir / "ref_emb.npy")
                    cache[ch] = (ref / (np.linalg.norm(ref) + 1e-9), base.get("qa_denominator"))
                else:
                    cache[ch] = None
            if it["id"] not in res.get("results", {}) or cache[ch] is None:
                continue
            e = np.load(it["out"])
            e = e / (np.linalg.norm(e) + 1e-9)
            ref, denom = cache[ch]
            sim = float(e @ ref)
            cand["spk_sim"] = round(sim, 4)
            cand["spk_sim_rel"] = round(sim / denom, 4) if denom else None
    except WorkerError as e:
        console.print(f"[red]qa konuşmacı benzerliği başarısız:[/] {e}")

    # 3) duygu benzerliği (ek metrik)
    if q.get("emotion_metrics", True):
        try:
            eitems = emo_items(ctx, [(k, it["path"]) for k, it in zip(targets, items)], "qa/emo")
            for it in eitems:
                it["emb_out"] = it["emb_out"].replace("|", "__").replace("+", "_")
            res = run_emo_worker(ctx, eitems, "qa_emotion")
            for k, (r, cand) in targets.items():
                out = res.get("results", {}).get(k)
                if not out:
                    continue
                cand["emotion"] = out["emotion"]
                cand["pitch_std"] = round(float(out["pitch_std"]), 3) if out.get("pitch_std") is not None else None
                cand["prosody"] = prosody_score(cand["pitch_std"], r.get("pitch_std"))
                cand["emo_match"] = out["emotion"] == r.get("emotion")
                if r.get("emotion_probs"):
                    cand["emo_sim"] = js_similarity(r["emotion_probs"], out["probs"])
        except WorkerError as e:
            console.print(f"[yellow]qa duygu metriği atlandı:[/] {e}")

    # 4) skor, en iyi aday, durum
    counts = {"ok": 0, "review": 0, "fail": 0}
    for r in rows:
        best = r.setdefault("best", {})
        seg_state = None
        for path in ("A", "B"):
            cands = r.get("candidates", {}).get(path, [])
            if not cands or not ctx.want(path):
                continue
            n_words = len(normalize_tr(r.get("text_tr") or "").split())
            for cand in cands:
                cand["score"], cand["qa"] = score_candidate(cand, q, n_words)
            ranked = sorted(range(len(cands)), key=lambda i: (cands[i]["qa"] != "fail", cands[i]["score"]), reverse=True)
            best[path] = ranked[0]
            if path == "A" or seg_state is None:
                seg_state = cands[ranked[0]]["qa"]
        if seg_state is None:
            add_error(r, "qa", "değerlendirilecek çıktı yok")
            continue
        clear_errors(r, "qa")
        set_status(r, "qa", seg_state)
        counts[seg_state] += 1
    ctx.save()
    console.print(f"qa: ok={counts['ok']} review={counts['review']} fail={counts['fail']}")
