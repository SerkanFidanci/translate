"""5.5 asr (env: core, ayrı worker süreci) — İngilizce yazıya dökme.

subtitles.csv (id,text_en) içinde satır varsa ASR yerine o kullanılır. id; seg_id, file_id ya da
kaynak dosya adı (uzantısız, tek segmentli dosyalar için) olabilir.
"""
from __future__ import annotations

import csv
from pathlib import Path

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..textutil import normalize_en
from ..workers import WorkerError, run_worker
from . import Ctx, console

# Whisper'ın müzik/sessizlik üzerinde sık ürettiği kalıplar
HALLUCINATIONS = ("thank you for watching", "thanks for watching", "subtitles by", "please subscribe",
                  "amara org", "like and subscribe", "transcribed by", "see you in the next video")


def suspect_reason(r: dict) -> str | None:
    t = normalize_en(r.get("text_en") or "")
    if any(h in t for h in HALLUCINATIONS):
        return "halusinasyon"
    words = t.split()
    # Whisper tekrar döngüsü: "Twenty. Twenty. Twenty." / "Come on. Come on. Come on. ..." (müzik/çığlık üzerinde)
    if len(words) >= 3 and len(set(words)) / len(words) <= 0.34:
        return "halusinasyon"
    dur = float(r.get("dur_speech") or r.get("dur_src") or 0)
    wps = len(t.split()) / dur if dur > 0 else 0.0
    if dur >= 3 and wps < 0.6:
        return f"az kelime ({wps:.2f} kelime/sn)"
    if (r.get("asr_no_speech_prob") or 0) > 0.8:
        return "no_speech olasılığı yüksek"
    return None


def load_subtitles(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            k, v = (row.get("id") or "").strip(), (row.get("text_en") or "").strip()
            if k and v:
                out[k] = v
    return out


def asr_options(ctx: Ctx, language: str, model_key: str = "model") -> dict:
    c = ctx.cfg["asr"]
    return {"model": ctx.cfg["qa"]["asr_model"] if model_key == "qa" else c["model"],
            "compute_type": c["compute_type"], "beam_size": int(c.get("beam_size", 5)),
            "language": language, "word_timestamps": True}


def run(ctx: Ctx) -> None:
    p = ctx.project
    rows = ctx.manifest.select("asr", ctx.force, ctx.only)
    if not rows:
        return
    subs = load_subtitles(p.subtitles_csv)
    file_ctx = bool(ctx.cfg["asr"].get("file_context", True))
    items, file_items, ctx_rows = [], {}, []
    for r in rows:
        n_in_file = len(ctx.manifest.rows_of_file(r["file_id"]))
        key_candidates = [r["seg_id"], r["file_id"]]
        if n_in_file == 1 and r.get("src_path"):
            key_candidates.append(Path(r["src_path"]).stem)
        hit = next((subs[k] for k in key_candidates if k in subs), None)
        if hit:
            r["text_en"], r["words"], r["text_source"] = hit, [], "subtitles"
            clear_errors(r, "asr")
            set_status(r, "asr", "ok")
        elif r.get("layer") == "bg" and r.get("bg_asr"):
            # arka plan konuşması: diarize'daki Whisper taraması (tüm stem, bağlamlı) zaten yazdı
            b = r["bg_asr"]
            _apply(r, b["text"], b.get("words", []), "asr_bg", b)
        elif r.get("layer") == "bg":
            items.append({"id": r["seg_id"], "path": str(p.abs(r["voice_path"]))})
        elif file_ctx and n_in_file > 1 and r.get("file_voice_path"):
            # Bağlamlı ASR: çok segmentli dosyada Whisper tüm dosyayı bir kerede yazar, kelimeler zaman
            # damgasıyla segmentlere dağıtılır. Kısa parçaları tek tek yazdırmak bağlamı kaybettiriyor
            # (ölçüldü: "Change of plans" → "Trash reports", "the Dodo" → "Madoto").
            # vad_filter: 8-9 dk'lık dosyada müzikli uzun bölümler Whisper'ı "Twenty. Twenty. ..." döngüsüne
            # sokuyordu; o segmentler elle "-" yapılınca gerçek replikler İngilizce kaldı (GTA testi)
            file_items.setdefault(r["file_id"], {"id": f"file:{r['file_id']}", "path": str(p.abs(r["file_voice_path"])),
                                                 "condition_on_previous_text": True,
                                                 "vad_filter": bool(ctx.cfg["asr"].get("file_vad", True))})
            ctx_rows.append(r)
        else:
            items.append({"id": r["seg_id"], "path": str(p.abs(r["voice_path"]))})
    all_items = items + list(file_items.values())
    if all_items:
        try:
            res = run_worker(p, "core", "asr_worker.py",
                             {"items": all_items, "options": asr_options(ctx, ctx.cfg["source_lang"])}, label="asr")
        except WorkerError as e:
            for r in [ctx.manifest.get(it["id"]) for it in items] + ctx_rows:
                add_error(r, "asr", e)
            ctx.save()
            console.print(f"[red]asr başarısız:[/] {e}")
            return
        for it in items:
            r = ctx.manifest.get(it["id"])
            out = res.get("results", {}).get(it["id"])
            if out is None:
                add_error(r, "asr", res.get("errors", {}).get(it["id"], "çıktı yok"))
                continue
            _apply(r, out["text"].strip(), out.get("words", []), "asr", out)
        retry_seg = []
        assign = {fid: _assign_words(_dialog(ctx.manifest.rows_of_file(fid)),
                                     (res.get("results", {}).get(f"file:{fid}") or {}).get("words", []))
                  for fid in file_items}
        for fid in file_items:
            for w in unassigned_words(_dialog(ctx.manifest.rows_of_file(fid)),
                                      (res.get("results", {}).get(f"file:{fid}") or {}).get("words", [])):
                msg = (f"{fid}: {w['s']:.2f} sn'deki konuşma hiçbir segmente düşmedi ({w['w']!r}); dublajda "
                       "atlanır. Segmentlemeyi kontrol edin (diarize).")
                if msg not in ctx.notes:
                    ctx.notes.append(msg)
                console.print(f"[yellow]{msg}[/]")
        for r in ctx_rows:
            out = res.get("results", {}).get(f"file:{r['file_id']}")
            words = assign.get(r["file_id"], {}).get(r["seg_id"], [])
            if out is None or not words:
                retry_seg.append(r)  # bu segmente kelime düşmediyse tek başına yazdır
                continue
            text = " ".join(w["w"] for w in words).strip()
            rel = [{**w, "s": round(w["s"] - r["start"], 3), "e": round(w["e"] - r["start"], 3)} for w in words]
            _apply(r, text, rel, "asr_file", {"no_speech_prob": None, "avg_logprob": None})
        if retry_seg:
            items2 = [{"id": r["seg_id"], "path": str(p.abs(r["voice_path"]))} for r in retry_seg]
            try:
                res2 = run_worker(p, "core", "asr_worker.py",
                                  {"items": items2, "options": asr_options(ctx, ctx.cfg["source_lang"])}, label="asr_seg")
                for r in retry_seg:
                    out = res2.get("results", {}).get(r["seg_id"])
                    if out is None:
                        add_error(r, "asr", res2.get("errors", {}).get(r["seg_id"], "çıktı yok"))
                    else:
                        _apply(r, out["text"].strip(), out.get("words", []), "asr", out)
            except WorkerError as e:
                for r in retry_seg:
                    add_error(r, "asr", e)
        _retry_suspects(ctx, [ctx.manifest.get(it["id"]) for it in items] + ctx_rows)
    for fid in sorted({r["file_id"] for r in rows}):
        for r, d in bg_duplicates(ctx.manifest.rows_of_file(fid)):
            r["bg_leak_of"] = d["seg_id"]
            set_status(r, "asr", "skip")
            console.print(f"  {r['seg_id']}: arka plandaki {r['text_en']!r} = {d['seg_id']} repliğinin sızıntısı "
                          "→ dublajlanmaz, arka plandan bastırılır")
        split_leak_extras(ctx, fid)
    ctx.save()
    n_sub = sum(1 for r in rows if r.get("text_source") == "subtitles")
    console.print(f"asr: {len(rows)} segment ({n_sub} altyazıdan, {len(ctx_rows)} dosya bağlamıyla)")


def bg_duplicates(rows: list[dict], max_gap: float = 1.0, min_share: float = 0.6) -> list[tuple[dict, dict]]:
    """Arka plan (bg) satırı yakınındaki (max_gap sn) bir diyalog repliğiyle aynı sözleri taşıyorsa o diyaloğun
    efekt/müzik stem'ine sızmış kopyasıdır: ayrıca dublajlanırsa replik iki kez duyulur. Whisper'ın arka plan
    kelime zamanları kaydığı için diarize'daki zaman örtüşmesi bunu her zaman yakalayamıyor."""
    from ..manifest import is_inactive

    dialog = [d for d in rows if d.get("layer") != "bg" and d.get("text_en")]
    out = []
    for r in rows:
        if r.get("layer") != "bg" or is_inactive(r) or not r.get("text_en"):
            continue
        bt = set(normalize_en(r["text_en"]).split())
        for d in dialog:
            if max(d["start"] - r["end"], r["start"] - d["end"], 0.0) > max_gap:
                continue
            if bt and len(bt & set(normalize_en(d["text_en"]).split())) / len(bt) >= min_share:
                out.append((r, d))
                break
    return out


FILLERS = {"uh", "um", "oh", "ah", "hmm", "mm", "huh", "eh"}


def leak_extra_words(words: list[dict], dialog_text: str, d_start: float, d_end: float, min_p: float = 0.45,
                     edge_s: float = 0.3) -> list[dict]:
    """Diyalog sızıntısı sayılan arka plan segmentinin, diyalog metninde olmayan baş/son kelimeleri (mutlak zaman).
    GTA 3:16: arka planda "Two, one, jump." diyalogdaki "Three, two, one." ile örtüştüğü için tamamen sızıntı
    sayıldı; diyalog dublajlanınca bölge bastırıldı ve "jump" ne İngilizce ne Türkçe duyuldu."""
    dt = set(normalize_en(dialog_text).split())
    if not words or not dt:
        return []

    def foreign(w: dict) -> bool:
        toks = normalize_en(w.get("w", "")).split()
        return bool(toks) and not (set(toks) & dt)

    tail = []
    for w in reversed(words):
        if not foreign(w):
            break
        tail.insert(0, w)
    head = []
    for w in words:
        if not foreign(w):
            break
        head.append(w)
    for run, ok in ((tail, bool(tail) and float(tail[0]["s"]) >= d_end - edge_s),
                    (head, bool(head) and float(head[-1]["e"]) <= d_start + edge_s)):
        if not ok or len(run) == len(words):
            continue
        toks = normalize_en(" ".join(w["w"] for w in run)).split()
        if not toks or set(toks) <= FILLERS:
            continue
        if sum(float(w.get("p", 0)) for w in run) / len(run) >= min_p:
            return run
    return []


def split_leak_extras(ctx: Ctx, fid: str) -> list[dict]:
    """Sızıntı satırındaki fazladan kelimeleri yeni bir arka plan satırına ayır (konuşmacı: sızdığı diyaloğunki);
    sızıntının bastırma bölgesi kalan kısma daraltılır."""
    from .diarize import BG_PAD_S, FILE_FIELDS, bg_row

    p = ctx.project
    rows = ctx.manifest.rows_of_file(fid)
    by_id = {r["seg_id"]: r for r in rows}
    cfg = ctx.cfg["diarization"].get("bg_speech") or {}
    done = {r.get("split_from") for r in rows}
    added = []
    for r in rows:
        d = by_id.get(r.get("bg_leak_of") or "")
        b = r.get("bg_asr") or {}
        if r.get("layer") != "bg" or not d or not d.get("text_en") or not b.get("words") or r["seg_id"] in done:
            continue
        words = [{**w, "s": round(r["start"] + w["s"], 3), "e": round(r["start"] + w["e"], 3)} for w in b["words"]]
        extra = leak_extra_words(words, d["text_en"], d["start"], d["end"], float(cfg.get("min_word_p", 0.45)))
        if not extra:
            continue
        x, sr = audio.read(p.abs(r["bg_path"]), mono=True)
        dur = len(x) / sr
        line = {"start": extra[0]["s"], "end": max(extra[-1]["e"], extra[0]["s"] + 0.2),
                "text": " ".join(w["w"].strip() for w in extra), "words": extra,
                "avg_logprob": b.get("avg_logprob"), "no_speech_prob": b.get("no_speech_prob")}
        region = (max(0.0, line["start"] - BG_PAD_S), min(dur, line["end"] + BG_PAD_S))
        _, speech = audio.suppress_speech(x, sr, [region], k_db=0.0)
        ids = {q["seg_id"] for q in ctx.manifest.rows_of_file(fid)}
        n = 1
        while f"{fid}_b{n:02d}" in ids:
            n += 1
        later = [q["start"] for q in ctx.manifest.rows_of_file(fid) if q["start"] > line["end"] and not q.get("bg_leak_of")]
        base = {k: r.get(k) for k in FILE_FIELDS}
        base_status = {k: ("ok" if k in ("ingest", "separate") else "pending") for k in r["status"]}
        new = bg_row(ctx, base, base_status, f"{fid}_b{n:02d}", line, speech, sr, dur,
                     (min(later) - BG_PAD_S) if later else dur, float(ctx.cfg["fit"]["silence_db"]))
        new.update({"char": d.get("char"), "cluster": d.get("cluster"), "split_from": r["seg_id"]})
        set_status(new, "speakers", "ok")
        _apply(new, line["text"], new["bg_asr"]["words"], "asr_bg_split", b)
        # sızıntının bastırılacak kısmı: ayrılan kelimelerin dışında kalan
        a0, a1 = r["bg_region"]
        if extra[-1] is words[-1]:  # sondaki kelimeler ayrıldı
            r["bg_region"] = [a0, round(max(a0 + 0.1, line["start"] - 0.05), 3)]
        else:
            r["bg_region"] = [round(min(a1 - 0.1, line["end"] + 0.05), 3), a1]
        ctx.manifest.replace_file_rows(fid, sorted(ctx.manifest.rows_of_file(fid) + [new], key=lambda q: q["start"]))
        console.print(f"  {new['seg_id']}: {r['seg_id']} sızıntısında diyalogda olmayan {line['text']!r} ayrı satır "
                      f"oldu ({line['start']:.2f} sn, konuşmacı {new['char']})")
        added.append(new)
    return added


def _dialog(rows: list[dict]) -> list[dict]:
    return [r for r in rows if r.get("layer") != "bg"]


def _assign_words(seg_rows: list[dict], words: list[dict], max_gap: float = 1.0) -> dict[str, list]:
    """Dosya düzeyindeki kelimeleri segmentlere dağıt: kelime orta noktası bir segmentin içindeyse ona,
    değilse en yakın segmente (en fazla max_gap sn). Whisper kelime zamanları ±0,2-0,3 sn kayabildiği için
    katı pencere ilk/son kelimeleri kaçırıyordu."""
    out: dict[str, list] = {r["seg_id"]: [] for r in seg_rows}
    if not seg_rows:
        return out
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        best, dist = None, None
        for r in seg_rows:
            d = 0.0 if r["start"] <= mid < r["end"] else min(abs(mid - r["start"]), abs(mid - r["end"]))
            if dist is None or d < dist:
                best, dist = r, d
        if best is not None and dist <= max_gap:
            out[best["seg_id"]].append(w)
    return out


def unassigned_words(seg_rows: list[dict], words: list[dict], max_gap: float = 1.0, min_p: float = 0.6) -> list[dict]:
    """Hiçbir segmente (max_gap içinde) düşmeyen, güvenilir (p >= min_p) kelimeler: dublajda sessiz kalır."""
    out = []
    for w in words:
        mid = (w["s"] + w["e"]) / 2
        near = any(r["start"] - max_gap <= mid <= r["end"] + max_gap for r in seg_rows)
        if not near and float(w.get("p", 1.0)) >= min_p:
            out.append(w)
    return out


def _apply(r: dict, text: str, words: list, source: str, out: dict) -> None:
    r["text_en"], r["words"], r["text_source"] = text, words, source
    r["asr_no_speech_prob"] = out.get("no_speech_prob")
    r["asr_avg_logprob"] = out.get("avg_logprob")
    clear_errors(r, "asr")
    set_status(r, "asr", "ok")


def _retry_suspects(ctx: Ctx, rows: list[dict]) -> None:
    """Şüpheli sonuçlarda ayrıştırılmamış (miks) segment üzerinde tekrar dene; Whisper müziğe
    dayanıklıdır, ayrıştırma artefaktları ise halüsinasyonu tetikleyebilir."""
    p = ctx.project
    sus = [r for r in rows if r and r["status"].get("asr") == "ok" and suspect_reason(r)]
    if not sus:
        return
    items = []
    for r in sus:
        if r.get("separation") == "separated" and r.get("ingest_path"):
            x, sr = audio.read(p.abs(r["ingest_path"]), mono=True)
            seg = x[int(r["start"] * sr): int(r["end"] * sr)]
            out = p.wdir("seg_mix") / f"{r['seg_id']}.wav"
            audio.write(out, seg, sr)
            items.append({"id": r["seg_id"], "path": str(out)})
    res = {"results": {}}
    if items:
        console.print(f"asr: {len(items)} şüpheli sonuç miks ses üzerinde tekrar deneniyor")
        try:
            res = run_worker(p, "core", "asr_worker.py",
                             {"items": items, "options": asr_options(ctx, ctx.cfg["source_lang"])}, label="asr_retry")
        except WorkerError as e:
            console.print(f"[yellow]asr tekrar denemesi başarısız: {e}[/]")
    for r in sus:
        before = suspect_reason(r)
        out = res.get("results", {}).get(r["seg_id"])
        if out:
            cand = {**r, "text_en": out["text"].strip(), "asr_no_speech_prob": out.get("no_speech_prob")}
            if not suspect_reason(cand) or (out.get("avg_logprob") or -9) > (r.get("asr_avg_logprob") or -9):
                r.update({"text_en": cand["text_en"], "words": out.get("words", []),
                          "asr_no_speech_prob": out.get("no_speech_prob"), "asr_avg_logprob": out.get("avg_logprob"),
                          "text_source": "asr_mix"})
        reason = suspect_reason(r)
        if reason == "halusinasyon":
            add_error(r, "asr", f"olası Whisper halüsinasyonu: {r['text_en']!r} (subtitles.csv ile düzeltin)")
        elif reason:
            r["asr_suspect"] = reason
            msg = f"{r['seg_id']}: ASR şüpheli ({reason}): {r['text_en']!r}"
            if msg not in ctx.notes:
                ctx.notes.append(msg)
        console.print(f"  {r['seg_id']}: {before} → {reason or 'düzeldi'}: {r['text_en']!r}")
