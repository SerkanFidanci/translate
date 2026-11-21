"""5.3 diarize (env: diar) — konuşmacı ayrıştırma ve segment kesme.

- pyannote/speaker-diarization-community-1, exclusive (çakışmasız) çıktı.
- min_segment_s'den kısa segmentler komşu aynı konuşmacı segmentiyle birleşir; birleşemeyen kendi
  segmenti olarak kalır ("short"). Eskiden skip ediliyordu ve mix orada orijinal İngilizce sesi
  çalıyordu ("Fuck.", "Okay.", "Take it." gibi tek-iki kelimelik replikler; GTA testi, kullanıcı:
  "bazı kelimeler İngilizce kalmış"). Efor/ünlem olanları emotion aşaması keep_original yapar.
- HF token yokken duraklama tabanlı bölme kullanılır; göreli enerji eşiğinin altında kalan sessiz
  replikler ("Come on.", "Let's go.") Silero VAD (faster-whisper ile gelir) ile bulunup eklenir. VAD 20 sn'lik
  örtüşen parçalarda çalışır: tüm dosyada tek geçişte uzun müzikten sonraki kısa "Huh?" kaçıyordu (GTA 8:19).
  0,15-0,3 sn'lik kısa ünlemler de (enerji eşiğini geçip min_seg_s yüzünden atılanlar) parça olur.
- auto modda min_duration_s'den kısa dosyalar ve HF token yoksa dosyalar tek segment sayılır.
- Arka plan konuşması katmanı (diarization.bg_speech): DnR megafon/telsiz anonsunu, uzaktan bağırmayı
  "efekt/müzik" sayıyor (vocals RoFormer, ikinci DnR geçişi ve gürültü giderici de ayıramadı; GTA testi).
  Arka plan (müzik+efekt) kısa örtüşen pencerelerle Whisper'la taranır (Silero bu sesleri konuşma saymıyor),
  güvenilir segmentler "bg" katmanı satırları olur (<file>_bNN). Satırın sesi, arka plandan spektral
  kenetlemeyle çıkarılan konuşma tahminidir (audio.suppress_speech); mix dublajlanan yerde aynı bölgeyi
  arka plandan bastırır. Diyaloğun üstüne düşenler (diyalog sızıntısı) bg_leak_of ile işaretlenip dublajlanmaz.
- Her segment için konuşma sınırları (speech_start/end), loudness ve bir sonraki segmente kadar
  kullanılabilir boşluk (slot_end) hesaplanır; fit ve mix bunları kullanır.
"""
from __future__ import annotations

import os

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..workers import WorkerError, run_worker
from . import Ctx, console

FILE_FIELDS = ["file_id", "src_path", "src_sr", "src_channels", "src_lufs", "src_dur", "ingest_path",
               "file_voice_path", "bg_path", "bg_rms_dbfs", "separation", "char_hint", "music_path", "sfx_path"]
EDGE_PAD_S = 0.08
# bg satırının Whisper kelime aralığına eklenen pay (bastırma bölgesi de bu). Zamanlar _refine_timing ile
# inceltiliyor ama ±0,2 sn kayma kalabiliyor; bastırma çevreye göre kenetlediği için pay müziği pek etkilemiyor.
BG_PAD_S = 0.25
MERGE_GAP_S = 0.3


def _postprocess(turns: list[dict], min_seg: float) -> list[dict]:
    turns = sorted(turns, key=lambda t: t["start"])
    # aynı konuşmacının ardışık, kısa boşluklu turlarını tek replik yap
    merged: list[dict] = []
    for t in turns:
        if merged and merged[-1]["speaker"] == t["speaker"] and t["start"] - merged[-1]["end"] <= MERGE_GAP_S:
            merged[-1]["end"] = max(merged[-1]["end"], t["end"])
        else:
            merged.append(dict(t))
    # kısa segmentleri komşu aynı konuşmacıya kat
    out: list[dict] = []
    for i, t in enumerate(merged):
        if t["end"] - t["start"] >= min_seg:
            out.append(t)
            continue
        prev = out[-1] if out and out[-1]["speaker"] == t["speaker"] else None
        nxt = merged[i + 1] if i + 1 < len(merged) and merged[i + 1]["speaker"] == t["speaker"] else None
        if prev is not None:
            prev["end"] = t["end"]
        elif nxt is not None:
            nxt["start"] = t["start"]
        else:
            t["short"] = True  # kendi segmenti olarak kalır; boş/ünlemse emotion keep_original yapar
            out.append(t)
    return out


def add_missed_speech(parts: list[tuple[float, float]], vad: list[tuple[float, float]],
                      min_dur: float = 0.15, max_overlap: float = 0.3) -> list[tuple[float, float]]:
    """Enerji tabanlı parçaların kaçırdığı VAD konuşma bölgelerini yeni parça olarak ekle.
    Var olan parçalarla kayda değer örtüşen bölgeler eklenmez (sınırları zaten enerjiyle belirlendi);
    parçaları birleştirmek de yok: farklı konuşmacıların replikleri ayrı kalmalı."""
    out = list(parts)
    for s, e in vad:
        if e - s < min_dur:
            continue
        ov = sum(max(0.0, min(e, b) - max(s, a)) for a, b in parts)
        if ov / (e - s) <= max_overlap:
            # örtüşen kısmı kırp: yeni parça mevcut parçaların dışında kalan en uzun aralık
            free = [(s, e)]
            for a, b in parts:
                nxt = []
                for x, y in free:
                    if b <= x or a >= y:
                        nxt.append((x, y))
                        continue
                    if a > x:
                        nxt.append((x, a))
                    if b < y:
                        nxt.append((b, y))
                free = nxt
            if free:
                x, y = max(free, key=lambda t: t[1] - t[0])
                if y - x >= min_dur:
                    out.append((x, y))
    return sorted(out)


def _vad_regions(x, sr: int) -> list[tuple[float, float]]:
    """Silero VAD (faster-whisper paketiyle gelir, CPU). Yoksa boş liste."""
    try:
        import numpy as np
        from faster_whisper.vad import VadOptions, get_speech_timestamps
    except Exception:  # noqa: BLE001 - VAD isteğe bağlı
        return []
    y = audio.resample(x, sr, 16000).astype(np.float32)
    opts = VadOptions(threshold=0.35, min_speech_duration_ms=120, min_silence_duration_ms=200, speech_pad_ms=60)
    # Silero durumlu (RNN): uzun müzik/sessizlikten sonra kısa ünlemi kaçırabiliyor; örtüşen parçalarda çalıştır
    win, hop = 20 * 16000, 18 * 16000
    found: list[tuple[float, float]] = []
    for a in range(0, max(1, len(y)), hop):
        ts = get_speech_timestamps(y[a: a + win], opts)
        found += [((a + t["start"]) / 16000, (a + t["end"]) / 16000) for t in ts]
        if a + win >= len(y):
            break
    found.sort()
    out: list[tuple[float, float]] = []
    for s, e in found:  # parça sınırındaki kopyaları birleştir
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def pick_bg_speech(segments: list[dict], dialog: list[tuple[float, float]], cfg: dict,
                   rms_of=None) -> tuple[list[dict], list[dict]]:
    """Arka plandaki Whisper segmentlerinden güvenilir olanlar: (yeni satırlar, diyalog sızıntıları).
    Sızıntı = büyük kısmı bir diyalog segmentinin üstüne düşen konuşma (aynı replik efekt stem'ine de kaçmış);
    dublajlanmaz, mix o diyalog dublajlanınca arka plandan bastırır."""
    from ..textutil import normalize_en
    from .asr import HALLUCINATIONS

    keep, leaks = [], []
    for sg in segments:
        words = [w for w in sg.get("words") or [] if (w.get("w") or "").strip()]
        toks = normalize_en(sg.get("text") or "").split()
        if not words or not toks:
            continue
        wp = sum(float(w.get("p", 0)) for w in words) / len(words)
        text_n = " ".join(toks)
        if (float(sg.get("avg_logprob", -9)) < float(cfg.get("min_logprob", -0.8))
                or float(sg.get("no_speech_prob", 1)) > float(cfg.get("max_no_speech", 0.6))
                or wp < float(cfg.get("min_word_p", 0.45))
                or float(sg.get("compression_ratio", 0)) > float(cfg.get("max_compression", 2.4))
                or any(h in text_n for h in HALLUCINATIONS)
                or (len(toks) >= 3 and len(set(toks)) / len(toks) <= 0.34)):
            continue
        a, b = float(words[0]["s"]), max(float(words[-1]["e"]), float(words[0]["s"]) + 0.2)
        if rms_of is not None and rms_of(a, b) < float(cfg.get("min_rms_dbfs", -50.0)):
            continue
        ov = [(min(b, y) - max(a, x), k) for k, (x, y) in enumerate(dialog) if min(b, y) > max(a, x)]
        d = {"start": round(a, 3), "end": round(b, 3), "text": (sg.get("text") or "").strip(), "words": words,
             "avg_logprob": sg.get("avg_logprob"), "no_speech_prob": sg.get("no_speech_prob"), "word_p": round(wp, 3)}
        if sum(o for o, _ in ov) / (b - a) > float(cfg.get("max_dialog_overlap", 0.5)):
            d["dialog_index"] = max(ov)[1]
            leaks.append(d)
        else:
            keep.append(d)
    return keep, leaks


def tighten_words(words: list[dict], per_char_s: float = 0.11, base_s: float = 0.15) -> list[dict]:
    """Whisper (DTW) önceki müzik/efekti ilk kelimeye katıyor: tek kelimelik ünlemde "Shit." 61,65-64,01 sn
    çıkıyor (gerçekte ~63,5-64,0). Kelime sonu güvenilir; ilk kelimenin süresi harf sayısıyla sınırlanır."""
    if not words:
        return words
    w = dict(words[0])
    letters = sum(ch.isalpha() for ch in w.get("w", "")) or 1
    w["s"] = round(max(float(w["s"]), float(w["e"]) - (base_s + per_char_s * letters)), 3)
    return [w] + [dict(x) for x in words[1:]]


def matching_words(text: str, words: list[dict]) -> list[dict]:
    """Yeniden yazımda pencereye komşu replikler de girebilir ("Shit! ... Fuck!"): sadece hedef metnin
    sözcükleriyle başlayıp biten kesit alınır; yarıdan azı eşleşiyorsa boş döner."""
    from ..textutil import normalize_en

    ta = set(normalize_en(text).split())
    hit = [i for i, w in enumerate(words) if set(normalize_en(w.get("w", "")).split()) & ta]
    if not ta or not hit:
        return []
    sel = words[hit[0]: hit[-1] + 1]
    got = set(normalize_en(" ".join(w["w"] for w in sel)).split())
    return sel if len(ta & got) / len(ta) >= 0.5 else []


def _refine_timing(ctx: Ctx, path: str, lines: list[dict], dur: float) -> None:
    """Bulunan her repliği kendi çevresinde (±1 sn) yeniden yazdır: 10 sn'lik tarama penceresinde kenara düşen
    kelimelerin zamanı 1-2 sn kayıyordu (GTA: "Fuck!" 65,0 dendi, gerçekte 67,0-67,6)."""
    from .asr import asr_options

    if not lines:
        return
    items = [{"id": f"r{i}", "path": path, "clip": [max(0.0, t["start"] - 1.0), min(dur, t["end"] + 1.0)]}
             for i, t in enumerate(lines)]
    try:
        res = run_worker(ctx.project, "core", "asr_worker.py",
                         {"items": items, "options": asr_options(ctx, ctx.cfg["source_lang"])}, label="diarize_bgalign")
    except WorkerError as e:
        console.print(f"[yellow]arka plan replik zamanları inceltilemedi: {e}[/]")
        return
    for i, t in enumerate(lines):
        out = res.get("results", {}).get(f"r{i}") or {}
        words = matching_words(t["text"], [w for w in out.get("words") or [] if w.get("w")])
        if words:
            t["words"] = words
        t["words"] = tighten_words(t["words"])
        t["start"], t["end"] = t["words"][0]["s"], max(t["words"][-1]["e"], t["words"][0]["s"] + 0.2)


def add_bg_speech_rows(ctx: Ctx, fids: list[str]) -> None:
    """Dosyaların bg katmanı satırlarını (yeniden) oluştur; diyalog satırlarına dokunmaz."""
    p, cfg = ctx.project, ctx.cfg["diarization"].get("bg_speech") or {}
    if not cfg.get("enabled", True):
        return
    todo = [fid for fid in fids if ctx.manifest.rows_of_file(fid) and ctx.manifest.rows_of_file(fid)[0].get("bg_path")]
    if not todo:
        return
    from .asr import asr_options

    items = [{"id": f"bg:{fid}", "path": str(p.abs(ctx.manifest.rows_of_file(fid)[0]["bg_path"])),
              "scan": {"win": float(cfg.get("scan_win_s", 10)), "hop": float(cfg.get("scan_hop_s", 5))}}
             for fid in todo]
    try:
        res = run_worker(p, "core", "asr_worker.py", {"items": items, "options": asr_options(ctx, ctx.cfg["source_lang"])},
                         label="diarize_bgspeech")
    except WorkerError as e:
        console.print(f"[yellow]arka plan konuşması taranamadı: {e}[/]")
        return
    for fid in todo:
        out = res.get("results", {}).get(f"bg:{fid}")
        if out:
            build_bg_rows(ctx, fid, out.get("segments") or [])
    ctx.save()


def build_bg_rows(ctx: Ctx, fid: str, segments: list[dict]) -> None:
    p, cfg = ctx.project, ctx.cfg["diarization"].get("bg_speech") or {}
    sil_db = float(ctx.cfg["fit"]["silence_db"])
    rows_f = ctx.manifest.rows_of_file(fid)
    dialog_rows = sorted((r for r in rows_f if r.get("layer") != "bg"), key=lambda r: r["start"])
    base = {k: rows_f[0].get(k) for k in FILE_FIELDS}
    base_status = {k: ("ok" if k in ("ingest", "separate") else "pending") for k in rows_f[0]["status"]}
    x, sr = audio.read(p.abs(base["bg_path"]), mono=True)
    dur = len(x) / sr
    keep, leaks = pick_bg_speech(segments, [(r["start"], r["end"]) for r in dialog_rows], cfg,
                                 rms_of=lambda a, b: audio.rms_dbfs(x[int(a * sr): int(b * sr)]))
    _refine_timing(ctx, str(p.abs(base["bg_path"])), keep + leaks, dur)
    regions = [(max(0.0, t["start"] - BG_PAD_S), min(dur, t["end"] + BG_PAD_S)) for t in keep + leaks]
    # satırın sesi (klon/loudness/spektrum referansı): yumuşak kenetleme (k=0) → daha az müzik karışır.
    # mix'teki bastırma daha serttir (suppress_k_db).
    _, speech = audio.suppress_speech(x, sr, regions, k_db=0.0)
    bg_rows = []
    for i, t in enumerate(sorted(keep + leaks, key=lambda t: t["start"])):
        later = [u["start"] for u in keep if u["start"] > t["start"]]
        nxt = (min(later) - BG_PAD_S) if later else dur
        row = bg_row(ctx, base, base_status, f"{fid}_b{i + 1:02d}", t, speech, sr, dur, nxt, sil_db)
        if "dialog_index" in t:
            # diyalog sızıntısı: dublajlanmaz; mix, diyalog dublajlanınca bu bölgeyi arka plandan bastırır.
            # Diyalogda olmayan kelimeleri (GTA: "Two, one, jump." ↔ "Three, two, one.") asr ayrı satır yapar.
            row["bg_leak_of"] = dialog_rows[t["dialog_index"]]["seg_id"]
            row["status"]["diarize"] = "skip"
        bg_rows.append(row)
    ctx.manifest.replace_file_rows(fid, sorted(dialog_rows + bg_rows, key=lambda r: r["start"]))
    console.print(f"  {fid}: arka planda {len(keep)} konuşma bulundu (dublajlanacak), "
                  f"{len(leaks)} diyalog sızıntısı (dublajlı diyaloğun altından bastırılacak)")


def dialog_row(ctx: Ctx, base: dict, base_status: dict, seg_id: str, voice, vsr: int, start: float, end: float,
               next_start: float, label: str, sil_db: float) -> dict:
    """Ses stem'inden bir diyalog satırı (klip work/seg/<seg_id>.wav, konuşma sınırları, loudness)."""
    p = ctx.project
    clip = voice[int(start * vsr): int(end * vsr)]
    seg_path = p.wdir("seg") / f"{seg_id}.wav"
    audio.write(seg_path, clip, vsr)
    s0, s1 = audio.speech_bounds(clip, vsr, sil_db)
    loud, kind = audio.loudness(clip[int(s0 * vsr): int(s1 * vsr)] if s1 > s0 else clip, vsr)
    row = dict(base)
    row.update({
        "seg_id": seg_id, "start": round(start, 4), "end": round(end, 4),
        "dur_src": round(end - start, 4),
        "voice_path": p.rel(seg_path), "diar_label": label,
        "speech_start": round(s0, 4), "speech_end": round(s1, 4),
        "dur_speech": round(max(s1 - s0, 0.0), 4),
        "seg_loudness": round(loud, 2), "seg_loudness_kind": kind,
        # bir sonraki replik başlayana kadar segment başından itibaren kullanılabilir süre
        "slot_dur": round(max(next_start - start, end - start), 4),
        "cluster": None, "char": None,
        "status": {**base_status}, "errors": [],
    })
    clear_errors(row, "diarize")
    if s1 <= s0:
        row["status"]["diarize"] = "skip"
    else:
        set_status(row, "diarize", "ok")
    return row


def bg_row(ctx: Ctx, base: dict, base_status: dict, seg_id: str, t: dict, speech, sr: int, dur: float,
           nxt: float, sil_db: float) -> dict:
    """Arka plan konuşması satırı; t: {start, end, text, words (mutlak zaman), avg_logprob, no_speech_prob}.
    speech: arka plandan kenetlemeyle çıkarılmış konuşma tahmini (tüm dosya)."""
    p = ctx.project
    a, b = max(0.0, t["start"] - BG_PAD_S), min(dur, t["end"] + BG_PAD_S)
    start, end = max(0.0, a - EDGE_PAD_S), min(dur, b + EDGE_PAD_S)
    clip = speech[int(start * sr): int(end * sr)]
    seg_path = p.wdir("seg") / f"{seg_id}.wav"
    audio.write(seg_path, clip, sr)
    s0, s1 = audio.speech_bounds(clip, sr, sil_db)
    if s1 <= s0:
        s0, s1 = a - start, b - start
    loud, kind = audio.loudness(clip[int(s0 * sr): int(s1 * sr)], sr)
    row = dict(base)
    row.update({
        "seg_id": seg_id, "layer": "bg", "bg_region": [round(a, 3), round(b, 3)],
        "start": round(start, 4), "end": round(end, 4), "dur_src": round(end - start, 4),
        "voice_path": p.rel(seg_path), "diar_label": "BG",
        "speech_start": round(s0, 4), "speech_end": round(s1, 4), "dur_speech": round(s1 - s0, 4),
        "seg_loudness": round(loud, 2), "seg_loudness_kind": kind,
        "slot_dur": round(max(nxt - start, end - start), 4),
        "bg_asr": {"text": t["text"], "avg_logprob": t.get("avg_logprob"), "no_speech_prob": t.get("no_speech_prob"),
                   "words": [{**w, "s": round(w["s"] - start, 3), "e": round(w["e"] - start, 3)}
                             for w in t["words"]]},
        "cluster": None, "char": None, "status": dict(base_status), "errors": [],
    })
    set_status(row, "diarize", "ok")
    return row


def _single(dur: float) -> list[dict]:
    return [{"start": 0.0, "end": dur, "speaker": "SPEAKER_00"}]


def run(ctx: Ctx) -> None:
    p, cfg = ctx.project, ctx.cfg["diarization"]
    rows = ctx.manifest.select("diarize", ctx.force, ctx.only)
    fids = sorted({r["file_id"] for r in rows})
    if not fids:
        return
    token = os.environ.get(cfg["hf_token_env"] or "HF_TOKEN", "")
    plan: dict[str, list[dict] | None] = {}
    need: list[dict] = []
    for fid in fids:
        r0 = ctx.manifest.rows_of_file(fid)[0]
        if r0["status"].get("separate") not in ("ok",) or not r0.get("file_voice_path"):
            continue
        dur = float(r0["src_dur"])
        mode = cfg["mode"]
        single = mode == "never" or (mode == "auto" and dur < float(cfg["min_duration_s"]))
        if not single and not token:
            # Yedek: duraklamalardan böl; konuşmacı ataması speakers aşamasında embedding ile yapılır
            x, vsr = audio.read(p.abs(r0["file_voice_path"]), mono=True)
            parts = audio.split_on_pauses(x, vsr, min_pause_s=float(cfg.get("fallback_min_pause_s", 0.35)))
            if cfg.get("fallback_vad", True):
                n0 = len(parts)
                parts = add_missed_speech(parts, _vad_regions(x, vsr))
                if len(parts) > n0:
                    console.print(f"  {fid}: VAD enerji eşiğinin kaçırdığı {len(parts) - n0} sessiz konuşma parçası ekledi")
            min_short = float(cfg.get("fallback_min_short_s", 0.15))
            if min_short > 0:
                n0 = len(parts)
                short = audio.split_on_pauses(x, vsr, min_pause_s=float(cfg.get("fallback_min_pause_s", 0.35)),
                                              min_seg_s=min_short)
                parts = add_missed_speech(parts, short, min_dur=min_short)
                if len(parts) > n0:
                    console.print(f"  {fid}: {len(parts) - n0} kısa ünlem parçası eklendi (< 0,3 sn)")
            console.print(f"[yellow]{fid}: {cfg['hf_token_env']} yok → duraklama tabanlı bölme ({len(parts)} parça); "
                          "konuşmacılar speakers aşamasında embedding ile atanacak.[/]")
            ctx.notes.append(f"{fid}: HF token yok → pyannote yerine duraklama tabanlı bölme kullanıldı")
            plan[fid] = [{"start": s, "end": e, "speaker": f"P{i:02d}"} for i, (s, e) in enumerate(parts)] or _single(dur)
            continue
        if single:
            plan[fid] = _single(dur)
        else:
            plan[fid] = None
            need.append({"id": fid, "path": str(p.abs(r0["file_voice_path"]))})

    if need:
        try:
            res = run_worker(p, "diar", "diar_worker.py", {
                "action": "diarize", "items": need,
                "options": {"token_env": cfg["hf_token_env"], "model": "pyannote/speaker-diarization-community-1"},
            }, label="diarize")
            for it in need:
                turns = res.get("results", {}).get(it["id"])
                if turns is None:
                    plan[it["id"]] = "error:" + str(res.get("errors", {}).get(it["id"], "çıktı yok"))  # type: ignore[assignment]
                else:
                    plan[it["id"]] = turns or _single(0.0)
        except WorkerError as e:
            for it in need:
                plan[it["id"]] = f"error:{e}"  # type: ignore[assignment]

    min_seg = float(cfg.get("min_segment_s", 0.4))
    sil_db = float(ctx.cfg["fit"]["silence_db"])
    for fid, turns in plan.items():
        file_rows = ctx.manifest.rows_of_file(fid)
        base = {k: file_rows[0].get(k) for k in FILE_FIELDS}
        base_status = dict(file_rows[0]["status"])
        if isinstance(turns, str):
            for r in file_rows:
                add_error(r, "diarize", turns[6:])
            continue
        voice, vsr = audio.read(p.abs(base["file_voice_path"]), mono=True)
        dur = len(voice) / vsr
        segs = _postprocess(turns, min_seg) if len(turns) > 1 else turns
        new_rows = []
        for i, t in enumerate(segs):
            prev_end = segs[i - 1]["end"] if i > 0 else 0.0
            next_start = segs[i + 1]["start"] if i + 1 < len(segs) else dur
            start = max(prev_end, t["start"] - EDGE_PAD_S, 0.0) if len(segs) > 1 else 0.0
            end = min(next_start, t["end"] + EDGE_PAD_S, dur) if len(segs) > 1 else dur
            new_rows.append(dialog_row(ctx, base, base_status, f"{fid}_s{i + 1:02d}", voice, vsr, start, end,
                                       next_start, t.get("speaker", "SPEAKER_00"), sil_db))
        ctx.manifest.replace_file_rows(fid, new_rows)
        console.print(f"  {fid}: {len(new_rows)} segment")
    ctx.save()
    add_bg_speech_rows(ctx, [fid for fid, turns in plan.items() if not isinstance(turns, str)])
