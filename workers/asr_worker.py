"""ASR worker (env: core) — faster-whisper.

options: model, compute_type, language, word_timestamps, beam_size, detect_lang (dil olasılıkları da döner)
item   : {id, path, condition_on_previous_text?, vad_filter?, return_segments?, scan?: {win, hop}, clip?: [t0, t1]}
clip: dosyanın sadece bu aralığını yaz; zamanlar dosyaya göre (t0 eklenmiş) döner.
çıktı  : {text, words:[{w,s,e}], no_speech_prob, avg_logprob, language, segments?}
options.hallucination_silence_threshold: VAD'sız taramada (arka plan konuşması) sessiz/gürültülü boşluklarda
uydurulan metni atlar.
scan: dosyayı kısa örtüşen pencerelerle (VAD'sız) tarar; tek geçişte kaçan kısa/arka plan replikleri için.
Komşu pencerelerde tekrar bulunan aynı konuşmadan avg_logprob'u en iyi olan kalır; "hits" kaç pencerede
bulunduğunu söyler.
"""
from _common import add_nvidia_dll_dirs, log, read_mono, run

add_nvidia_dll_dirs()


def setup(job):
    import ctranslate2
    from faster_whisper import WhisperModel

    o = job.get("options", {})
    device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    if device == "cpu" and job.get("require_cuda", True):
        raise RuntimeError("ctranslate2 CUDA cihazı görmüyor (cuBLAS/cuDNN 9 DLL'leri?)")
    ct = o.get("compute_type", "float16") if device == "cuda" else "int8"
    log(f"faster-whisper {o.get('model', 'large-v3')} · {device} · {ct}")
    try:
        model = WhisperModel(o.get("model", "large-v3"), device=device, compute_type=ct)
    except (RuntimeError, ValueError) as e:  # örn. 6 GB VRAM'de bellek yetmezse
        if device == "cuda" and ct == "float16":
            log(f"float16 yüklenemedi ({e}); int8_float16 deneniyor")
            model = WhisperModel(o.get("model", "large-v3"), device=device, compute_type="int8_float16")
        else:
            raise
    return {"model": model, "o": o}


def _scan(state, it):
    o, m = state["o"], state["model"]
    audio, _ = read_mono(it["path"], 16000)
    win, hop = float(it["scan"].get("win", 10)), float(it["scan"].get("hop", 5))
    found = []
    t0 = 0.0
    while t0 < len(audio) / 16000 - 0.5:
        clip = audio[int(t0 * 16000): int((t0 + win) * 16000)]
        segs, _ = m.transcribe(clip, language=o.get("language"), beam_size=int(o.get("beam_size", 5)),
                               word_timestamps=True, condition_on_previous_text=False, vad_filter=False)
        for s in segs:
            ws = [w for w in (s.words or []) if w.word.strip()]
            if not ws:
                continue
            found.append({"start": round(t0 + ws[0].start, 3), "end": round(t0 + ws[-1].end, 3), "text": s.text.strip(),
                          "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                          "compression_ratio": round(s.compression_ratio, 3),
                          "words": [{"w": w.word.strip(), "s": round(t0 + w.start, 3), "e": round(t0 + w.end, 3),
                                     "p": round(w.probability, 3)} for w in ws]})
        t0 += hop
    found.sort(key=lambda d: d["start"])
    merged = []
    for d in found:
        dup = next((k for k, x in enumerate(merged) if min(x["end"], d["end"]) - max(x["start"], d["start"])
                    > 0.3 * min(x["end"] - x["start"], d["end"] - d["start"])), None)
        if dup is None:
            merged.append({**d, "hits": 1})
            continue
        hits = merged[dup]["hits"] + 1
        if d["avg_logprob"] > merged[dup]["avg_logprob"]:
            merged[dup] = {**d}
        merged[dup]["hits"] = hits
    return {"text": " ".join(d["text"] for d in merged), "words": [w for d in merged for w in d["words"]],
            "no_speech_prob": None, "avg_logprob": None, "language": o.get("language"), "lang_probs": None,
            "segments": merged}


def item(state, it, job):
    o = state["o"]
    if it.get("scan"):
        return _scan(state, it)
    audio, _ = read_mono(it["path"], 16000)
    t0 = 0.0
    if it.get("clip"):
        t0 = max(0.0, float(it["clip"][0]))
        audio = audio[int(t0 * 16000): int(float(it["clip"][1]) * 16000)]
    segments, info = state["model"].transcribe(
        audio, language=o.get("language"), beam_size=int(o.get("beam_size", 5)),
        word_timestamps=bool(o.get("word_timestamps", True)),
        condition_on_previous_text=bool(it.get("condition_on_previous_text", False)),
        vad_filter=bool(it.get("vad_filter", False)),
        temperature=[0.0, 0.2, 0.4],
        hallucination_silence_threshold=o.get("hallucination_silence_threshold"),
    )
    texts, words, nsp, alp, segs = [], [], [], [], []
    for s in segments:
        if it.get("return_segments"):
            segs.append({"start": round(t0 + s.start, 3), "end": round(t0 + s.end, 3), "text": s.text.strip(),
                         "avg_logprob": round(s.avg_logprob, 3), "no_speech_prob": round(s.no_speech_prob, 3),
                         "compression_ratio": round(s.compression_ratio, 3),
                         "words": [{"w": w.word.strip(), "s": round(t0 + w.start, 3), "e": round(t0 + w.end, 3),
                                    "p": round(w.probability, 3)} for w in (s.words or [])]})
        texts.append(s.text)
        nsp.append(s.no_speech_prob)
        alp.append(s.avg_logprob)
        for w in s.words or []:
            words.append({"w": w.word.strip(), "s": round(t0 + w.start, 3), "e": round(t0 + w.end, 3),
                          "p": round(w.probability, 3)})
    lang_probs = None
    if o.get("detect_lang"):
        # Whisper dil tanıma (dil zorlanmadan): aynı metnin adayları arasında İngilizce aksanlı okunanı ayırır
        lid_audio = read_mono(it["lid_path"], 16000)[0] if it.get("lid_path") else audio
        _, _, probs = state["model"].detect_language(lid_audio)
        lang_probs = {k: round(float(v), 4) for k, v in probs if k in set(o["detect_lang"])}
    return {
        "lang_probs": lang_probs,
        "text": " ".join(t.strip() for t in texts).strip(),
        "words": words,
        "no_speech_prob": round(max(nsp), 3) if nsp else None,
        "avg_logprob": round(sum(alp) / len(alp), 3) if alp else None,
        "language": info.language,
        "segments": segs if it.get("return_segments") else None,
    }


if __name__ == "__main__":
    run(setup, item)
