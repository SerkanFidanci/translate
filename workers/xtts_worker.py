"""XTTS-v2 worker'ı (env: xtts) — coqui-tts (idiap fork), Türkçe zero-shot klonlama, 24 kHz.

Lisans: XTTS-v2 ağırlıkları Coqui Public Model License (CPML, ticari olmayan). İndirme ancak
options.tos_agreed = true ise (kullanıcı açıkça kabul ettiyse) yapılır.

options: language_id ("tr"), temperature (0.75), tos_agreed, duration_retry (true), max_speed_adjust (0.3)
item   : {id, text, outs:[{cid, seed, path, refs:[wav,...], duration}]}
çıktı  : {cid: {path, dur, sr, speed}}
Süre: üretilen konuşma süresi hedeften %12'den fazla saparsa XTTS'in speed parametresiyle bir kez yeniden
üretilir (sonradan esnetmekten daha doğal; model hızı içeriden ayarlar).
"""
import os
import random

import numpy as np
import soundfile as sf

from _common import log, run, torch_device

SR = 24000


def _speech_dur(y: np.ndarray, thresh_db: float = -45.0) -> float:
    hop = int(SR * 0.01)
    n = len(y) // hop
    if n == 0:
        return 0.0
    db = 20 * np.log10(np.sqrt(np.mean(y[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-10)
    idx = np.flatnonzero(db > thresh_db)
    return 0.0 if idx.size == 0 else (idx[-1] - idx[0] + 1) * hop / SR


def setup(job):
    o = job.get("options", {})
    if not o.get("tos_agreed"):
        raise RuntimeError("XTTS-v2 lisansı (CPML) kabul edilmedi: config tts.xtts.tos_agreed: true gerekli")
    os.environ["COQUI_TOS_AGREED"] = "1"
    import torch
    from TTS.api import TTS

    dev = torch_device(job.get("require_cuda", True))
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(dev)
    model = tts.synthesizer.tts_model
    log("XTTS-v2 hazır")
    return {"model": model, "torch": torch, "cache": {}}


def _latents(st, refs):
    key = tuple(refs)
    if key not in st["cache"]:
        st["cache"][key] = st["model"].get_conditioning_latents(audio_path=list(refs), gpt_cond_len=6,
                                                                 max_ref_length=20)
    return st["cache"][key]


def _gen(st, text, lang, refs, seed, temperature, speed):
    torch = st["torch"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    gpt, spk = _latents(st, refs)
    with torch.inference_mode():
        out = st["model"].inference(text, lang, gpt, spk, temperature=temperature, speed=float(speed),
                                    enable_text_splitting=True)
    wav = out["wav"]
    wav = wav.detach().cpu().numpy() if hasattr(wav, "detach") else np.asarray(wav)
    return wav.astype(np.float32).reshape(-1)


def item(st, it, job):
    o = job.get("options", {})
    lang = o.get("language_id", "tr")
    temp = float(o.get("temperature", 0.75))
    max_adj = float(o.get("max_speed_adjust", 0.3))
    res = {}
    for out in it["outs"]:
        speed = 1.0
        y = _gen(st, it["text"], lang, out["refs"], int(out["seed"]), temp, speed)
        target = out.get("duration")
        if target and o.get("duration_retry", True):
            ratio = _speech_dur(y) / float(target) if target else 1.0
            if ratio > 0 and abs(ratio - 1) > 0.12:
                speed = float(min(max(ratio, 1 - max_adj), 1 + max_adj))
                y = _gen(st, it["text"], lang, out["refs"], int(out["seed"]), temp, speed)
        sf.write(out["path"], y, SR)
        res[out["cid"]] = {"path": out["path"], "dur": len(y) / SR, "sr": SR, "speed": round(speed, 3)}
    if st["torch"].cuda.is_available():
        st["torch"].cuda.empty_cache()
    return res


if __name__ == "__main__":
    run(setup, item)
