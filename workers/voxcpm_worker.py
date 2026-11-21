"""VoxCPM2 worker'ı (env: voxcpm) — OpenBMB VoxCPM2 (2B, Apache-2.0), 30 dil, 48 kHz.

"Kontrollü klonlama": reference_wav_path ile karakterin tınısı, metnin başındaki "(talimat)" ile duygu/üslup
(örn. "(shocked and alarmed, urgent)Dur, ne oluyor?"). 6 GB kartta ~5,6 GB tepe VRAM: sığıyor ama yavaş.

options: cfg_value (2.0), inference_timesteps (10)
item   : {id, text, outs:[{cid, seed, path, ref, style, prompt_wav?, prompt_text?}]}
çıktı  : {cid: {path, dur, sr}}
"""
import random
import sys
import types

import numpy as np
import soundfile as sf

from _common import log, run, torch_device

# voxcpm "wetext" (Çince/İngilizce metin normalizasyonu) import ediyor; Windows'ta kurulamıyor ve Türkçede
# kullanılmıyor (normalize=False) → boş modül ile geç.
sys.modules.setdefault("wetext", types.ModuleType("wetext"))


def setup(job):
    import torch
    from voxcpm import VoxCPM

    dev = torch_device(job.get("require_cuda", True))
    model = VoxCPM.from_pretrained(job.get("options", {}).get("model", "openbmb/VoxCPM2"), load_denoiser=False,
                                   optimize=False, device=str(dev))
    log(f"VoxCPM2 hazır (sr={model.tts_model.sample_rate})")
    return {"model": model, "torch": torch}


def item(st, it, job):
    o = job.get("options", {})
    torch, model = st["torch"], st["model"]
    sr = model.tts_model.sample_rate
    res = {}
    for out in it["outs"]:
        seed = int(out["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        style = (out.get("style") or "").strip()
        kw = {"reference_wav_path": out["ref"], "cfg_value": float(o.get("cfg_value", 2.0)),
              "inference_timesteps": int(o.get("inference_timesteps", 10)), "normalize": False}
        if out.get("prompt_wav"):
            # birleşik mod: referans (tını, yalıtılmış) + Türkçe istemden devam (vurgu/tonlama akışı)
            kw.update(prompt_wav_path=out["prompt_wav"], prompt_text=out["prompt_text"])
            text = it["text"]
        else:
            text = f"({style}){it['text']}" if style else it["text"]
        wav = model.generate(text=text, **kw)
        y = np.asarray(wav, dtype=np.float32).reshape(-1)
        sf.write(out["path"], y, sr)
        res[out["cid"]] = {"path": out["path"], "dur": len(y) / sr, "sr": sr}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    run(setup, item)
