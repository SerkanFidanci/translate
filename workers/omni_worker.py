"""OmniVoice TTS worker'ı (env: omni) — k2-fsa/OmniVoice, hedef süreli zero-shot klonlama.

options: language_id ("tr"), num_step (32), guidance_scale (2.0), dtype ("float16")
item   : {id, text, ref, ref_text, duration, outs:[{cid, seed, path}]}
çıktı  : {cid: {path, dur, sr}}
Not: ref_text verilirse OmniVoice referansı Whisper ile yazıya dökmez (VRAM ve süre tasarrufu).
"""
import random

import numpy as np
import soundfile as sf

from _common import log, read_mono, run, torch_device

SR = 24000


def setup(job):
    import torch
    from omnivoice import OmniVoice

    dev = torch_device(job.get("require_cuda", True))
    o = job.get("options", {})
    dtype = torch.float16 if o.get("dtype", "float16") == "float16" and dev.type == "cuda" else torch.float32
    model = OmniVoice.from_pretrained(o.get("model", "k2-fsa/OmniVoice"),
                                      device_map=f"{dev.type}:0" if dev.type == "cuda" else "cpu", dtype=dtype)
    log("OmniVoice hazır")
    return {"model": model, "torch": torch}


def item(state, it, job):
    torch, model = state["torch"], state["model"]
    o = job.get("options", {})
    ref, ref_sr = read_mono(it["ref"])
    if not it.get("ref_text"):
        # metin yoksa OmniVoice referansı Whisper ile kendisi yazıya döker; kırpmak güvenli
        ref = ref[: int(ref_sr * 12)]
    kwargs = dict(text=it["text"], language=o.get("language_id", "tr"),
                  ref_audio=(torch.from_numpy(ref)[None, :], ref_sr),
                  num_step=int(o.get("num_step", 32)), guidance_scale=float(o.get("guidance_scale", 2.0)))
    if it.get("ref_text"):
        kwargs["ref_text"] = it["ref_text"]
    res = {}
    for out in it["outs"]:
        kw = dict(kwargs)
        dur = out.get("duration", it.get("duration"))
        if dur:
            kw["duration"] = float(dur)
            kw["postprocess_output"] = False  # sessizlik kırpma süreyi bozmasın
        seed = int(out["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        with torch.inference_mode():
            audio = model.generate(**kw)
        y = np.asarray(audio[0], dtype=np.float32).reshape(-1)
        sf.write(out["path"], y, SR)
        res[out["cid"]] = {"path": out["path"], "dur": len(y) / SR, "sr": SR}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    run(setup, item)
