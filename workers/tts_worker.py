"""Chatterbox Multilingual TTS worker'ı (env: tts).

options: language_id ("tr"), t3_model ("v3"), max_chars (250)
item   : {id, text, ref, exaggeration, cfg_weight, temperature, outs:[{cid, seed, path}]}
çıktı  : {cid: {path, dur, sr}}
Notlar: model bir kez yüklenir; referans koşulları öğe başına bir kez hazırlanır, adaylar farklı
seed'lerle üretilir. 250 karakteri aşan replikler cümle bazında bölünüp birleştirilir.
"""
import random
import re

import numpy as np
import soundfile as sf

from _common import log, run, torch_device


def split_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    chunks, cur = [], ""
    for p in parts:
        if cur and len(cur) + 1 + len(p) > max_chars:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur} {p}".strip()
    if cur:
        chunks.append(cur)
    out = []
    for c in chunks:  # tek cümle hâlâ uzunsa virgüllerden böl
        if len(c) > max_chars:
            out += [x.strip() for x in re.split(r"(?<=[,;:])\s+", c) if x.strip()]
        else:
            out.append(c)
    return out


def setup(job):
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    dev = torch_device(job.get("require_cuda", True))
    t3 = job.get("options", {}).get("t3_model", "v3")
    try:
        model = ChatterboxMultilingualTTS.from_pretrained(device=dev, t3_model=t3)
    except Exception as e:  # noqa: BLE001
        log(f"t3_model={t3} yüklenemedi ({e}); varsayılan checkpoint deneniyor")
        model = ChatterboxMultilingualTTS.from_pretrained(device=dev)
        t3 = "default"
    log(f"Chatterbox multilingual hazır (t3={t3}, sr={model.sr})")
    return {"model": model, "torch": torch, "_extra": {"t3_model": t3}}


def item(state, it, job):
    torch, model = state["torch"], state["model"]
    o = job.get("options", {})
    lang = o.get("language_id", "tr")
    chunks = split_text(it["text"], int(o.get("max_chars", 250)))
    gap = np.zeros(int(0.12 * model.sr), dtype=np.float32)
    res = {}
    prepared = None
    for out in it["outs"]:
        ref = out.get("ref", it["ref"])  # aday bazında referans (örn. duygu için orijinal replik + banka)
        if ref != prepared:
            model.prepare_conditionals(ref, exaggeration=float(it["exaggeration"]))
            prepared = ref
        cfg_w = float(out.get("cfg_weight", it["cfg_weight"]))
        seed = int(out["seed"])
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        pieces = []
        with torch.inference_mode():
            for c in chunks:
                wav = model.generate(c, language_id=lang, exaggeration=float(it["exaggeration"]),
                                     cfg_weight=cfg_w, temperature=float(it.get("temperature", 0.8)))
                pieces += [wav.squeeze(0).detach().cpu().numpy().astype(np.float32), gap]
        y = np.concatenate(pieces[:-1]) if pieces else np.zeros(1, dtype=np.float32)
        sf.write(out["path"], y, model.sr)
        res[out["cid"]] = {"path": out["path"], "dur": len(y) / model.sr, "sr": model.sr}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


if __name__ == "__main__":
    run(setup, item)
