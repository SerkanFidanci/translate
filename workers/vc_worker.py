"""Ses dönüştürme worker'ı.

backend=seed_vc       (env: vc)  — tools/seed-vc (v1, konuşma checkpoint'i
                                   DiT_seed_v2_uvit_whisper_small_wavenet, 22.05 kHz, f0'sız)
backend=chatterbox_vc (env: tts) — ChatterboxVC (24 kHz); length_adjust desteklemez.

options: backend, diffusion_steps, inference_cfg_rate, fp16, seed_vc_repo
item   : {id, source, target, out, length_adjust}
çıktı  : {path, dur, sr}
Seed-VC modeli bir kez yüklenir; aynı hedef referansın (bank) koşulları önbelleğe alınır.
"""
import os
import sys
from argparse import Namespace

import numpy as np
import soundfile as sf

from _common import log, run, torch_device


# ---------------------------------------------------------------- Seed-VC
def _seed_setup(job):
    o = job["options"]
    repo = o["seed_vc_repo"]
    if not os.path.exists(os.path.join(repo, "inference.py")):
        raise RuntimeError(f"Seed-VC reposu yok: {repo}")
    os.chdir(repo)  # inference.py göreli yollar kullanıyor (checkpoints/, configs/)
    sys.path.insert(0, repo)
    import torch
    torch_device(job.get("require_cuda", True))
    import inference as svc  # noqa: E402  (tools/seed-vc/inference.py)

    args = Namespace(f0_condition=False, checkpoint=None, config=None, fp16=bool(o.get("fp16", True)))
    model, semantic_fn, _f0, vocoder_fn, campplus, mel_fn, mel_args = svc.load_models(args)
    log(f"Seed-VC hazır (sr={mel_args['sampling_rate']}, device={svc.device})")
    return {"backend": "seed_vc", "svc": svc, "torch": torch, "model": model, "semantic_fn": semantic_fn,
            "vocoder_fn": vocoder_fn, "campplus": campplus, "mel_fn": mel_fn,
            "sr": mel_args["sampling_rate"], "cache": {}}


def _seed_target(st, path):
    if path in st["cache"]:
        return st["cache"][path]
    import librosa
    torch, svc = st["torch"], st["svc"]
    sr = st["sr"]
    ref = librosa.load(path, sr=sr)[0][: sr * 25]
    ref = torch.tensor(ref).unsqueeze(0).float().to(svc.device)
    import torchaudio
    ref16 = torchaudio.functional.resample(ref, sr, 16000)
    S_ori = st["semantic_fn"](ref16)
    mel2 = st["mel_fn"](ref.float())
    feat2 = torchaudio.compliance.kaldi.fbank(ref16, num_mel_bins=80, dither=0, sample_frequency=16000)
    feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
    style2 = st["campplus"](feat2.unsqueeze(0))
    t2len = torch.LongTensor([mel2.size(2)]).to(mel2.device)
    prompt_cond = st["model"].length_regulator(S_ori, ylens=t2len, n_quantizers=3, f0=None)[0]
    st["cache"][path] = (mel2, style2, prompt_cond)
    return st["cache"][path]


def _seed_convert(st, src_path, tgt_path, steps, cfg_rate, length_adjust):
    import librosa
    import torchaudio
    torch, svc = st["torch"], st["svc"]
    sr, hop = st["sr"], 256
    with torch.no_grad():
        mel2, style2, prompt_cond = _seed_target(st, tgt_path)
        src = librosa.load(src_path, sr=sr)[0]
        src = torch.tensor(src).unsqueeze(0).float().to(svc.device)
        src16 = torchaudio.functional.resample(src, sr, 16000)
        if src16.size(-1) > 16000 * 30:
            raise RuntimeError("30 sn'den uzun kaynak desteklenmiyor (replikleri bölün)")
        S_alt = st["semantic_fn"](src16)
        mel = st["mel_fn"](src.float())
        tlen = torch.LongTensor([int(mel.size(2) * length_adjust)]).to(mel.device)
        cond = st["model"].length_regulator(S_alt, ylens=tlen, n_quantizers=3, f0=None)[0]
        max_ctx = sr // hop * 30
        if cond.size(1) + mel2.size(2) > max_ctx:
            raise RuntimeError("kaynak + referans bağlam penceresini aşıyor")
        cat = torch.cat([prompt_cond, cond], dim=1)
        with torch.autocast(device_type=svc.device.type, dtype=torch.float16 if svc.fp16 else torch.float32):
            vc_target = st["model"].cfm.inference(cat, torch.LongTensor([cat.size(1)]).to(mel2.device),
                                                  mel2, style2, None, steps, inference_cfg_rate=cfg_rate)
            vc_target = vc_target[:, :, mel2.size(-1):]
        wav = st["vocoder_fn"](vc_target.float()).squeeze().cpu().numpy().astype(np.float32)
    return wav, sr


# ---------------------------------------------------------------- Chatterbox VC
def _cb_setup(job):
    import torch
    from chatterbox.vc import ChatterboxVC

    dev = torch_device(job.get("require_cuda", True))
    model = ChatterboxVC.from_pretrained(dev)
    return {"backend": "chatterbox_vc", "model": model, "torch": torch, "sr": model.sr, "last_target": None}


def _cb_convert(st, src_path, tgt_path):
    torch, model = st["torch"], st["model"]
    with torch.inference_mode():
        if st["last_target"] != tgt_path:
            model.set_target_voice(tgt_path)
            st["last_target"] = tgt_path
        wav = model.generate(audio=src_path)
    return wav.squeeze(0).detach().cpu().numpy().astype(np.float32), model.sr


# ---------------------------------------------------------------- iskelet
def setup(job):
    be = job["options"].get("backend", "seed_vc")
    return _seed_setup(job) if be == "seed_vc" else _cb_setup(job)


def item(st, it, job):
    o = job["options"]
    la = float(it.get("length_adjust", 1.0))
    if st["backend"] == "seed_vc":
        wav, sr = _seed_convert(st, it["source"], it["target"], int(o.get("diffusion_steps", 30)),
                                float(o.get("inference_cfg_rate", 0.7)), la)
    else:
        wav, sr = _cb_convert(st, it["source"], it["target"])
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 0.99:
        wav = wav * (0.99 / peak)
    sf.write(it["out"], wav, sr)
    return {"path": it["out"], "dur": len(wav) / sr, "sr": sr}


if __name__ == "__main__":
    run(setup, item)
