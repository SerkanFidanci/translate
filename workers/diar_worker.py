"""pyannote worker'ı (env: diar).

action=diarize: item {id, path} → [{start, end, speaker}] (exclusive_speaker_diarization)
action=embed  : item {id, path, out} → embedding out'a .npy kaydedilir; çıktı {dim}
Ses dosyaları soundfile ile okunup {"waveform", "sample_rate"} olarak verilir: pyannote 4'ün
torchcodec/FFmpeg bağımlılığı (Windows'ta sorunlu) devreye girmez.
"""
import os

import numpy as np

from _common import log, read_mono, run, torch_device


def _token(job):
    env = job.get("options", {}).get("token_env") or "HF_TOKEN"
    return os.environ.get(env) or None


def setup(job):
    import torch

    dev = torch_device(job.get("require_cuda", True))
    o = job.get("options", {})
    action = job.get("action")
    if action == "diarize":
        from pyannote.audio import Pipeline

        tok = _token(job)
        if not tok:
            raise RuntimeError("HF token yok; diarization için gerekli")
        pipe = Pipeline.from_pretrained(o.get("model", "pyannote/speaker-diarization-community-1"), token=tok)
        if pipe is None:
            raise RuntimeError("Pipeline yüklenemedi (model koşulları HF'de kabul edildi mi?)")
        pipe.to(dev)
        return {"pipe": pipe, "torch": torch}
    if action == "embed":
        from pyannote.audio import Inference, Model

        model = Model.from_pretrained(o.get("model", "pyannote/wespeaker-voxceleb-resnet34-LM"), token=_token(job))
        if model is None:
            raise RuntimeError("Embedding modeli yüklenemedi")
        inf = Inference(model, window="whole", device=dev)
        return {"inf": inf, "torch": torch}
    raise RuntimeError(f"bilinmeyen action: {action}")


def item(state, it, job):
    torch = state["torch"]
    x, sr = read_mono(it["path"], 16000)
    if job["action"] == "diarize":
        wav = torch.from_numpy(x)[None, :]
        out = state["pipe"]({"waveform": wav, "sample_rate": sr})
        ann = getattr(out, "exclusive_speaker_diarization", None) or getattr(out, "speaker_diarization", out)
        turns = [{"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": str(spk)}
                 for seg, _, spk in ann.itertracks(yield_label=True)]
        log(f"{it['id']}: {len(turns)} tur, {len({t['speaker'] for t in turns})} konuşmacı")
        return turns
    # embed: çok kısa klipleri sıfırla doldur (model en az ~0.5 sn ister)
    min_len = int(0.6 * sr)
    if len(x) < min_len:
        x = np.pad(x, (0, min_len - len(x)))
    emb = state["inf"]({"waveform": torch.from_numpy(x)[None, :], "sample_rate": sr})
    emb = np.asarray(emb, dtype=np.float32).reshape(-1)
    np.save(it["out"], emb)
    return {"dim": int(emb.shape[0])}


if __name__ == "__main__":
    run(setup, item)
