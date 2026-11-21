"""Duygu worker'ı (env: emo) — funasr emotion2vec_plus_large + RMS + perde std.

item : {id, path, emb_out}
çıktı: {emotion, score, probs{label: p}, rms_db, pitch_std (yarım ton), emb}
"""
import numpy as np

from _common import log, read_mono, run, torch_device


def _label(raw: str) -> str:
    # "生气/angry" → "angry"; "<unk>" → "unknown"
    lab = raw.split("/")[-1].strip().lower()
    return "unknown" if lab in ("<unk>", "unk", "") else lab


def setup(job):
    import torch
    from funasr import AutoModel

    dev = torch_device(job.get("require_cuda", True))
    name = job.get("options", {}).get("model", "emotion2vec/emotion2vec_plus_large")
    hub = "hf" if "/" in name and not name.startswith("iic/") else "ms"
    model = AutoModel(model=name, hub=hub, device=str(dev), disable_update=True)
    log(f"emotion model: {name} ({hub})")
    return {"model": model, "torch": torch}


def item(state, it, job):
    import librosa

    x, sr = read_mono(it["path"], 16000)
    rec = state["model"].generate(x, granularity="utterance", extract_embedding=True, disable_pbar=True)[0]
    labels = [_label(l) for l in rec["labels"]]
    scores = [float(s) for s in rec["scores"]]
    probs: dict[str, float] = {}
    for l, s in zip(labels, scores):
        probs[l] = probs.get(l, 0.0) + s
    top = max(probs, key=probs.get)
    feats = rec.get("feats")
    emb_path = None
    if feats is not None and it.get("emb_out"):
        np.save(it["emb_out"], np.asarray(feats, dtype=np.float32).reshape(-1))
        emb_path = it["emb_out"]
    # RMS (konuşma kısmı) ve perde değişkenliği
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-10)
    pitch_std = None
    try:
        f0, voiced, _ = librosa.pyin(x, fmin=65, fmax=700, sr=sr, frame_length=1024)
        f0 = f0[voiced & ~np.isnan(f0)] if f0 is not None else np.array([])
        if f0.size >= 5:
            pitch_std = float(np.std(12 * np.log2(f0 / np.median(f0))))
    except Exception as e:  # noqa: BLE001
        log(f"pyin hata: {e}")
    return {"emotion": top, "score": probs[top], "probs": probs, "rms_db": 20 * np.log10(rms),
            "pitch_std": pitch_std, "emb": emb_path}


if __name__ == "__main__":
    run(setup, item)
