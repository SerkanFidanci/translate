"""DnR ayrıştırma worker'ı (env: dnr) — Bandit v2 (diyalog / müzik / efekt), 48 kHz.

Model kodu: tools/msst (ZFTurbo Music-Source-Separation-Training) models/bandit_v2.
Ağırlık: Zenodo 12701995 checkpoint-multi.ckpt (çok dilli DnR v3, CC BY-SA 4.0). Lightning checkpoint'inde
anahtarlar "model." önekli ve "loss_handler*" fazlalıkları var; ilk yüklemede temizlenip
<ckpt>.dubpoc.pt olarak önbelleğe alınır (optimizer durumu olmadan ~3 kat küçük).

options: msst_repo, checkpoint, config, batch_size (2), num_overlap (4), fp16 (true)
item   : {id, path, voice_out, bg_out, music_out, sfx_out}
çıktı  : {voice, bg, music, sfx, sr}
"""
import os
import sys

import numpy as np
import soundfile as sf
import yaml

from _common import log, run, torch_device

SR = 48000


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader + sadece MSST config'lerindeki !!python/tuple etiketi (keyfi nesne kurulmaz)."""


_ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                              lambda loader, node: tuple(loader.construct_sequence(node)))


def _load_weights(torch, ckpt_path: str):
    cache = ckpt_path + ".dubpoc.pt"
    if os.path.exists(cache) and os.path.getmtime(cache) >= os.path.getmtime(ckpt_path):
        return torch.load(cache, map_location="cpu", weights_only=True)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # sadece tensör/temel tipler
    sd = ck.get("state_dict", ck)
    clean = {k[len("model."):] if k.startswith("model.") else k: v for k, v in sd.items()
             if not k.startswith("loss_handler")}
    torch.save(clean, cache)
    log(f"ağırlıklar dönüştürüldü → {os.path.basename(cache)} ({len(clean)} anahtar)")
    return clean


def setup(job):
    import torch

    o = job["options"]
    repo = o["msst_repo"]
    if not os.path.exists(os.path.join(repo, "models", "bandit_v2", "bandit.py")):
        raise RuntimeError(f"MSST reposu yok: {repo} (scripts/setup_envs -Only dnr)")
    if not os.path.exists(o["checkpoint"]):
        raise RuntimeError(f"Bandit v2 checkpoint yok: {o['checkpoint']}")
    sys.path.insert(0, repo)
    from models.bandit_v2.bandit import Bandit  # noqa: E402

    dev = torch_device(job.get("require_cuda", True))
    with open(o["config"], encoding="utf-8") as f:
        cfg = yaml.load(f, Loader=_ConfigLoader)  # noqa: S506 (SafeLoader tabanlı)
    model = Bandit(**cfg["kwargs"])
    missing, unexpected = model.load_state_dict(_load_weights(torch, o["checkpoint"]), strict=False)
    if missing or unexpected:
        raise RuntimeError(f"ağırlık uyuşmazlığı: eksik={missing[:3]} fazla={unexpected[:3]}")
    model.eval().to(dev)
    stems = list(cfg["kwargs"]["stems"])
    log(f"Bandit v2 hazır ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M param, stems={stems})")
    return {"model": model, "torch": torch, "dev": dev, "stems": stems,
            "chunk": int(cfg["audio"]["chunk_size"]), "o": o,
            "_extra": {"model": "bandit_v2:" + os.path.basename(o["checkpoint"])}}


def _window(n: int, fade: int):
    w = np.ones(n, dtype=np.float32)
    w[:fade] = np.linspace(0, 1, fade, dtype=np.float32)
    w[-fade:] = np.linspace(1, 0, fade, dtype=np.float32)
    return w


def demix(st, mix: np.ndarray) -> np.ndarray:
    """Örtüşen pencerelerle (lineer geçiş) parça parça ayrıştırma. mix: (C, T) → (n_stems, C, T)."""
    torch, model, dev = st["torch"], st["model"], st["dev"]
    o = st["o"]
    chunk = st["chunk"]
    step = chunk // int(o.get("num_overlap", 4))
    fade = chunk // 10
    border = chunk - step
    T0 = mix.shape[-1]
    x = mix
    padded = T0 > 2 * border
    if padded:
        x = np.pad(mix, ((0, 0), (border, border)), mode="reflect")
    T = x.shape[-1]
    win = _window(chunk, fade)
    out = np.zeros((len(st["stems"]),) + x.shape, dtype=np.float32)
    cnt = np.zeros(T, dtype=np.float32)
    starts = list(range(0, T, step))
    bs = int(o.get("batch_size", 2))
    use_amp = bool(o.get("fp16", True)) and dev.type == "cuda"
    with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_amp):
        for b in range(0, len(starts), bs):
            group = starts[b: b + bs]
            parts = []
            for s in group:
                p = x[:, s: s + chunk]
                n = p.shape[-1]
                mode = "reflect" if n > chunk // 2 else "constant"
                parts.append(np.pad(p, ((0, 0), (0, chunk - n)), mode=mode))
            y = model(torch.from_numpy(np.stack(parts)).to(dev)).float().cpu().numpy()  # (B, S, C, chunk)
            for j, s in enumerate(group):
                n = min(chunk, T - s)
                w = win.copy()
                if s == 0:
                    w[:fade] = 1
                if s + step >= T:
                    w[-fade:] = 1
                out[:, :, s: s + n] += y[j, :, :, :n] * w[:n]
                cnt[s: s + n] += w[:n]
    out /= np.maximum(cnt, 1e-8)
    if padded:
        out = out[..., border: border + T0]
    return out


def item(st, it, job):
    x, sr = sf.read(it["path"], dtype="float32", always_2d=True)  # (T, C)
    if x.shape[1] > 2:
        x = np.stack([x.mean(axis=1)] * 2, axis=1)
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR).astype(np.float32)
    est = demix(st, x.T.copy())  # (S, C, T)
    stems = dict(zip(st["stems"], est))
    music, sfx, speech = stems["music"], stems["sfx"], stems["speech"]
    sf.write(it["voice_out"], speech.T, SR)
    sf.write(it["bg_out"], (music + sfx).T, SR)
    if it.get("music_out"):
        sf.write(it["music_out"], music.T, SR)
    if it.get("sfx_out"):
        sf.write(it["sfx_out"], sfx.T, SR)
    if st["dev"].type == "cuda":
        st["torch"].cuda.empty_cache()
    return {"voice": it["voice_out"], "bg": it["bg_out"], "music": it.get("music_out"),
            "sfx": it.get("sfx_out"), "sr": SR}


if __name__ == "__main__":
    run(setup, item)
