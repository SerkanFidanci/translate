"""Worker ortak yardımcıları. Her worker kendi venv'inde çalışır; dubpoc paketini import ETMEZ.

Sözleşme: python <worker>.py <in.json> <out.json>
  in.json : {"items": [{"id": ..., ...}], "options": {...}, "project_root": ..., "require_cuda": bool}
  out.json: {"results": {id: {...}}, "errors": {id: "mesaj"}, ...}  (ölümcül hata: {"fatal": "..."})
stdout sadece log içindir.
"""
from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback
from pathlib import Path


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_job() -> tuple[dict, Path]:
    if len(sys.argv) != 3:
        print("kullanım: worker.py in.json out.json", file=sys.stderr)
        sys.exit(2)
    job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    return job, Path(sys.argv[2])


def save(out_path: Path, data: dict) -> None:
    """Atomik yazma. Windows'ta antivirüs/indeksleyici dosyayı kısa süre kilitleyebilir (WinError 5):
    birkaç kez tekrar dene; ara kayıtlarda başarısızlık işi durdurmaz."""
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for i in range(20):
        try:
            os.replace(tmp, out_path)
            return
        except PermissionError:
            time.sleep(0.1 * (i + 1))
    os.replace(tmp, out_path)


def add_nvidia_dll_dirs() -> None:
    """Windows'ta pip ile gelen cuBLAS/cuDNN DLL'lerini (nvidia-*-cu12) yükleyiciye göster."""
    if sys.platform != "win32":
        return
    for sp in sys.path:
        for d in glob.glob(os.path.join(sp, "nvidia", "*", "bin")):
            try:
                os.add_dll_directory(d)
            except (OSError, AttributeError):
                pass
            os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def torch_device(require_cuda: bool):
    import torch

    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)} · {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GB")
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA bulunamadı (torch.cuda.is_available() == False). "
                           "GPU sürücüsünü / torch CUDA kurulumunu kontrol edin ya da config gpu.require_cuda: false.")
    log("UYARI: CUDA yok, CPU kullanılıyor (yavaş)")
    return torch.device("cpu")


def read_mono(path: str, sr: int | None = None):
    """soundfile ile oku, mono'ya indir, istenirse yeniden örnekle. (np.float32, sr)"""
    import numpy as np
    import soundfile as sf

    x, s = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr and s != sr:
        try:
            import soxr
            x = soxr.resample(x, s, sr).astype(np.float32)
        except ImportError:
            import librosa
            x = librosa.resample(x, orig_sr=s, target_sr=sr).astype(np.float32)
        s = sr
    return x, s


def run(fn_setup, fn_item, extra: dict | None = None) -> None:
    """Ortak iskelet: modeli bir kez kur, öğeleri tek tek işle, hataları öğe bazında topla."""
    job, out_path = load_job()
    results, errors = {}, {}
    out = {"results": results, "errors": errors}
    try:
        state = fn_setup(job)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        save(out_path, {"fatal": f"{type(e).__name__}: {e}", "results": {}, "errors": {}})
        return
    items = job.get("items", [])
    t0 = last_save = time.time()
    for i, it in enumerate(items, 1):
        try:
            results[it["id"]] = fn_item(state, it, job)
            log(f"{i}/{len(items)} {it['id']} ok")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            errors[it["id"]] = f"{type(e).__name__}: {e}"
            log(f"{i}/{len(items)} {it['id']} HATA: {e}")
        if time.time() - last_save > 10:  # ara kayıt: çökmeye karşı, ama her öğede değil
            try:
                save(out_path, {**out, **(extra or {}), **(state.get('_extra', {}) if isinstance(state, dict) else {})})
                last_save = time.time()
            except OSError as e:
                log(f"ara kayıt atlandı: {e}")
    log(f"bitti: {len(results)} ok, {len(errors)} hata, {time.time() - t0:.1f} sn")
    save(out_path, {**out, **(extra or {}), **(state.get('_extra', {}) if isinstance(state, dict) else {})})
