"""Ayrıştırma worker'ı (env: sep) — python-audio-separator.

options: model ("" → en yüksek vocals SDR'li RoFormer seçilir), output_dir
item   : {id, path, voice_out, bg_out}
çıktı  : {voice, bg}; ek alan "model"
"""
import os
import shutil
from pathlib import Path

from _common import log, run


def pick_model(sep) -> str:
    models = sep.get_simplified_model_list(filter_sort_by="vocals")
    best, best_sdr = None, -1.0
    for fname, info in models.items():
        name = (info.get("Name") or "") + " " + fname
        sdr = (info.get("SDR") or {}).get("vocals")
        if sdr is None or "roformer" not in name.lower():
            continue
        if sdr > best_sdr:
            best, best_sdr = fname, sdr
    if not best:
        best = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    log(f"seçilen model: {best} (vocals SDR {best_sdr})")
    return best


def setup(job):
    import torch  # noqa: F401
    from audio_separator.separator import Separator

    o = job.get("options", {})
    out_dir = o.get("output_dir") or "."
    model_dir = str(Path(job["project_root"]) / "tools" / "models" / "audio-separator")
    os.makedirs(model_dir, exist_ok=True)
    sep = Separator(output_dir=out_dir, model_file_dir=model_dir, output_format="WAV", use_autocast=True,
                    mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": 1,
                                 "overlap": 8, "pitch_shift": 0})
    model = o.get("model") or pick_model(sep)
    sep.load_model(model_filename=model)
    return {"sep": sep, "_extra": {"model": model}, "out_dir": out_dir}


def item(state, it, job):
    sep = state["sep"]
    stem_base = Path(it["voice_out"]).stem.rsplit("_voice", 1)[0]
    files = sep.separate(it["path"], custom_output_names={"Vocals": f"{stem_base}_voice", "Instrumental": f"{stem_base}_bg"})
    files = [f if os.path.isabs(f) else os.path.join(state["out_dir"], f) for f in files]
    voice = next((f for f in files if f.endswith("_voice.wav")), None)
    bg = next((f for f in files if f.endswith("_bg.wav")), None)
    if voice is None or bg is None:
        # bazı modeller farklı stem adları döndürür: vocals'u ses, kalanların toplamını bg say
        import soundfile as sf
        voice = next(f for f in files if "vocal" in f.lower())
        others = [f for f in files if f != voice]
        acc = None
        for f in others:
            x, sr = sf.read(f, dtype="float32", always_2d=True)
            acc = x if acc is None else acc[: len(x)] + x[: len(acc)]
        sf.write(it["bg_out"], acc, sr)
        bg = it["bg_out"]
    if os.path.abspath(voice) != os.path.abspath(it["voice_out"]):
        shutil.move(voice, it["voice_out"])
    if os.path.abspath(bg) != os.path.abspath(it["bg_out"]):
        shutil.move(bg, it["bg_out"])
    return {"voice": it["voice_out"], "bg": it["bg_out"]}


if __name__ == "__main__":
    run(setup, item)
