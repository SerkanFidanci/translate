"""Proje kökü, config ve yol yardımcıları."""
from __future__ import annotations

import copy
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# config.yaml'da eksik anahtar varsa kullanılan varsayılanlar (spec bölüm 3 + eklemeler)
DEFAULTS: dict[str, Any] = {
    "paths": {
        "ffmpeg": "ffmpeg",
        "vgmstream": "tools/vgmstream-cli.exe",
        "rubberband": "tools/rubberband.exe",
        "envs": "envs",
        "seed_vc_repo": "tools/seed-vc",
        "msst_repo": "tools/msst",
    },
    "source_lang": "en",
    "target_lang": "tr",
    "gpu": {"require_cuda": True},
    "separation": {"mode": "auto", "engine": "dnr", "model": "", "clean_bg_threshold_dbfs": -45,
                   "voice_missing_db": 15, "dnr_refine_vocals": False,
                   "dnr": {"checkpoint": "tools/models/bandit_v2/checkpoint-multi.ckpt",
                           "config": "tools/msst/configs/config_dnr_bandit_v2_mus64.yaml",
                           "batch_size": 2, "num_overlap": 4, "fp16": True}},
    "diarization": {"mode": "auto", "min_duration_s": 8, "hf_token_env": "HF_TOKEN", "min_segment_s": 0.4,
                    "bg_speech": {"enabled": True, "min_logprob": -0.8, "max_no_speech": 0.6, "min_word_p": 0.45,
                                  "max_compression": 2.4, "min_rms_dbfs": -50.0, "max_dialog_overlap": 0.5,
                                  "scan_win_s": 10, "scan_hop_s": 5, "suppress_k_db": -6.0}},
    "speakers": {
        "cluster_threshold": 0.55,
        "num_speakers": None,
        "bank_max_seconds": 20,
        "bank_short_seconds": 12,
        "bank_min_clip_s": 1.5,
        "bank_target_min_s": 6.0,
        "folder_name_as_character": True,
    },
    "asr": {"model": "large-v3", "compute_type": "float16", "beam_size": 5, "file_context": True},
    "emotion": {"model": "emotion2vec/emotion2vec_plus_large",
                "intensity_source": "one_minus_neutral"},
    "translate": {
        "provider": "anthropic",
        "model": "claude-sonnet-5-5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "base_url": "",
        "effort": "medium",
        "fallbacks": True,
        "tr_syllables_per_sec": 6.0,
        "max_shorten_rounds": 2,
        "shorten_trigger_syll_ratio": 1.15,
    },
    "tts": {
        "backends": ["chatterbox_mtl"],
        "chatterbox_t3_model": "v3",
        "language_id": "tr",
        "reference_mode": "bank",
        "n_candidates": 3,
        "seeds": [11, 22, 33],
        "exaggeration_range": [0.35, 0.9],
        "cfg_weight_bank": 0.5,
        "cfg_weight_original_line": 0.0,
        "high_exaggeration_cfg": 0.3,
        "emotion_ref_candidates": 2,
        "cfg_weight_emotion_ref": 0.3,
        "temperature": 0.8,
        "max_chars_per_chunk": 250,
        "omnivoice": {"n_candidates": 2, "num_step": 32, "guidance_scale": 2.0, "use_duration": True,
                      "duration_modes": ["target", "natural"], "min_target_s": 1.0, "dtype": "float16"},
    },
    "vc": {
        "backend": "seed_vc",
        "diffusion_steps": 30,
        "inference_cfg_rate": 0.7,
        "fp16": True,
        "length_adjust": True,
        "max_length_adjust": 0.2,
        "also_score_without_vc": True,
    },
    "fit": {
        "max_stretch": 0.15,
        "shorten_trigger_ratio": 1.25,
        "pad_ms": 40,
        "silence_db": -45,
        "stretch_engine": "auto",
        "true_peak_db": -1.0,
    },
    "qa": {
        "wer_max": 0.15,
        "cer_max": 0.08,
        "text_metric": "wer",
        "fail_wer": 0.4,
        "fail_cer": 0.25,
        "spk_sim_min_relative": 0.85,
        "dur_tolerance": 0.10,
        "weights": {"wer": 0.45, "spk_sim": 0.25, "dur": 0.15, "emo": 0.15},
        "asr_model": "large-v3",
        "emotion_metrics": True,
    },
    "mix": {"limiter_db": -1.0},
    "fx": {"enabled": True, "auto": True, "phone_lo_db": -6.0, "phone_hi_db": -26.0, "hint_hi_db": -24.0,
           "overrides": {}},
}

STAGES = [
    "ingest", "separate", "diarize", "speakers", "asr", "emotion",
    "translate", "tts", "vc", "fit", "qa", "mix",
]
ALL_STEPS = STAGES + ["report"]


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def find_root(start: Path | None = None) -> Path:
    env = os.environ.get("DUBPOC_ROOT")
    if env:
        return Path(env).resolve()
    cur = (start or Path.cwd()).resolve()
    for p in [cur, *cur.parents]:
        if (p / "config.yaml").exists() and (p / "workers").is_dir():
            return p
    # paket kaynağından (editable install) geri dön
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "config.yaml").exists() and (p / "workers").is_dir():
            return p
    raise SystemExit("Proje kökü bulunamadı (config.yaml + workers/). --root ile belirtin.")


class Project:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.config_path = self.root / "config.yaml"
        raw = {}
        if self.config_path.exists():
            raw = yaml.safe_load(self.config_path.read_text(encoding="utf-8")) or {}
        self.cfg: dict[str, Any] = deep_merge(DEFAULTS, raw)
        # config.local.yaml: makineye özel geçersiz kılmalar (örn. API anahtarı yokken yerel Ollama)
        local = self.root / "config.local.yaml"
        if local.exists():
            self.cfg = deep_merge(self.cfg, yaml.safe_load(local.read_text(encoding="utf-8")) or {})

    # ---- klasörler -------------------------------------------------------
    @property
    def input_dir(self) -> Path: return self.root / "input"
    @property
    def record_dir(self) -> Path: return self.root / "record"
    @property
    def work(self) -> Path: return self.root / "work"
    @property
    def out(self) -> Path: return self.root / "out"
    @property
    def manifest_path(self) -> Path: return self.work / "manifest.jsonl"
    @property
    def speakers_yaml(self) -> Path: return self.root / "speakers.yaml"
    @property
    def subtitles_csv(self) -> Path: return self.root / "subtitles.csv"

    def wdir(self, *parts: str) -> Path:
        p = self.work.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def odir(self, *parts: str) -> Path:
        p = self.out.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ---- yol dönüşümleri ---------------------------------------------------
    def rel(self, p: Path | str | None) -> str | None:
        if p is None:
            return None
        p = Path(p)
        try:
            return p.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return p.resolve().as_posix()

    def abs(self, rel: str | Path | None) -> Path | None:
        if rel is None or rel == "":
            return None
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    def assert_not_input(self, p: Path) -> None:
        """input/ altındaki dosyalar hiçbir koşulda değiştirilmez (kabul kriteri 5)."""
        try:
            Path(p).resolve().relative_to(self.input_dir.resolve())
        except ValueError:
            return
        raise RuntimeError(f"input/ altına yazma girişimi engellendi: {p}")

    # ---- harici araçlar ----------------------------------------------------
    def env_python(self, name: str) -> Path:
        base = self.abs(self.cfg["paths"]["envs"]) / name
        if sys.platform == "win32":
            return base / "Scripts" / "python.exe"
        return base / "bin" / "python"

    def tool(self, key: str) -> str | None:
        """config'teki yolu çöz: önce proje göreli/mutlak dosya, sonra PATH."""
        val = str(self.cfg["paths"].get(key) or "")
        if not val:
            return None
        p = self.abs(val)
        if p is not None and p.exists():
            return str(p)
        found = shutil.which(val) or shutil.which(Path(val).name)
        if found:
            return found
        if key == "ffmpeg":
            return self._bundled_ffmpeg()
        return None

    def _bundled_ffmpeg(self) -> str | None:
        """Sistemde ffmpeg yoksa imageio-ffmpeg'in paketlediği exe'yi tools/bin/ffmpeg(.exe) olarak kopyala.

        audio-separator gibi araçlar 'ffmpeg' adını PATH'te aradığı için kopya, worker'ların PATH'ine eklenir.
        """
        exe_name = "ffmpeg.exe" if sys.platform == "win32" else "ffmpeg"
        target = self.root / "tools" / "bin" / exe_name
        if target.exists():
            return str(target)
        try:
            import imageio_ffmpeg  # type: ignore
        except ImportError:
            return None
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        return str(target)

    def tool_dirs_for_path(self) -> list[str]:
        dirs = []
        for key in ("ffmpeg", "vgmstream", "rubberband"):
            t = self.tool(key)
            if t:
                dirs.append(str(Path(t).parent))
        return list(dict.fromkeys(dirs))

    # ---- config yazma (yorumları koruyarak) ----------------------------
    def set_config_value(self, section: str, key: str, value: str) -> None:
        text = self.config_path.read_text(encoding="utf-8")
        pattern = re.compile(
            rf"(^{re.escape(section)}:\s*\n(?:^[ \t]+.*\n|^\s*\n)*?^[ \t]+{re.escape(key)}:)[ \t]*(\"[^\"]*\"|'[^']*'|[^#\n]*)",
            re.M,
        )
        new_text, n = pattern.subn(lambda m: f'{m.group(1)} "{value}"', text, count=1)
        if n == 0:
            raise RuntimeError(f"config.yaml içinde {section}.{key} bulunamadı")
        tmp = self.config_path.with_suffix(".yaml.tmp")
        tmp.write_text(new_text, encoding="utf-8")
        os.replace(tmp, self.config_path)
        self.cfg[section][key] = value
