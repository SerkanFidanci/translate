"""`dubpoc doctor` — ortam, binary, GPU, HF token ve API anahtarı kontrolü."""
from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import httpx
from rich.console import Console
from rich.table import Table

from . import audio
from .project import Project

console = Console()

ENV_CHECKS = {
    "core": "import faster_whisper, ctranslate2; print('cuda_devices', ctranslate2.get_cuda_device_count())",
    "dnr": "import pytorch_lightning, librosa, torch; print(torch.__version__, torch.cuda.is_available())",
    "sep": "import audio_separator, torch, onnxruntime as o; print(torch.__version__, torch.cuda.is_available(), 'ort', o.get_device())",
    "diar": "import pyannote.audio, torch; print(pyannote.audio.__version__, torch.__version__, torch.cuda.is_available())",
    "emo": "import funasr, torch; print(funasr.__version__, torch.__version__, torch.cuda.is_available())",
    "tts": "import chatterbox.mtl_tts, torch; print(torch.__version__, torch.cuda.is_available())",
    "vc": "import munch, dac, transformers, torch; print(torch.__version__, torch.cuda.is_available())",
    "omni": "import omnivoice, torch; print(torch.__version__, torch.cuda.is_available())",
    "xtts": "import TTS, torch; print(torch.__version__, torch.cuda.is_available())",
    "voxcpm": "import sys, types; sys.modules['wetext'] = types.ModuleType('wetext'); import voxcpm, torch; "
              "print(torch.__version__, torch.cuda.is_available())",
}


def _check_env(project: Project, name: str, code: str) -> tuple[str, bool, str]:
    py = project.env_python(name)
    if not py.exists():
        return name, False, "kurulu değil (scripts/setup_envs.ps1 -Only " + name + ")"
    env = os.environ.copy()
    env.update({"PYTHONWARNINGS": "ignore", "KMP_DUPLICATE_LIB_OK": "TRUE", "PYTHONIOENCODING": "utf-8"})
    try:
        p = subprocess.run([str(py), "-c", code], capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=300, env=env)
    except subprocess.TimeoutExpired:
        return name, False, "zaman aşımı"
    out = (p.stdout.strip().splitlines() or [""])[-1]
    if p.returncode != 0:
        return name, False, (p.stderr.strip().splitlines() or ["?"])[-1][:160]
    ok = "False" not in out.split() and "cuda_devices 0" not in out
    return name, ok, out[:160]


def run(project: Project) -> int:
    t = Table(title="dubpoc doctor", show_lines=False)
    t.add_column("Kontrol")
    t.add_column("Durum")
    t.add_column("Ayrıntı", overflow="fold")
    problems = 0

    def add(name, ok, detail, warn=False):
        nonlocal problems
        mark = "[green]✓[/]" if ok else ("[yellow]![/]" if warn else "[red]✗[/]")
        if not ok and not warn:
            problems += 1
        t.add_row(name, mark, str(detail))

    add("python (core)", sys.version_info[:2] == (3, 11), sys.version.split()[0], warn=True)

    # GPU
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        name, mem, drv = [x.strip() for x in q.stdout.strip().splitlines()[0].split(",")]
        vram = float(mem) / 1024
        add("GPU", True, f"{name}, {vram:.1f} GB, sürücü {drv}")
        if vram < 7.5:
            add("VRAM", False, f"{vram:.1f} GB < 8 GB: modeller tek tek yüklenir; sorun olursa asr.compute_type=int8_float16, "
                "tts.omnivoice.dtype=float16", warn=True)
    except Exception as e:  # noqa: BLE001
        add("GPU", not project.cfg["gpu"]["require_cuda"], f"nvidia-smi çalışmadı: {e}")

    # binary'ler
    ff = project.tool("ffmpeg")
    if ff:
        try:
            v = subprocess.run([ff, "-version"], capture_output=True, text=True, timeout=20).stdout.split("\n")[0]
        except Exception as e:  # noqa: BLE001
            v = f"çalışmadı: {e}"
        add("ffmpeg", True, f"{ff} · {v[:60]}")
        add("ffmpeg rubberband filtresi", audio.ffmpeg_has_filter(ff, "rubberband"),
            "var" if audio.ffmpeg_has_filter(ff, "rubberband") else "yok (pedalboard kullanılır)", warn=True)
    else:
        add("ffmpeg", False, "bulunamadı (winget install Gyan.FFmpeg)")
    engines = audio.available_stretch_engines(ff, project.tool("rubberband"))
    add("zaman esnetme", bool(engines), ", ".join(engines) or "yok")
    vg = project.tool("vgmstream")
    has_wem = any(project.input_dir.rglob("*.wem")) if project.input_dir.exists() else False
    add("vgmstream-cli", bool(vg) or not has_wem, vg or ("gerekli: input/ içinde .wem var" if has_wem else "yok (.wem girdisi yoksa gerekmez)"),
        warn=not has_wem)
    repo = project.abs(project.cfg["paths"]["seed_vc_repo"])
    add("seed-vc repo", (repo / "inference.py").exists(), str(repo))
    if project.cfg["separation"].get("engine", "dnr") == "dnr":
        msst = project.abs(project.cfg["paths"]["msst_repo"])
        ck = project.abs(project.cfg["separation"]["dnr"]["checkpoint"])
        add("MSST repo (Bandit v2)", (msst / "models" / "bandit_v2" / "bandit.py").exists(), str(msst))
        add("Bandit v2 checkpoint", ck.exists(), f"{ck} ({ck.stat().st_size / 1e6:.0f} MB)" if ck.exists()
            else f"yok: {ck} (scripts/setup_envs -Only dnr)")

    # anahtarlar
    hf_env = project.cfg["diarization"]["hf_token_env"]
    add(f"{hf_env}", bool(os.environ.get(hf_env)),
        "ayarlı" if os.environ.get(hf_env) else "yok → çok konuşmacılı dosyalarda diarization atlanır "
        "(pyannote/speaker-diarization-community-1 koşullarını HF'de kabul edin)", warn=True)
    tcfg = project.cfg["translate"]
    if tcfg["provider"] == "anthropic":
        key = os.environ.get(tcfg.get("api_key_env") or "ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")
        add("Anthropic API", bool(key), f"{tcfg['model']} · " + ("anahtar ayarlı" if key else
            "anahtar yok (ANTHROPIC_API_KEY ya da `ant auth login`; alternatif: provider: openai_compat + Ollama)"))
    else:
        base = (tcfg.get("base_url") or "http://localhost:11434/v1").rstrip("/")
        try:
            r = httpx.get(f"{base}/models", timeout=10)
            names = [m.get("id") for m in r.json().get("data", [])]
            add("LLM (openai_compat)", tcfg["model"] in names, f"{base} · {tcfg['model']} " +
                ("bulundu" if tcfg["model"] in names else f"yok (mevcut: {', '.join(names[:8])})"))
        except Exception as e:  # noqa: BLE001
            add("LLM (openai_compat)", False, f"{base} erişilemedi: {e}")

    # ortamlar (paralel)
    backends = project.cfg["tts"].get("backends") or [project.cfg["tts"].get("backend")]
    sep_cfg = project.cfg["separation"]
    sep_env = "dnr" if sep_cfg.get("engine", "dnr") == "dnr" else "sep"
    needed = ["core", sep_env] + (["sep"] if sep_env == "dnr" and sep_cfg.get("dnr_refine_vocals") else []) + ["diar", "emo", "vc" if project.cfg["vc"]["backend"] == "seed_vc" else "tts"]
    needed += [{"chatterbox_mtl": "tts", "omnivoice": "omni", "xtts": "xtts"}.get(b, b) for b in backends]  # voxcpm → envs/voxcpm
    needed = list(dict.fromkeys(needed))
    console.print(f"[dim]{len(needed)} ortam kontrol ediliyor (torch importları biraz sürebilir)...[/]")
    with ThreadPoolExecutor(max_workers=4) as ex:
        for name, ok, detail in ex.map(lambda n: _check_env(project, n, ENV_CHECKS[n]), needed):
            add(f"env: {name}", ok, detail)

    console.print(t)
    console.print("[green]Her şey hazır.[/]" if problems == 0 else f"[red]{problems} sorun var.[/]")
    return 1 if problems else 0
