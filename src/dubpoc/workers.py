"""Ağır modelleri kendi venv'lerinde subprocess olarak çalıştırma.

Sözleşme (spec 1.1): girdi JSON dosya yolu → çıktı JSON dosya yolu; stdout sadece log.
Her worker tek model yükler, işi bitirir ve kapanır (GPU belleği tutulmaz).
Aynı anda iki GPU worker çalışmasın diye work/.gpu.lock üzerinde OS seviyesinde kilit alınır.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager

from rich.console import Console

from .project import Project

console = Console()


class WorkerError(RuntimeError):
    pass


@contextmanager
def gpu_lock(project: Project):
    path = project.wdir() / ".gpu.lock"
    f = open(path, "a+")
    try:
        if sys.platform == "win32":
            import msvcrt
            while True:
                try:
                    f.seek(0)
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    console.print("[yellow]GPU kilidi bekleniyor (başka bir dubpoc süreci çalışıyor)...[/]")
                    time.sleep(3)
        else:
            import fcntl
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if sys.platform == "win32":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


def run_worker(project: Project, env: str, script: str, payload: dict, label: str,
               gpu: bool = True) -> dict:
    """workers/<script> dosyasını envs/<env> python'u ile çalıştırır ve çıktı JSON'unu döndürür."""
    py = project.env_python(env)
    if not py.exists():
        raise WorkerError(f"'{env}' ortamı yok ({py}). Önce: scripts/setup_envs.ps1 -Only {env}")
    jobs = project.wdir("jobs")
    logs = project.wdir("logs")
    stamp = time.strftime("%Y%m%d-%H%M%S")
    in_path = jobs / f"{label}_{stamp}_in.json"
    out_path = jobs / f"{label}_{stamp}_out.json"
    log_path = logs / f"{label}.log"
    payload = dict(payload)
    payload.setdefault("project_root", str(project.root))
    payload.setdefault("require_cuda", bool(project.cfg["gpu"]["require_cuda"]))
    in_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    env_vars = os.environ.copy()
    env_vars.update({
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUTF8": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "TOKENIZERS_PARALLELISM": "false",
        "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
        "PYTHONWARNINGS": "ignore",
    })
    extra_path = project.tool_dirs_for_path()
    if extra_path:
        env_vars["PATH"] = os.pathsep.join(extra_path + [env_vars.get("PATH", "")])

    cmd = [str(py), str(project.root / "workers" / script), str(in_path), str(out_path)]
    n_items = len(payload.get("items", []))
    console.print(f"[cyan]▶ worker[/] {label} [dim]({env} env, {n_items} öğe)[/]")
    t0 = time.time()
    lock = gpu_lock(project) if gpu else _null()
    with lock, open(log_path, "a", encoding="utf-8") as log:
        log.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} {label} =====\n")
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                encoding="utf-8", errors="replace", env=env_vars, cwd=str(project.root))
        tail: list[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            line = line.rstrip()
            tail = (tail + [line])[-40:]
            if line.startswith("[") or "error" in line.lower():
                console.print(f"  [dim]{line[:200]}[/]")
        rc = proc.wait()
    dt = time.time() - t0
    if not out_path.exists():
        raise WorkerError(f"{label} worker çıktı üretmedi (rc={rc}, {dt:.0f} sn). Log: {log_path}\n" + "\n".join(tail[-15:]))
    result = json.loads(out_path.read_text(encoding="utf-8"))
    if result.get("fatal"):
        raise WorkerError(f"{label}: {result['fatal']}\nLog: {log_path}")
    # Worker yarıda kaldıysa (rc≠0) eksik öğeleri sessizce "başarılı" sayma
    done = set(result.get("results", {})) | set(result.get("errors", {}))
    missing = [it["id"] for it in payload.get("items", []) if it["id"] not in done]
    if missing:
        result.setdefault("errors", {}).update({m: f"worker yarıda kaldı (rc={rc}); log: {log_path}" for m in missing})
        console.print(f"[red]{label}: {len(missing)} öğe işlenmeden kaldı (rc={rc})[/]")
    console.print(f"[green]✓[/] {label} [dim]{dt:.0f} sn, hata: {len(result.get('errors', {}))}[/]")
    return result


@contextmanager
def _null():
    yield
