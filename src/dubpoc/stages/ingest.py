"""5.1 ingest — input/ taranır, her dosya work/ingest/<file_id>.wav olarak (orijinal SR'de) kopyalanır."""
from __future__ import annotations


from .. import audio
from ..manifest import add_error, clear_errors, new_status, set_status
from . import Ctx, console

AUDIO_EXT = {".wav", ".ogg", ".mp3", ".wem", ".flac", ".m4a", ".opus", ".aac", ".wma", ".aiff", ".aif"}


def file_row(file_id: str) -> dict:
    return {
        "seg_id": f"{file_id}_s01", "file_id": file_id, "src_path": None,
        "src_sr": None, "src_channels": None, "src_lufs": None, "src_dur": None,
        "start": 0.0, "end": None, "dur_src": None,
        "ingest_path": None, "file_voice_path": None, "voice_path": None, "bg_path": None,
        "char_hint": None, "cluster": None, "char": None,
        "status": new_status(), "errors": [],
    }


def run(ctx: Ctx) -> None:
    p = ctx.project
    files = sorted(f for f in p.input_dir.rglob("*") if f.is_file() and f.suffix.lower() in AUDIO_EXT)
    if not files:
        console.print("[yellow]input/ altında ses dosyası yok.[/]")
        return
    by_src = {}
    for r in ctx.manifest.rows:
        by_src.setdefault(r["src_path"], r["file_id"])
    used = [int(fid[1:]) for fid in by_src.values() if fid and fid[1:].isdigit()]
    next_id = max(used, default=0) + 1
    ffmpeg, vgm = p.tool("ffmpeg"), p.tool("vgmstream")
    n_new = 0
    for f in files:
        rel = p.rel(f)
        fid = by_src.get(rel)
        if fid is None:
            fid = f"f{next_id:03d}"
            next_id += 1
        existing = ctx.manifest.rows_of_file(fid)
        if ctx.only and not set(ctx.only.split(",")) & {fid, *(r["seg_id"] for r in existing)}:
            continue
        if existing and not ctx.force and all(r["status"]["ingest"] == "ok" for r in existing):
            continue
        row = file_row(fid)
        row["src_path"] = rel
        try:
            dst = p.wdir("ingest") / f"{fid}.wav"
            p.assert_not_input(dst)
            audio.convert_to_wav(f, dst, ffmpeg, vgm)
            data, sr = audio.read(dst)
            lufs_val = audio.lufs(data, sr)
            dur = data.shape[0] / sr
            row.update({
                "src_sr": sr, "src_channels": int(data.shape[1]),
                "src_lufs": round(lufs_val, 2) if lufs_val is not None else None,
                "src_dur": round(dur, 4), "end": round(dur, 4), "dur_src": round(dur, 4),
                "ingest_path": p.rel(dst), "file_voice_path": p.rel(dst),
            })
            # input/<karakter>/dosya.wav → karakter ipucu
            parent = f.parent.resolve()
            if parent != p.input_dir.resolve():
                row["char_hint"] = f.relative_to(p.input_dir).parts[0]
            clear_errors(row, "ingest")
            set_status(row, "ingest", "ok")
            n_new += 1
        except Exception as e:  # noqa: BLE001
            add_error(row, "ingest", e)
            console.print(f"[red]ingest hata[/] {rel}: {e}")
        ctx.manifest.replace_file_rows(fid, [row])
    ctx.save()
    console.print(f"ingest: {n_new} dosya işlendi, toplam {len(ctx.manifest.file_ids())} dosya")
