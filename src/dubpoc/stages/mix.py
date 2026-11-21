"""5.12 mix (env: core) — segmentleri zaman çizelgesine yerleştir, arka planla birleştir.

- Her dosya ve yol (A, B) için orijinal uzunlukta boş timeline; segmentlerin best çıktıları start'a konur.
- Yol B'de kaydı olmayan segment → Yol A'nın en iyisi (rapora not düşülür).
- keep_original / skip segmentleri ve çıktısı olmayanlar → orijinal ses segmenti.
- bg varsa toplanır; clip olursa limiter. Orijinal SR ve kanal ile out/<A|B>/<file_id>.wav (PCM 16).
- Arka plan konuşması katmanı (layer "bg"): orijinali zaten bg'de olduğu için dublajı yoksa hiçbir şey
  eklenmez. Dublajlanan bg satırının bölgesi (bg_region) ve dublajlanan diyaloğun arka plana sızan kopyası
  (bg_leak_of) arka plandan spektral kenetlemeyle bastırılır (audio.suppress_speech): anonsun ve
  diyalog sızıntısının İngilizcesi Türkçenin altında duyulmaz (mix.remove_bg_speech).
"""
from __future__ import annotations

import numpy as np

from .. import audio
from ..manifest import is_inactive, set_status
from . import Ctx, console


def best_fit(row: dict, path: str) -> str | None:
    cands = row.get("candidates", {}).get(path) or []
    idx = (row.get("best") or {}).get(path)
    if idx is None or idx >= len(cands):
        return None
    return cands[idx].get("fit")


def run(ctx: Ctx) -> None:
    p = ctx.project
    sel = ctx.manifest.select("mix", ctx.force, ctx.only, include_inactive=True)
    fids = sorted({r["file_id"] for r in sel})
    if not fids:
        return
    lim = float(ctx.cfg["mix"]["limiter_db"])
    for fid in fids:
        rows = ctx.manifest.rows_of_file(fid)
        r0 = rows[0]
        if not r0.get("src_sr"):
            continue
        sr, ch, total = int(r0["src_sr"]), int(r0["src_channels"]), float(r0["src_dur"])
        bg = None
        if r0.get("bg_path"):
            b, bsr = audio.read(p.abs(r0["bg_path"]))
            bg = audio.to_channels(audio.resample(b, bsr, sr), ch)
        k_db = float((ctx.cfg["diarization"].get("bg_speech") or {}).get("suppress_k_db", -6.0))
        for path in ("A", "B"):
            if not ctx.want(path):
                continue
            n = int(round(total * sr))
            tl = np.zeros((n, ch), dtype=np.float32)
            notes = []
            dubbed: set[str] = set()
            for r in rows:
                clip_path, src = None, "dub"
                if not is_inactive(r):
                    clip_path = best_fit(r, path)
                    if clip_path is None and path == "B":
                        clip_path = best_fit(r, "A")
                        if clip_path:
                            src = "A_fallback"
                            notes.append(f"{r['seg_id']}: Yol B kaydı yok → Yol A kullanıldı")
                if clip_path is None and r.get("layer") == "bg":
                    r.setdefault("mix_source", {})[path] = "background"  # orijinali bg'de kalır
                    continue
                if clip_path is None:
                    if not r.get("voice_path"):
                        continue
                    clip_path, src = r["voice_path"], "original"
                    if not is_inactive(r):
                        notes.append(f"{r['seg_id']}: dublaj çıktısı yok → orijinal ses")
                x, xsr = audio.read(p.abs(clip_path))
                x = audio.to_channels(audio.resample(x, xsr, sr), ch)
                a = int(round(float(r["start"]) * sr))
                if a >= n:
                    continue
                m = min(len(x), n - a)
                tl[a: a + m] += x[:m]
                if len(x) - m > int(0.05 * sr) and src != "original":
                    notes.append(f"{r['seg_id']}: {(len(x) - m) / sr:.2f} sn dosya sonunu taştı (kırpıldı)")
                r.setdefault("mix_source", {})[path] = src
                if src != "original":
                    dubbed.add(r["seg_id"])
            if bg is not None:
                m = min(len(bg), n)
                by_id = {r["seg_id"]: r for r in rows}
                regions = [tuple(r["bg_region"]) for r in rows if r.get("layer") == "bg" and r.get("bg_region")
                           and (r["seg_id"] in dubbed or r.get("bg_leak_of") in dubbed)]
                # sızıntının kelime zamanı kayabilir: dublajlı diyaloğun kendi aralığı da bastırılır
                regions += [(float(by_id[r["bg_leak_of"]]["start"]), float(by_id[r["bg_leak_of"]]["end"]))
                            for r in rows if r.get("bg_leak_of") in dubbed]
                if regions and ctx.cfg["mix"].get("remove_bg_speech", True):
                    bgp, _ = audio.suppress_speech(bg, sr, regions, k_db=k_db)
                    notes.append(f"{fid}: arka plandaki {len(regions)} İngilizce konuşma bastırıldı")
                else:
                    bgp = bg
                tl[:m] += bgp[:m]
            peak_db = 20 * np.log10(max(float(np.max(np.abs(tl))) if tl.size else 0.0, 1e-10))
            if peak_db > lim:
                tl = audio.limit_true_peak(tl, sr, lim)
            out = p.odir(path) / f"{fid}.wav"
            audio.write(out, tl, sr, subtype="PCM_16")
            for r in rows:
                r.setdefault("mix_notes", {})[path] = [x for x in notes if x.startswith(r["seg_id"])] or None
            console.print(f"  mix {path}/{fid}.wav" + (f" [dim]({len(notes)} not)[/]" if notes else ""))
        for r in rows:
            if not is_inactive(r):
                set_status(r, "mix", "ok")
            else:
                r["status"]["mix"] = r["status"].get("mix") if r["status"].get("mix") in ("keep_original", "skip") else "ok"
    ctx.save()
