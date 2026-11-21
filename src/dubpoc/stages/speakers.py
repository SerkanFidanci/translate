"""5.4 speakers (env: diar) — karakter eşleme, ses bankası ve baseline benzerlik.

1. Her segment için wespeaker embedding'i (work/emb/<seg_id>.npy).
2. input/<karakter>/ klasör modu yoksa: dosyalar arası global agglomerative clustering (cosine,
   average linkage). Kısa klipler (<1 sn) merkez tahminine katılmaz, sonra en yakın kümeye atanır.
3. speakers.yaml üretilir/güncellenir; kullanıcının yazdığı isimler korunur.
4. Ses bankası: bank/<char>/ref.wav (+ clips.json, ref_emb.npy, baseline_sim.txt, baseline.json).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from .. import audio
from ..manifest import add_error, clear_errors, set_status
from ..workers import WorkerError, run_worker
from . import Ctx, console

MIN_CLUSTER_CLIP_S = 1.0


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def agglomerative(embs: np.ndarray, threshold: float, n_clusters: int | None = None) -> list[int]:
    """Average-linkage, cosine distance. Küçük n için yeterli (O(n^3)).
    n_clusters verilirse eşik yok sayılır, küme sayısı n_clusters'a inene kadar birleştirilir."""
    n = len(embs)
    if n == 0:
        return []
    X = np.stack([_norm(e) for e in embs])
    sim = X @ X.T
    clusters: list[list[int]] = [[i] for i in range(n)]
    while len(clusters) > 1:
        best, pair = None, None
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = 1.0 - float(sim[np.ix_(clusters[a], clusters[b])].mean())
                if best is None or d < best:
                    best, pair = d, (a, b)
        if best is None:
            break
        if n_clusters is not None:
            if len(clusters) <= n_clusters:
                break
        elif best > threshold:
            break
        a, b = pair  # type: ignore[misc]
        clusters[a] = clusters[a] + clusters[b]
        del clusters[b]
    labels = [0] * n
    for ci, members in enumerate(clusters):
        for m in members:
            labels[m] = ci
    return labels


def _embed(ctx: Ctx, items: list[dict], label: str) -> dict:
    if not items:
        return {"results": {}, "errors": {}}
    return run_worker(ctx.project, "diar", "diar_worker.py", {
        "action": "embed", "items": items,
        "options": {"token_env": ctx.cfg["diarization"]["hf_token_env"],
                    "model": "pyannote/wespeaker-voxceleb-resnet34-LM"},
    }, label=label)


def _load_yaml(path: Path) -> dict:
    if path.exists():
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return {}


def run(ctx: Ctx) -> None:
    p, cfg = ctx.project, ctx.cfg["speakers"]
    pending = ctx.manifest.select("speakers", ctx.force, ctx.only)
    if not pending:
        return
    rows = [r for r in ctx.manifest.select("speakers", force=True) if r["status"].get("diarize") == "ok"]
    pending_ids = {r["seg_id"] for r in pending}
    emb_dir = p.wdir("emb")

    # 1) embedding'ler (önbellekli)
    todo = []
    for r in rows:
        out = emb_dir / f"{r['seg_id']}.npy"
        seg_wav = p.abs(r["voice_path"])
        # Önbellek seg_id'ye bağlı; diarize segmenti yeniden kestiyse (aynı id, farklı ses) embedding bayattır
        stale = out.exists() and seg_wav.exists() and out.stat().st_mtime < seg_wav.stat().st_mtime
        if ctx.force or not out.exists() or stale:
            todo.append({"id": r["seg_id"], "path": str(p.abs(r["voice_path"])), "out": str(out)})
    try:
        res = _embed(ctx, todo, "speakers_embed")
    except WorkerError as e:
        for r in pending:
            add_error(r, "speakers", e)
        ctx.save()
        console.print(f"[red]speakers başarısız:[/] {e}")
        return
    for sid, err in res.get("errors", {}).items():
        r = ctx.manifest.get(sid)
        if r:
            add_error(r, "speakers", f"embedding: {err}")
    embs = {r["seg_id"]: np.load(emb_dir / f"{r['seg_id']}.npy") for r in rows
            if (emb_dir / f"{r['seg_id']}.npy").exists()}

    # 2) karakter ataması
    folder_mode = bool(cfg["folder_name_as_character"])
    to_cluster = [r for r in rows if r["seg_id"] in embs and not (folder_mode and r.get("char_hint"))]
    # Arka plan konuşması (megafon/telsiz/kalabalık) ana karakter kümelerine katılmaz: kendi aralarında
    # kümelenir (Arka1, Arka2, ...), konuşmacı sayısı ipucu (num_speakers) sadece diyaloğa uygulanır.
    bg_of = _cluster_bg([r for r in to_cluster if r.get("layer") == "bg"], embs,
                        float(cfg["cluster_threshold"]), str(cfg.get("bg_char_prefix", "Arka")))
    to_cluster = [r for r in to_cluster if r.get("layer") != "bg"]
    names = _load_yaml(p.speakers_yaml)
    # çapa: klasörsüz segment (örn. ara sahne repliği), bilinen bir klasör karakterinin merkezine
    # yeterince yakınsa doğrudan o karaktere atanır; kalanlar kümelenir
    anchor_of: dict[str, str] = {}
    anchors: dict[str, np.ndarray] = {}
    thr = float(cfg.get("anchor_threshold", cfg["cluster_threshold"]))
    if folder_mode and to_cluster:
        for ch in sorted({r["char_hint"] for r in rows if r.get("char_hint")}):
            mem = [_norm(embs[r["seg_id"]]) for r in rows if r.get("char_hint") == ch and r["seg_id"] in embs]
            if mem:
                anchors[ch] = _norm(np.mean(mem, axis=0))
        for r in to_cluster if anchors else []:
            e = _norm(embs[r["seg_id"]])
            ch, sim = max(((c, float(e @ v)) for c, v in anchors.items()), key=lambda x: x[1])
            r["anchor_sim"] = round(sim, 3)
            if 1 - sim <= thr:
                anchor_of[r["seg_id"]] = ch
        to_cluster = [r for r in to_cluster if r["seg_id"] not in anchor_of]
    cluster_of: dict[str, str] = {}
    if to_cluster:
        core = [r for r in to_cluster if r.get("dur_speech", 0) >= MIN_CLUSTER_CLIP_S] or to_cluster
        k = cfg.get("num_speakers")
        k = int(k) if k else None
        labels = agglomerative(np.stack([embs[r["seg_id"]] for r in core]), float(cfg["cluster_threshold"]),
                               n_clusters=min(k, len(core)) if k else None)
        # c1, c2, ... ilk görünme sırasına göre
        order: dict[int, str] = {}
        for r, lab in sorted(zip(core, labels), key=lambda x: (x[0]["file_id"], x[0]["start"])):
            order.setdefault(lab, f"c{len(order) + 1}")
            cluster_of[r["seg_id"]] = order[lab]
        cents = {}
        for cid in set(cluster_of.values()):
            members = [embs[s] for s, c in cluster_of.items() if c == cid]
            cents[cid] = _norm(np.mean([_norm(m) for m in members], axis=0))
        for r in to_cluster:
            if r["seg_id"] in cluster_of:
                continue
            e = _norm(embs[r["seg_id"]])
            cid, sim = max(((c, float(e @ v)) for c, v in cents.items()), key=lambda x: x[1])
            if not k and 1 - sim > float(cfg["cluster_threshold"]):  # konuşmacı sayısı verildiyse yeni küme açma
                cid = f"c{len(cents) + 1}"
                cents[cid] = e
            cluster_of[r["seg_id"]] = cid
        # küme → çapa: tek tek eşiği geçemeyen kısa replikler küme merkeziyle bir karaktere net
        # yakınsa (en yakın çapa eşiği geçer ve ikinciden belirgin uzaksa) o karaktere bağlanır
        if folder_mode and anchors:
            margin = float(cfg.get("anchor_margin", 0.1))
            for cid in sorted(set(cluster_of.values())):
                members = [s for s, c in cluster_of.items() if c == cid]
                cen = _norm(np.mean([_norm(embs[s]) for s in members], axis=0))
                sims = sorted(((float(cen @ v), ch) for ch, v in anchors.items()), reverse=True)
                best_sim, best_ch = sims[0]
                second = sims[1][0] if len(sims) > 1 else -1.0
                if 1 - best_sim <= thr and best_sim - second >= margin:
                    for s in members:
                        anchor_of[s] = best_ch
                        del cluster_of[s]
                    console.print(f"  küme {cid} ({len(members)} replik) → {best_ch} (merkez benzerliği {best_sim:.2f})")
            to_cluster = [r for r in to_cluster if r["seg_id"] not in anchor_of]

    new_yaml: dict = {}
    for cid in sorted(set(cluster_of.values()), key=lambda c: int(c[1:])):
        members = [r for r in to_cluster if cluster_of.get(r["seg_id"]) == cid]
        members.sort(key=lambda r: -r.get("dur_speech", 0))
        prev = names.get(cid) or {}
        new_yaml[cid] = {"name": prev.get("name"), "samples": [r["voice_path"] for r in members[:3]]}
    if new_yaml or p.speakers_yaml.exists():
        header = ("# dubpoc: cluster → karakter adı. 'name' alanlarını doldurun ve `dubpoc speakers` ile\n"
                  "# (veya `dubpoc run --from speakers`) tekrar çalıştırın. Boş kalırsa char = cluster id.\n")
        out_yaml = dict(new_yaml)
        if isinstance(names.get("overrides"), dict):  # elle düzeltmeler (overrides: {seg_id: karakter}) korunur
            out_yaml["overrides"] = names["overrides"]
        body = yaml.safe_dump(out_yaml, allow_unicode=True, sort_keys=False) if out_yaml else "{}\n"
        p.speakers_yaml.write_text(header + body, encoding="utf-8")
        unnamed = [c for c, v in new_yaml.items() if not v.get("name")]
        if unnamed:
            console.print(f"[yellow]speakers.yaml: {len(unnamed)} kümenin adı boş ({', '.join(unnamed)}).[/]")

    overrides = names.get("overrides") if isinstance(names.get("overrides"), dict) else {}
    for r in rows:
        old_char = r.get("char")
        if r["seg_id"] in overrides:  # speakers.yaml → overrides: {seg_id: karakter} (insan düzeltmesi önceliklidir)
            r["cluster"], r["char"] = "override", str(overrides[r["seg_id"]])
        elif folder_mode and r.get("char_hint"):
            r["cluster"], r["char"] = f"dir:{r['char_hint']}", r["char_hint"]
        elif r["seg_id"] in bg_of:
            r["cluster"], r["char"] = "bg", bg_of[r["seg_id"]]
        elif r["seg_id"] in anchor_of:
            r["cluster"], r["char"] = f"anchor:{anchor_of[r['seg_id']]}", anchor_of[r["seg_id"]]
        elif r["seg_id"] in cluster_of:
            cid = cluster_of[r["seg_id"]]
            r["cluster"] = cid
            r["char"] = (new_yaml.get(cid) or {}).get("name") or cid
        else:
            continue
        # --force sadece --only ile seçilen satırları geçersiz kılar (eskiden tüm dosyanın tts/fit/qa durumu sıfırlanıyordu)
        if r["status"]["speakers"] not in ("ok",) or old_char != r["char"] or (ctx.force and r["seg_id"] in pending_ids):
            clear_errors(r, "speakers")
            set_status(r, "speakers", "ok")

    # 3) ses bankası + baseline
    _build_banks(ctx, rows, embs)
    ctx.save()


def _cluster_bg(bg_rows: list[dict], embs: dict[str, np.ndarray], threshold: float, prefix: str) -> dict[str, str]:
    if not bg_rows:
        return {}
    bg_rows = sorted(bg_rows, key=lambda r: (r["file_id"], r["start"]))
    labels = (agglomerative(np.stack([embs[r["seg_id"]] for r in bg_rows]), threshold)
              if len(bg_rows) > 1 else [0])
    order: dict[int, str] = {}
    out = {}
    for r, lab in zip(bg_rows, labels):
        order.setdefault(lab, f"{prefix}{len(order) + 1}")
        out[r["seg_id"]] = order[lab]
    return out


def _build_banks(ctx: Ctx, rows: list[dict], embs: dict[str, np.ndarray]) -> None:
    p, cfg = ctx.project, ctx.cfg["speakers"]
    sil_db = float(ctx.cfg["fit"]["silence_db"])
    chars = sorted({r["char"] for r in rows if r.get("char")})
    ref_items = []
    bank_info: dict[str, dict] = {}
    for ch in chars:
        members = [r for r in rows if r.get("char") == ch]
        # arka plan konuşması klipleri (müzik/efektten kenetlemeyle çıkarılmış, gürültülü) diyalogu olan bir
        # karakterin bankasına girmez; sadece kendi kümesinin (Arka1, ...) bankası onlardan kurulur
        if any(r.get("layer") != "bg" for r in members):
            members = [r for r in members if r.get("layer") != "bg"]
        cands = []
        for r in members:
            x, sr = audio.read(p.abs(r["voice_path"]), mono=True)
            sp = audio.trim(x, sr, sil_db)
            cands.append((r, sp, sr, audio.snr_estimate(x, sr), len(sp) / sr))
        good = [c for c in cands if c[4] >= float(cfg["bank_min_clip_s"])] or cands
        good.sort(key=lambda c: (-round(c[3] / 3), -c[4]))  # ~3 dB'lik SNR kovaları, sonra uzunluk
        chosen, total = [], 0.0
        for c in good:
            if total >= float(cfg["bank_max_seconds"]):
                break
            chosen.append(c)
            total += c[4]
        # Uzun klipler yetmiyorsa (kısa repliklerden oluşan karakter) kısa kliplerle tamamla: 1,7 sn'lik
        # tek klipten klonlanan ses belirgin şekilde kötüydü (GTA testi)
        target_min = float(cfg.get("bank_target_min_s", 6.0))
        if total < target_min:
            for c in sorted((c for c in cands if c not in chosen and c[4] >= 0.35), key=lambda c: -c[4]):
                if total >= target_min:
                    break
                chosen.append(c)
                total += c[4]
        if not chosen:
            continue
        sr0 = chosen[0][2]
        gap = np.zeros(int(0.2 * sr0), dtype=np.float32)
        parts = []
        for (_, sp, sr, _, _) in chosen:
            parts += [audio.resample(sp, sr, sr0), gap]
        ref = np.concatenate(parts[:-1])[: int(float(cfg["bank_max_seconds"]) * sr0)]
        bdir = p.wdir("bank", ch)
        audio.write(bdir / "ref.wav", ref, sr0, subtype="PCM_16")
        clips = [c[0]["seg_id"] for c in chosen]
        (bdir / "clips.json").write_text(json.dumps(clips, ensure_ascii=False), encoding="utf-8")
        # Kısa referans: sadece TAM kliplerden (≤ bank_short_seconds). Transkript isteyen TTS'ler
        # (OmniVoice ref_text) için ses ve metin birebir eşleşmeli; kesik klip metni bozar.
        short_max = float(cfg.get("bank_short_seconds", 12))
        short, tot = [], 0.0
        for c in chosen:
            if short and tot + c[4] > short_max:
                break
            short.append(c)
            tot += c[4] + 0.2
        sparts = []
        for (_, sp, sr, _, _) in short:
            sparts += [audio.resample(sp, sr, sr0), gap]
        audio.write(bdir / "ref_short.wav", np.concatenate(sparts[:-1]), sr0, subtype="PCM_16")
        (bdir / "clips_short.json").write_text(json.dumps([c[0]["seg_id"] for c in short]), encoding="utf-8")
        bank_info[ch] = {"clips": clips, "members": [r["seg_id"] for r in members], "dur": round(len(ref) / sr0, 2)}
        ref_items.append({"id": ch, "path": str(bdir / "ref.wav"), "out": str(bdir / "ref_emb.npy")})

    try:
        res = _embed(ctx, ref_items, "speakers_bank")
    except WorkerError as e:
        console.print(f"[red]bank embedding başarısız:[/] {e}")
        return
    for ch, info in bank_info.items():
        bdir = p.work / "bank" / ch
        if not (bdir / "ref_emb.npy").exists():
            console.print(f"[red]{ch}: ref_emb üretilemedi: {res.get('errors', {}).get(ch)}[/]")
            continue
        ref_emb = _norm(np.load(bdir / "ref_emb.npy"))
        # Sadece güvenilir (yeterince uzun) klipler: "Hmm." gibi 0,4 sn'lik bir klibin embedding'i
        # anlamsızdır ve QA paydasını çökertir (oranlar 2-3'e fırlar).
        min_s = float(cfg["bank_min_clip_s"])
        dur_of = {r["seg_id"]: float(r.get("dur_speech") or 0) for r in rows}
        reliable = [s for s in info["members"] if s in embs and dur_of.get(s, 0) >= min_s]
        mem = [_norm(embs[s]) for s in reliable]
        pair = [float(mem[i] @ mem[j]) for i in range(len(mem)) for j in range(i + 1, len(mem))]
        to_ref = [float(m @ ref_emb) for m in mem]
        held = [float(_norm(embs[s]) @ ref_emb) for s in reliable if s not in info["clips"]]
        if len(held) < 2:
            held = []  # tek bir bank dışı klip güvenilir payda değil
        base = {
            "pairwise": round(float(np.mean(pair)), 4) if pair else None,
            "orig_to_ref": round(float(np.mean(to_ref)), 4) if to_ref else None,
            "orig_to_ref_heldout": round(float(np.mean(held)), 4) if held else None,
            "n_clips": len(mem), "bank_clips": info["clips"], "bank_dur": info["dur"],
        }
        # QA paydası: bank dışı orijinal kliplerin ref'e benzerliği (yoksa tümü, o da yoksa pairwise)
        base["qa_denominator"] = base["orig_to_ref_heldout"] or base["orig_to_ref"] or base["pairwise"]
        (bdir / "baseline.json").write_text(json.dumps(base, indent=1), encoding="utf-8")
        (bdir / "baseline_sim.txt").write_text(
            f"{base['pairwise'] if base['pairwise'] is not None else base['qa_denominator']}\n", encoding="utf-8")
        console.print(f"  bank {ch}: {info['dur']} sn, {len(info['clips'])} klip, "
                      f"baseline pairwise={base['pairwise']} to_ref={base['orig_to_ref']}")
