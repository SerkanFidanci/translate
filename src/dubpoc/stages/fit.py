"""5.10 fit (env: core) — süre ve ses seviyesi eşleme.

Her aday için:
1. Baştaki/sondaki sessizlik kırpılır (silence_db eşiği).
2. ratio = dub konuşma süresi / orijinal konuşma süresi (dur_speech).
   - Yol A'da adayların en iyisinin "doğal" oranı (VC length_adjust geri alınarak) shorten_trigger_ratio'yu
     aşarsa: status.translate = "shorten" → translate→tts→vc→fit sadece bu segment için tekrar
     (maks max_shorten_rounds).
   - |ratio-1| ≤ max_stretch → tam hedef süreye esnetilir (perde korunur).
   - Uzun kalırsa en fazla max_stretch kadar sıkıştırılır (taşma QA'da görünür); kısa kalırsa esnetilmez,
     sessizlikle doldurulur.
3. Orijinal replikteki konuşma başlangıcı (speech_start, en az pad_ms) kadar başa sessizlik eklenir.
4. Kanal efekti (fx modülü): orijinal replik telefondan/megafondan/telsizden duyuluyorsa dublaja aynı cihaz
   zinciri uygulanır (bant sınırları orijinalden ölçülür, sonda tınısı aktarılır). Tür: translations.csv "fx"
   sütunu / fx.overrides > style notu (+ bant ölçümü) > otomatik bant profili (telefon). Efektsiz kopya
   (fit_dry) QA ölçümleri için ayrıca yazılır. Efekt yoksa arka plan konuşmasına (layer "bg") orijinalin uzun
   dönem spektrumu aktarılır (fit.bg_match_spectrum).
   Loudness orijinal segmentin konuşma kısmına eşitlenir (LUFS; < 0.4 sn ise RMS), sonra -1 dBTP limiter.
5. Orijinal SR ve kanal sayısına çevrilir → work/fit/<seg_id>_<A_cid|B>.wav
"""
from __future__ import annotations

import numpy as np

from .. import audio, fx
from ..manifest import add_error, clear_errors, set_status
from . import Ctx, console


def _engine(ctx: Ctx) -> str:
    p, want = ctx.project, ctx.cfg["fit"].get("stretch_engine", "auto")
    avail = audio.available_stretch_engines(p.tool("ffmpeg"), p.tool("rubberband"))
    if want != "auto":
        if want not in avail:
            raise RuntimeError(f"fit.stretch_engine={want} kullanılamıyor (mevcut: {avail})")
        return want
    if not avail:
        raise RuntimeError("Zaman esnetme motoru yok (pedalboard / ffmpeg / rubberband)")
    return avail[0]


def channel_fx(ctx: Ctx, row: dict, manual: dict[str, str]) -> dict | None:
    """Satırın kanal efekti (yoksa None): orijinal konuşmanın bant profili + elle/style ipucu."""
    c = ctx.cfg.get("fx") or {}
    if not c.get("enabled", True):
        return None
    x, sr = reference_speech(ctx, row)
    prof = fx.band_profile(x, sr)
    explicit = (c.get("overrides") or {}).get(row["seg_id"]) or manual.get(row["seg_id"])
    kind, why = fx.choose_kind(prof, row.get("layer"), explicit, row.get("tr_style"), auto=bool(c.get("auto", True)),
                               phone_lo_db=float(c.get("phone_lo_db", -6.0)),
                               phone_hi_db=float(c.get("phone_hi_db", -26.0)),
                               hint_hi_db=float(c.get("hint_hi_db", -24.0)))
    return {"kind": kind, "why": why, **prof} if kind else None


_REF_CACHE: dict[str, tuple] = {}


def reference_speech(ctx: Ctx, row: dict) -> tuple[np.ndarray, int]:
    """Orijinal repliğin konuşma kısmı (loudness ve tını referansı). Ayrıştırıcı repliği iki stem'e bölmüşse
    (arka plan satırı bg_leak_of ile bu repliği gösteriyorsa) arka plandaki kısım da eklenir: 007 fragmanında
    "So you shouldn't have." ses stem'inde -34,7 LUFS, arka plan stem'inde -21,7 LUFS'tu; dublaj yalnızca ilkine
    eşlenip arka plan kısmı da mix'te bastırılınca replik müziğin altında kayboldu."""
    key = row["seg_id"]
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    p = ctx.project
    ox, osr = audio.read(p.abs(row["voice_path"]), mono=True)
    s0, s1 = row.get("speech_start", 0.0), row.get("speech_end") or len(ox) / osr
    ospeech = ox[int(s0 * osr): int(s1 * osr)].astype(np.float32)
    leaks = [q for q in ctx.manifest.rows_of_file(row["file_id"])
             if q.get("bg_leak_of") == row["seg_id"] and q.get("bg_region")]
    if leaks and row.get("layer") != "bg" and row.get("bg_path") and ctx.cfg["fit"].get("leak_loudness", True):
        bg, bsr = audio.read(p.abs(row["bg_path"]), mono=True)
        bg = audio.resample(bg, bsr, osr)
        # konuşma tahmini: bg satırlarının sesiyle aynı yöntem (çevre medyanının üstünde kalan, k=0)
        _, est = audio.suppress_speech(bg, osr, [tuple(q["bg_region"]) for q in leaks], k_db=0.0)
        a = int(round((float(row["start"]) + s0) * osr))
        part = est[a: a + len(ospeech)]
        ospeech = ospeech + np.pad(part, (0, len(ospeech) - len(part))).astype(np.float32)
    _REF_CACHE[key] = (ospeech, osr)
    return ospeech, osr


def fit_one(ctx: Ctx, row: dict, cand: dict, out_name: str, engine: str) -> dict:
    p, c = ctx.project, ctx.cfg["fit"]
    sil = float(c["silence_db"])
    x, sr = audio.read(p.abs(cand["src"]), mono=True)
    speech = audio.trim(x, sr, sil, keep_ms=10)
    if speech.size == 0:
        raise RuntimeError("aday tamamen sessiz")
    target = float(row.get("dur_speech") or row["dur_src"])
    dub = len(speech) / sr
    ratio = dub / target
    ms = float(c["max_stretch"])
    slow = float(c["max_slowdown"]) if c.get("max_slowdown") is not None else ms
    lead = max(float(row.get("speech_start", 0.0)), float(c["pad_ms"]) / 1000)
    use_slot = bool(c.get("use_slot", False))
    # use_slot: uzun aday sonraki repliğin başlangıcına kadar (slot_margin_s pay) taşabilir;
    # yalnızca ondan da uzunsa en fazla max_stretch sıkıştırılır.
    allowed = target
    if use_slot:
        slot = float(row.get("slot_dur") or row["dur_src"])
        allowed = max(target, slot - lead - float(c.get("slot_margin_s", 0.12)))
    exact = False
    if ratio <= 1:
        exact = (1 - ratio) <= slow
        speed = ratio if exact else 1.0  # yavaşlatma sınırı aşılırsa sessizlikle doldurulur
    elif use_slot:
        speed = 1.0 if dub <= allowed else min(dub / allowed, 1 + ms)
    else:
        exact = ratio <= 1 + ms
        speed = ratio if exact else 1 + ms
    y = audio.time_stretch(speech, sr, speed, engine, p.tool("ffmpeg"), p.tool("rubberband")) if speed != 1.0 else speech
    if exact:  # esnetme sonrası örnek sayısını tam hedefe eşitle
        n = int(round(target * sr))
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))

    # loudness: orijinal segmentin konuşma kısmıyla aynı yöntemle ölç
    ospeech, osr = reference_speech(ctx, row)
    dry = y
    kind = (row.get("fx") or {}).get("kind")
    if kind:
        # telefon/megafon/telsiz: cihaz zinciri + orijinalin tınısı (fx modülü)
        y = fx.apply(y, sr, kind, row["fx"], ref=ospeech, ref_sr=osr)
    elif row.get("layer") == "bg" and c.get("bg_match_spectrum", True):
        # arka plan konuşması: uzaktan gelen sesin rengini (bant sınırı, rezonans) aktar;
        # temiz stüdyo sesiyle okunan replik sahnede "önde" ve yapay duruyordu
        y = audio.match_spectrum(y, sr, ospeech, osr)
    o_l, o_kind = audio.loudness(ospeech, osr)

    def level(z: np.ndarray) -> np.ndarray:
        ol = o_l
        d_l, d_kind = audio.loudness(z, sr)
        if o_kind != d_kind:
            ol, d_l = audio.rms_dbfs(ospeech), audio.rms_dbfs(z)
        return (z * (10 ** (float(np.clip(ol - d_l, -30, 30)) / 20))).astype(np.float32)

    src_sr, ch = int(row["src_sr"]), int(row["src_channels"])
    fitted_speech = len(dry) / sr  # megafon yankı kuyruğu süreye sayılmaz

    def place(z: np.ndarray, name: str) -> str:
        total = max(float(row["dur_src"]), lead + len(z) / sr + 0.05)
        out = np.zeros(int(round(total * sr)), dtype=np.float32)
        a = int(round(lead * sr))
        out[a: a + len(z)] = z[: len(out) - a]
        out = audio.resample(out, sr, src_sr)
        out = audio.limit_true_peak(out, src_sr, float(c["true_peak_db"]))
        out = audio.to_channels(out, ch)
        path = p.wdir("fit") / f"{row['seg_id']}_{name}.wav"
        audio.write(path, out, src_sr)
        return p.rel(path)

    extra = {}
    if kind:  # QA efektsiz kopyayı ölçer (dar bant/distorsiyon Whisper'ı ve konuşmacı benzerliğini düşürüyor)
        extra = {"fit_dry": place(level(dry), out_name + "_dry"), "fx": kind}
    fit_path = place(level(y), out_name)
    dr = fitted_speech / target
    score_r = dr
    if use_slot and ((dr > 1 and fitted_speech <= allowed + 1e-3) or dr < 1):
        score_r = 1 + 0.5 * (dr - 1)  # sonraki repliğe çarpmayan fark yarı ağırlıkla
    return {"fit": fit_path, **extra, "dub_speech_dur": round(dub, 3), "fit_speech_dur": round(fitted_speech, 3),
            "stretch_speed": round(speed, 4), "dur_ratio": round(dr, 4), "dur_score_ratio": round(score_r, 4),
            "slot_allowed": round(allowed, 3),
            # doğal modda (use_slot) sonraki repliğin üstüne binen kısım; QA bunu ağır cezalandırır
            "slot_overflow_s": round(max(0.0, fitted_speech - allowed), 3) if use_slot else 0.0,
            "post_vc_ratio": round(ratio, 4), "stretch_engine": engine if speed != 1.0 else None}


def run(ctx: Ctx) -> None:
    p, c = ctx.project, ctx.cfg["fit"]
    max_rounds = int(ctx.cfg["translate"]["max_shorten_rounds"])
    trig = float(c["shorten_trigger_ratio"])
    rows = [r for r in ctx.manifest.select("fit", ctx.force, ctx.only)
            if r["status"].get("vc") == "ok" or r["status"].get("tts") == "ok"]
    if not rows:
        return
    engine = _engine(ctx)
    console.print(f"fit: {len(rows)} segment, esnetme motoru: {engine}")
    from .translate import load_manual

    manual_fx = load_manual(p.root / "translations.csv", column="fx")
    _REF_CACHE.clear()
    n_short = 0
    for r in rows:
        r["fx"] = channel_fx(ctx, r, manual_fx)
        if r["fx"]:
            console.print(f"  {r['seg_id']}: kanal efekti {r['fx']['kind']} ({r['fx']['why']})")
        cands_a = r.get("candidates", {}).get("A", []) if ctx.want("A") else []
        cands_b = r.get("candidates", {}).get("B", []) if ctx.want("B") else []
        target = float(r.get("dur_speech") or r["dur_src"])
        # Yol A: kısaltma gerekli mi? (VC length_adjust etkisini geri alarak doğal oran)
        nat = []
        for cand in cands_a:
            try:
                x, sr = audio.read(p.abs(cand["src"]), mono=True)
            except Exception:  # noqa: BLE001
                continue
            s0, s1 = audio.speech_bounds(x, sr, float(c["silence_db"]))
            if s1 > s0:
                cand["natural_ratio"] = round((s1 - s0) / float(cand.get("length_adjust", 1.0)) / target, 4)
                nat.append(cand["natural_ratio"])
        if nat and min(nat) > trig and int(r.get("shorten_round", 0)) < max_rounds and r.get("tr_source") != "manual":
            r["fit_overflow"] = round(min(nat), 4)
            set_status(r, "translate", "shorten")
            n_short += 1
            console.print(f"  {r['seg_id']}: en iyi aday {min(nat):.2f}x uzun → çeviri kısaltılacak "
                          f"(tur {int(r.get('shorten_round', 0)) + 1}/{max_rounds})")
            continue
        errs = []
        for cand in cands_a + cands_b:
            cand.pop("fit_dry", None)
            cand.pop("fx", None)
        for cand in cands_a:
            try:
                cand.update(fit_one(ctx, r, cand, "A_" + cand["id"].replace("+", "_"), engine))
            except Exception as e:  # noqa: BLE001
                cand["fit"] = None
                errs.append(f"A {cand['id']}: {e}")
        for cand in cands_b:
            try:
                cand.update(fit_one(ctx, r, cand, "B", engine))
            except Exception as e:  # noqa: BLE001
                cand["fit"] = None
                errs.append(f"B: {e}")
        if any(cand.get("fit") for cand in cands_a + cands_b):
            clear_errors(r, "fit")
            r["fit_warnings"] = errs or None
            set_status(r, "fit", "ok")
        else:
            add_error(r, "fit", "; ".join(errs) or "fit edilecek aday yok")
    ctx.save()
    if n_short:
        console.print(f"fit: {n_short} segment kısaltma için translate'e geri gönderildi")


def pending_shorten(ctx: Ctx) -> list[dict]:
    return [r for r in ctx.manifest.rows if r["status"].get("translate") == "shorten"]
