"""5.7 translate (env: core) — hece bütçeli, sahne bağlamlı EN→TR çeviri + out/script.md.

- syll_target = round(dur_speech * tr_syllables_per_sec). (Spec dur_src diyor; baştaki/sondaki
  sessizlik bütçeyi şişirmesin diye konuşma süresi kullanılır, yoksa dur_src.)
- Hece sayımı kodda doğrulanır (Türkçe hece = sesli harf sayısı).
- Kısaltma: model tek seferde 3 kısa alternatif üretir, kod hedefe en yakın olanı seçer
  (IWSLT 2025 / EMNLP 2025 isochrony bulguları: örnekli istem + çoklu aday + seçim).
- fit aşaması bir segmenti status.translate="shorten" yaparsa ölçülen taşmaya göre yeni hedefle kısaltılır.
"""
from __future__ import annotations

import json
import math
import time

from ..llm import LLM, LLMError
from ..manifest import add_error, clear_errors, is_inactive, set_status
from ..textutil import count_syllables_tr, speakable_tr
from . import Ctx, console

SYSTEM = "Sen deneyimli bir oyun dublaj çevirmenisin. Sadece istenen JSON'u döndürürsün."

PROMPT = """Aşağıdaki replikleri İngilizceden doğal, konuşma dili Türkçeye çevir.
Kurallar:
- Her replik verilen hece bütçesine (syll_target) ±%10 içinde sığmalı (Türkçede hece = sesli harf sayısı).
- Karakterin üslubunu ve duygusunu koru (emotion/intensity alanlarına bak). Argo ve küfür seviyesini koru.
- Dudak senkronu yok; anlamı ve süreyi önceliklendir. Kelime kelime çevirme.
- Özel isimleri, yer adlarını, oyun terimlerini çevirme.
- Ünlem, duraksama ("...") ve vurguyu noktalamayla ver; TTS bunları kullanır.
- Sayıları, kısaltmaları ve sembolleri yazıyla yaz (TTS ve hece sayımı için): "3" → "üç", "%50" → "yüzde elli".
- Sadece "translate": true olan replikleri çevir; diğerleri yalnızca bağlam içindir.
Sadece şu JSON'u döndür: {{"lines": [{{"seg_id": "...", "text_tr": "..."}}]}}

Sahne bağlamı (sırayla):
{scene}"""

SHORTEN = """Bir oyun dublajı repliğini kısaltman gerekiyor. Anlamı, duyguyu ve karakter üslubunu koru;
gerekirse ayrıntıyı at, eş anlamlı daha kısa kelimeler kullan. Türkçede hece = sesli harf sayısı.

Örnekler (hedef çok kısa olsa da sığdırılır):
- "Hemen buradan çıkmamız lazım, yoksa hepimiz öleceğiz!" (17 hece) → hedef 8: "Çıkalım, yoksa öldük!"
- "Bunu yapabileceğini gerçekten düşünüyor musun?" (16 hece) → hedef 7: "Yapabilir misin sence?"
- "Sana söylemiştim, o adama asla güvenmemeliydik." (15 hece) → hedef 9: "Demiştim, güvenmeyecektik."

Karakter: {char} | Duygu: {emotion} (yoğunluk {intensity})
İngilizce orijinal: {text_en}
Mevcut Türkçe ({cur} hece): {text_tr}
Hedef: en fazla {target} hece.

Hedefe uyan 3 farklı alternatif üret. Sadece JSON döndür: {{"candidates": ["...", "...", "..."]}}"""

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object",
        "properties": {"seg_id": {"type": "string"}, "text_tr": {"type": "string"}},
        "required": ["seg_id", "text_tr"], "additionalProperties": False}}},
    "required": ["lines"], "additionalProperties": False,
}
SHORTEN_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {"type": "string"}}},
    "required": ["candidates"], "additionalProperties": False,
}


def syll_budget(row: dict, rate: float) -> int:
    d = row.get("dur_speech") or row.get("dur_src") or 1.0
    return max(1, int(round(float(d) * rate)))


def _pick(cands: list[str], target: int) -> str | None:
    cands = [c.strip() for c in cands if c and c.strip()]
    if not cands:
        return None
    # önce hedefin altında/±%10'da kalanlar, sonra hedefe en yakın
    return min(cands, key=lambda c: (count_syllables_tr(c) > target * 1.1, abs(count_syllables_tr(c) - target)))


def _shorten(llm: LLM, row: dict, target: int) -> str | None:
    msg = SHORTEN.format(char=row.get("char"), emotion=row.get("emotion"), intensity=row.get("intensity"),
                         text_en=row.get("text_en"), text_tr=row["text_tr"],
                         cur=count_syllables_tr(row["text_tr"]), target=target)
    data = llm.json_call(SYSTEM, msg, SHORTEN_SCHEMA, name="shorten")
    best = _pick([speakable_tr(c) for c in data.get("candidates", [])], target)
    if best and count_syllables_tr(best) < count_syllables_tr(row["text_tr"]):
        return best
    return None


def run(ctx: Ctx) -> None:
    cfg = ctx.cfg["translate"]
    rate = float(cfg["tr_syllables_per_sec"])
    max_rounds = int(cfg["max_shorten_rounds"])
    trig = float(cfg.get("shorten_trigger_syll_ratio", 1.15))
    rows = [r for r in ctx.manifest.select("translate", ctx.force, ctx.only) if r["status"].get("emotion") == "ok"]
    if not rows:
        write_script(ctx)
        return
    # 0) translations.csv (seg_id,text_tr): insan/elle çeviri her zaman LLM'den önceliklidir
    manual = load_manual(ctx.project.root / "translations.csv")
    styles = load_manual(ctx.project.root / "translations.csv", column="style")
    for r in [r for r in rows if r["seg_id"] in manual]:
        r["tr_style"] = styles.get(r["seg_id"]) or None  # "yönetmen notu" (VoxCPM2 duygu talimatı)
        new = speakable_tr(manual[r["seg_id"]])
        if r.get("text_tr") != new:
            r["text_tr_time"] = time.time()
        r.update({"text_tr": new, "tr_source": "manual", "shorten_round": 0, "syll_shorten_rounds": 0})
        r["syll_target"] = syll_budget(r, rate)
        r["tr_syllables"] = count_syllables_tr(new)
        clear_errors(r, "translate")
        set_status(r, "translate", "ok")
    rows = [r for r in rows if r["seg_id"] not in manual]
    if manual:
        console.print(f"translate: {len(manual)} satır translations.csv'den")
    if not rows:
        ctx.save()
        write_script(ctx)
        return
    llm = LLM(cfg)
    console.print(f"translate: {len(rows)} segment, {llm.describe()}")
    try:
        # 1) fit'ten gelen kısaltma istekleri (ölçülen süre taşmasına göre)
        fit_req = [r for r in rows if r["status"]["translate"] == "shorten" and r.get("text_tr")]
        handled = {r["seg_id"] for r in fit_req}
        for r in fit_req:
            over = float(r.get("fit_overflow") or 1.3)
            target = max(1, math.floor(count_syllables_tr(r["text_tr"]) / over * 0.95))
            new = None
            try:
                new = _shorten(llm, r, target)
            except LLMError as e:
                add_error(r, "translate", e)
                continue
            r["shorten_round"] = int(r.get("shorten_round", 0)) + 1
            if new:
                console.print(f"  {r['seg_id']} kısaltıldı ({count_syllables_tr(r['text_tr'])}→"
                              f"{count_syllables_tr(new)} hece): {new}")
                r.setdefault("text_tr_history", []).append(r["text_tr"])
                r["text_tr"] = new
                r["text_tr_time"] = time.time()
            r["tr_syllables"] = count_syllables_tr(r["text_tr"])
            set_status(r, "translate", "ok")
        rows = [r for r in rows if r["seg_id"] not in handled]

        # 2) sahne (dosya) bazında toplu çeviri
        for fid in sorted({r["file_id"] for r in rows}):
            todo = {r["seg_id"] for r in rows if r["file_id"] == fid}
            scene_rows = [r for r in ctx.manifest.rows_of_file(fid) if not is_inactive(r) and r.get("text_en")]
            scene = []
            for r in scene_rows:
                r["syll_target"] = syll_budget(r, rate)
                item = {"seg_id": r["seg_id"], "char": r.get("char"), "emotion": r.get("emotion"),
                        "intensity": r.get("intensity"), "syll_target": r["syll_target"],
                        "text_en": r["text_en"], "translate": r["seg_id"] in todo}
                if r["seg_id"] not in todo and r.get("text_tr"):
                    item["text_tr"] = r["text_tr"]
                scene.append(item)
            try:
                data = llm.json_call(SYSTEM, PROMPT.format(scene=json.dumps(scene, ensure_ascii=False, indent=0)),
                                     TRANSLATE_SCHEMA, name="translation")
            except LLMError as e:
                for sid in todo:
                    add_error(ctx.manifest.get(sid), "translate", e)
                console.print(f"[red]{fid} çeviri hatası:[/] {e}")
                continue
            got = {d["seg_id"]: speakable_tr(d["text_tr"].strip()) for d in data.get("lines", []) if d.get("seg_id") in todo}
            for sid in todo:
                r = ctx.manifest.get(sid)
                if not got.get(sid):
                    add_error(r, "translate", "LLM bu segment için çeviri döndürmedi")
                    continue
                if r.get("text_tr") != got[sid]:
                    r["text_tr_time"] = time.time()
                r["text_tr"] = got[sid]
                r["tr_source"] = "llm"
                r["shorten_round"] = 0
                r["syll_shorten_rounds"] = 0
                r.pop("text_tr_history", None)
                # 3) hece bütçesi aşıldıysa kısaltma turları
                while (count_syllables_tr(r["text_tr"]) > r["syll_target"] * trig
                       and r["syll_shorten_rounds"] < max_rounds):
                    r["syll_shorten_rounds"] += 1
                    try:
                        new = _shorten(llm, r, r["syll_target"])
                    except LLMError as e:
                        console.print(f"[yellow]{sid} kısaltma hatası: {e}[/]")
                        break
                    if not new:
                        break
                    r.setdefault("text_tr_history", []).append(r["text_tr"])
                    r["text_tr"] = new
                    r["text_tr_time"] = time.time()
                r["tr_syllables"] = count_syllables_tr(r["text_tr"])
                clear_errors(r, "translate")
                set_status(r, "translate", "ok")
            ctx.save()
    finally:
        llm.release()
    ctx.save()
    write_script(ctx)


def load_manual(path, column: str = "text_tr") -> dict[str, str]:
    """translations.csv: seg_id,text_tr[,style] (UTF-8). Boş hücreler yok sayılır.
    style: İngilizce kısa oyunculuk talimatı, örn. "shocked and alarmed, urgent" (VoxCPM2 kullanır)."""
    import csv

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {(row.get("seg_id") or "").strip(): (row.get(column) or "").strip()
                for row in csv.DictReader(f) if (row.get("seg_id") or "").strip() and (row.get(column) or "").strip()}


def write_script(ctx: Ctx) -> None:
    """out/script.md — Yol B için okuma senaryosu."""
    p = ctx.project
    rows = [r for r in ctx.manifest.rows if not is_inactive(r) and r.get("text_tr")]
    lines = ["# Yol B okuma senaryosu", "",
             "Her repliği hedef süreye yakın, belirtilen duyguyla okuyun. Kaydı `record/<seg_id>.wav` "
             "olarak kaydedin (mono, 24 kHz+ önerilir; sessiz ortam, ağız-mikrofon ~20 cm).", "",
             "| seg_id | karakter | duygu | hedef süre | EN | TR | kayıt dosyası |",
             "|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["file_id"], r["start"])):
        dur = r.get("dur_speech") or r.get("dur_src")
        emo = f"{r.get('emotion', '?')} ({r.get('intensity', 0):.2f})"
        esc = lambda s: str(s or "").replace("|", "\\|").replace("\n", " ")  # noqa: E731
        lines.append(f"| {r['seg_id']} | {esc(r.get('char'))} | {emo} | {dur:.2f} sn | {esc(r.get('text_en'))} | "
                     f"{esc(r.get('text_tr'))} | `record/{r['seg_id']}.wav` |")
    p.odir().joinpath("script.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
