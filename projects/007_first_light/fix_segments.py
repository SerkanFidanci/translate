"""007 fragmanı: otomatik segmentlemenin elle düzeltmeleri (sıfırdan kurulumda `--to asr`dan sonra bir kez çalıştır).

- s15: "Do you have an appointment?" (resepsiyonist, f0 ~350 Hz) + "The name's Bond." (Bond, ~115 Hz) → s15 + s15b
- s17: "He's with me." (kadın ajan, ~250 Hz) + "And you are?" (Bond, ~130 Hz) → s17 + s17b
- s31: "Lately," Bond'un cevabının başı (143,1 sn): klip 142,9 sn'den başlar
- s32 + s33: tek replik ("I suppose this is another teachable ... moment.") → s32

Konuşmacılar (speakers.yaml overrides), düzeltilmiş İngilizce (subtitles.csv) ve çeviriler (translations.csv) ayrı
dosyalarda duruyor. Sonra: `dubpoc --root projects/007_first_light run --from speakers --force`.

Kullanım (ana dubpoc klasöründen):
    envs\\core\\Scripts\\python.exe projects\\007_first_light\\fix_segments.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / "src"))
from dubpoc import audio  # noqa: E402
from dubpoc.pipeline import make_ctx  # noqa: E402
from dubpoc.project import Project  # noqa: E402
from dubpoc.stages.diarize import FILE_FIELDS, dialog_row  # noqa: E402

ctx = make_ctx(Project(ROOT))
fid = "f001"
rows = ctx.manifest.rows_of_file(fid)
by = {r["seg_id"]: r for r in rows}
if "f001_s15b" in by:
    sys.exit("zaten düzeltilmiş (f001_s15b var)")
need = ["f001_s15", "f001_s16", "f001_s17", "f001_s18", "f001_s31", "f001_s32", "f001_s33", "f001_s34"]
missing = [s for s in need if s not in by]
if missing:
    sys.exit(f"segmentleme farklı çıkmış, bu satırlar yok: {missing} — zamanları elle kontrol edin")
shutil.copy(ROOT / "work/manifest.jsonl", ROOT / "work/manifest.before_fix.jsonl")
r0 = rows[0]
voice, vsr = audio.read(ctx.project.abs(r0["file_voice_path"]), mono=True)
base = {k: r0.get(k) for k in FILE_FIELDS}
base_status = {k: ("ok" if k in ("ingest", "separate") else "pending") for k in r0["status"]}
sil = float(ctx.cfg["fit"]["silence_db"])


def mk(seg_id, a, b, nxt):
    return dialog_row(ctx, base, base_status, seg_id, voice, vsr, a, b, nxt, "fix", sil)


edits = {
    "f001_s15": [("f001_s15", 70.59, 71.55, 71.55), ("f001_s15b", 71.55, 72.53, by["f001_s16"]["start"])],
    "f001_s17": [("f001_s17", 74.58, 75.05, 75.05), ("f001_s17b", 75.05, 76.02, by["f001_s18"]["start"])],
    "f001_s31": [("f001_s31", 142.90, 144.05, by["f001_s32"]["start"])],
    "f001_s32": [("f001_s32", 172.87, 175.67, by["f001_s34"]["start"])],
    "f001_s33": [],
}
new_rows = []
for r in rows:
    new_rows += [mk(*e) for e in edits[r["seg_id"]]] if r["seg_id"] in edits else [r]
ctx.manifest.replace_file_rows(fid, sorted(new_rows, key=lambda r: r["start"]))
ctx.save()
print(f"tamam: {len(new_rows)} satır")
