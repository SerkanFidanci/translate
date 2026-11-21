"""GPU gerektirmeyen birim testleri: metin, manifest, ses yardımcıları, QA skoru, kümeleme, fit/mix/report."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from dubpoc import audio
from dubpoc.manifest import Manifest, new_status, set_status
from dubpoc.pipeline import make_ctx
from dubpoc.project import Project
from dubpoc.stages import fit, mix, report
from dubpoc.stages.diarize import _postprocess
from dubpoc.stages.qa import js_similarity, score_candidate, text_errors
from dubpoc.stages.speakers import agglomerative
from dubpoc.stages.translate import _pick, syll_budget
from dubpoc.textutil import count_syllables_tr, is_nonverbal_text, normalize_tr, tr_lower

ROOT = Path(__file__).resolve().parents[1]


# ---- metin --------------------------------------------------------------------
def test_syllables():
    assert count_syllables_tr("Merhaba dünya") == 5
    assert count_syllables_tr("İstanbul'a gidiyoruz!") == 8
    assert count_syllables_tr("kâğıt") == 2


def test_turkish_lower_and_normalize():
    assert tr_lower("IŞIK İZMİR") == "ışık izmir"
    assert normalize_tr("Ankara'ya   GİDİYORUZ!!") == "ankaraya gidiyoruz"
    assert normalize_tr("Kâğıt, kalem...") == "kağıt kalem"


def test_text_errors_turkish_case():
    wer, cer = text_errors("IŞIK geldi", "ışık geldi.")
    assert wer == 0 and cer == 0


@pytest.mark.parametrize("text,expected", [
    ("", True), ("Uh...", True), ("[laughs]", True), ("Hahaha!", True), ("*grunts*", True),
    ("Run!", False), ("Get down, now!", False), ("Oh no, not again.", False),
])
def test_nonverbal(text, expected):
    assert is_nonverbal_text(text) is expected


def test_pick_shorten_candidate():
    cands = ["Hemen buradan çıkmamız lazım", "Çıkalım hemen", "Gidelim"]
    assert _pick(cands, 5) == "Çıkalım hemen"
    assert syll_budget({"dur_speech": 2.0}, 6.0) == 12


# ---- manifest ---------------------------------------------------------------------
def test_manifest_atomic_roundtrip(tmp_path):
    m = Manifest(tmp_path / "m.jsonl")
    r = {"seg_id": "f001_s01", "file_id": "f001", "start": 0.0, "status": new_status(), "errors": []}
    m.rows.append(r)
    set_status(r, "ingest", "ok")
    set_status(r, "translate", "ok")
    r["status"]["tts"] = "ok"
    set_status(r, "translate", "shorten")  # aşağı akış sıfırlanmalı
    assert r["status"]["tts"] == "pending" and r["status"]["ingest"] == "ok"
    m.save()
    m2 = Manifest(tmp_path / "m.jsonl")
    assert m2.rows[0]["status"]["translate"] == "shorten"
    assert [x["seg_id"] for x in m2.select("translate")] == ["f001_s01"]
    assert m2.select("ingest") == []
    assert len(m2.select("ingest", force=True)) == 1


# ---- ses --------------------------------------------------------------------------
def _tone(sr=24000, dur=1.0, lead=0.3, tail=0.2, amp=0.3):
    t = np.arange(int(sr * dur)) / sr
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    return np.concatenate([np.zeros(int(sr * lead), np.float32), x, np.zeros(int(sr * tail), np.float32)])


def test_speech_bounds_and_trim():
    x = _tone()
    s, e = audio.speech_bounds(x, 24000)
    assert abs(s - 0.3) < 0.03 and abs(e - 1.3) < 0.03
    assert abs(len(audio.trim(x, 24000, keep_ms=0)) / 24000 - 1.0) < 0.05


def test_time_stretch_pedalboard():
    x = _tone(lead=0, tail=0)
    y = audio.time_stretch(x, 24000, 1.1, "pedalboard")
    assert abs(len(y) / 24000 - 1 / 1.1) < 0.03


def test_true_peak_limit():
    x = (np.sin(np.linspace(0, 200, 48000)) * 1.2).astype(np.float32)
    y = audio.limit_true_peak(x, 48000, -1.0)
    assert audio.true_peak_db(y, 48000) <= -0.95


# ---- QA ------------------------------------------------------------------------
Q = {"weights": {"wer": 0.5, "spk_sim": 0.3, "dur": 0.2, "emo": 0.0}, "text_metric": "wer", "fail_wer": 0.4,
     "fail_cer": 0.25, "wer_max": 0.15, "cer_max": 0.08, "spk_sim_min_relative": 0.85, "dur_tolerance": 0.1}


def test_score_candidate_states():
    good = {"fit": "x", "wer": 0.05, "cer": 0.01, "spk_sim_rel": 0.95, "dur_ratio": 1.02}
    s, st = score_candidate(good, Q)
    assert st == "ok" and s > 0.8
    assert score_candidate({**good, "spk_sim_rel": 0.6}, Q)[1] == "review"
    assert score_candidate({**good, "wer": 0.5}, Q)[1] == "fail"
    assert score_candidate({**good, "fit": None}, Q)[1] == "fail"


def test_js_similarity():
    p = {"angry": 0.8, "neutral": 0.2, "other": 0.5}
    assert js_similarity(p, p) == 1.0
    assert js_similarity(p, {"happy": 1.0}) < 0.2


def test_agglomerative_two_speakers():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=64), rng.normal(size=64)
    embs = np.stack([a + rng.normal(scale=0.1, size=64) for _ in range(4)] +
                    [b + rng.normal(scale=0.1, size=64) for _ in range(3)])
    labels = agglomerative(embs, 0.55)
    assert len(set(labels[:4])) == 1 and len(set(labels[4:])) == 1 and labels[0] != labels[4]


def test_diarize_postprocess():
    turns = [{"start": 0, "end": 2, "speaker": "A"}, {"start": 2.1, "end": 2.3, "speaker": "A"},
             {"start": 3, "end": 3.2, "speaker": "B"}, {"start": 4, "end": 6, "speaker": "A"}]
    out = _postprocess(turns, 0.4)
    assert out[0]["end"] == 2.3            # aynı konuşmacı, kısa boşluk → birleşti
    assert out[1].get("short") is True and not out[1].get("skip")  # komşusu farklı konuşmacı → kendi segmenti (atlanmaz)


# ---- fit + mix + report uçtan uca (sentetik, GPU'suz) ---------------------------------------
@pytest.fixture()
def mini_project(tmp_path):
    root = tmp_path / "proj"
    (root / "workers").mkdir(parents=True)
    shutil.copy(ROOT / "config.yaml", root / "config.yaml")
    sr = 48000
    orig = _tone(sr=sr, dur=2.0, lead=0.25, tail=0.25)
    seg = root / "work" / "seg" / "f001_s01.wav"
    seg.parent.mkdir(parents=True)
    sf.write(seg, orig, sr)
    ing = root / "work" / "ingest" / "f001.wav"
    ing.parent.mkdir(parents=True)
    sf.write(ing, np.stack([orig, orig], axis=1), sr)
    dub = _tone(sr=24000, dur=2.15, lead=0.1, tail=0.3, amp=0.1)  # %7.5 uzun, sessiz
    cand = root / "work" / "tts" / "f001_s01" / "cb1.wav"
    cand.parent.mkdir(parents=True)
    sf.write(cand, dub, 24000)
    st = new_status()
    for k in ("ingest", "separate", "diarize", "speakers", "asr", "emotion", "translate", "tts", "vc"):
        st[k] = "ok"
    row = {"seg_id": "f001_s01", "file_id": "f001", "src_path": "input/a.wav", "src_sr": sr, "src_channels": 2,
           "src_dur": 2.5, "start": 0.0, "end": 2.5, "dur_src": 2.5, "ingest_path": "work/ingest/f001.wav",
           "voice_path": "work/seg/f001_s01.wav", "bg_path": None, "speech_start": 0.25, "speech_end": 2.25,
           "dur_speech": 2.0, "char": "hero", "text_en": "Hello there.", "text_tr": "Merhaba.",
           "candidates": {"A": [{"id": "cb1", "backend": "chatterbox_mtl", "tts": "work/tts/f001_s01/cb1.wav",
                                 "src": "work/tts/f001_s01/cb1.wav"}]},
           "status": st, "errors": []}
    (root / "work" / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    return Project(root)


def test_fit_mix_report(mini_project):
    ctx = make_ctx(mini_project)
    fit.run(ctx)
    r = ctx.manifest.rows[0]
    c = r["candidates"]["A"][0]
    assert r["status"]["fit"] == "ok", r["errors"]
    assert abs(c["dur_ratio"] - 1.0) < 0.02          # ±%15 içinde → tam hedefe esnetildi
    out, osr = audio.read(mini_project.abs(c["fit"]))
    assert osr == 48000 and out.shape[1] == 2
    # loudness orijinale eşitlendi (±1 dB)
    o, _ = audio.read(mini_project.abs(r["voice_path"]), mono=True)
    assert abs(audio.rms_dbfs(audio.trim(out.mean(axis=1), osr)) - audio.rms_dbfs(audio.trim(o, osr))) < 1.0
    r["best"] = {"A": 0}
    r["status"]["qa"] = "ok"
    c.update({"wer": 0.0, "cer": 0.0, "spk_sim_rel": 0.9, "score": 0.9, "qa": "ok"})
    ctx.save()
    mix.run(ctx)
    mixed = mini_project.out / "A" / "f001.wav"
    info = sf.info(str(mixed))
    assert info.samplerate == 48000 and info.channels == 2 and abs(info.duration - 2.5) < 0.01
    report.run(ctx)
    html = (mini_project.out / "report.html").read_text(encoding="utf-8")
    assert "f001_s01" in html and "../work/fit/" in html


def test_turkish_numbers():
    from dubpoc.textutil import tr_num2words
    assert tr_num2words(2) == "iki"
    assert tr_num2words(1000) == "bin"
    assert tr_num2words(2025) == "iki bin yirmi beş"
    assert tr_num2words(115) == "yüz on beş"
    assert tr_num2words(1_200_000) == "bir milyon iki yüz bin"
    wer, _ = text_errors("Gözcüler 2 dakikada nöbet değiştiriyor.", "Gözcüler iki dakikada nöbet değiştiriyor.")
    assert wer == 0


def test_fix_caps_tr():
    from dubpoc.textutil import fix_caps_tr
    assert fix_caps_tr("GÜVENLİ") == "Güvenli"
    assert fix_caps_tr("YERE YATIN! HEMEN ŞİMDİ.") == "Yere yatın! Hemen şimdi."
    assert fix_caps_tr("Bu GERÇEKTEN önemli, FBI geliyor.") == "Bu gerçekten önemli, FBI geliyor."
    assert fix_caps_tr("Normal bir cümle.") == "Normal bir cümle."


def test_agglomerative_fixed_k():
    rng = np.random.default_rng(1)
    embs = rng.normal(size=(6, 32))  # birbirine benzemeyen 6 vektör: eşikle 6 küme, k=2 ile 2 küme
    assert len(set(agglomerative(embs, 0.55))) == 6
    assert len(set(agglomerative(embs, 0.55, n_clusters=2))) == 2


def test_speakable_tr_numbers():
    from dubpoc.textutil import speakable_tr, tr_ordinal
    assert [tr_ordinal(n) for n in (1, 2, 3, 4, 6, 10, 40, 100)] == [
        "birinci", "ikinci", "üçüncü", "dördüncü", "altıncı", "onuncu", "kırkıncı", "yüzüncü"]
    assert speakable_tr("2. Birim, bu Komutanlık.") == "İkinci Birim, bu Komutanlık."
    assert speakable_tr("Gözcüler 2 dakikada geliyor.") == "Gözcüler iki dakikada geliyor."


def test_assign_words_nearest_segment():
    from dubpoc.stages.asr import _assign_words
    segs = [{"seg_id": "a", "start": 4.4, "end": 6.2}, {"seg_id": "b", "start": 7.0, "end": 7.8}]
    words = [{"w": "Wait,", "s": 4.1, "e": 4.5}, {"w": "what", "s": 4.6, "e": 4.8},
             {"w": "No.", "s": 6.9, "e": 7.05}, {"w": "uzak", "s": 12.0, "e": 12.3}]
    got = _assign_words(segs, words)
    assert [w["w"] for w in got["a"]] == ["Wait,", "what"]
    assert [w["w"] for w in got["b"]] == ["No."]      # biraz erken başlayan kelime de doğru segmente


def test_short_line_uses_cer():
    cand = {"fit": "x", "wer": 1.0, "cer": 0.0, "spk_sim_rel": 0.9, "dur_ratio": 1.0}  # "Biri mi?" → "BİRİMİ"
    assert score_candidate(cand, {**Q, "short_ref_words": 2}, ref_words=2)[1] == "ok"
    assert score_candidate(cand, Q, ref_words=5)[1] == "fail"


def test_prosody_score_and_slot_duration():
    from dubpoc.stages.qa import prosody_score, score_candidate
    assert prosody_score(None, 3.0) is None and prosody_score(1.0, 0) is None
    assert prosody_score(3.0, 3.0) == 1.0
    assert prosody_score(1.05, 3.0) == 0.5   # tekdüze okuma: 0.35x → 0.35/0.7
    assert prosody_score(9.6, 3.0) == 0.5    # abartılı: 3.2x → 1.6/3.2
    q = {"weights": {"wer": 0.5, "dur": 0.5}, "fail_wer": 0.4, "wer_max": 0.15, "cer_max": 0.08,
         "spk_sim_min_relative": 0.85, "dur_tolerance": 0.1}
    base = {"fit": "x.wav", "wer": 0.0, "dur_ratio": 1.2}
    s_hard, _ = score_candidate(base, q)
    s_slot, _ = score_candidate({**base, "dur_score_ratio": 1.1}, q)  # sonraki repliğe çarpmayan taşma
    assert s_slot > s_hard


def test_short_segments_kept_and_missed_speech_added():
    from dubpoc.stages.asr import unassigned_words
    from dubpoc.stages.diarize import _postprocess, add_missed_speech
    from dubpoc.textutil import is_nonverbal_text

    # pause-fallback: her parça ayrı etiket → kısa olan komşuya katılamaz ama artık atlanmaz
    out = _postprocess([{"start": 0.0, "end": 2.0, "speaker": "P00"}, {"start": 3.0, "end": 3.3, "speaker": "P01"}], 0.4)
    assert len(out) == 2 and out[1].get("short") and not out[1].get("skip")
    # VAD'ın bulduğu, enerjinin kaçırdığı bölge eklenir; çoğu örtüşen bölge eklenmez, parçalar birleşmez
    parts = add_missed_speech([(30.5, 32.9), (39.7, 40.6)], [(30.53, 33.02), (33.41, 33.88), (40.0, 40.5), (50.0, 50.05)])
    assert parts == [(30.5, 32.9), (33.41, 33.88), (39.7, 40.6)]
    # segmente düşmeyen güvenilir kelime raporlanır
    rows = [{"start": 10.0, "end": 12.0}]
    ws = [{"w": "Hey", "s": 10.5, "e": 10.8, "p": 0.9}, {"w": "Come", "s": 15.2, "e": 15.5, "p": 0.9},
          {"w": "uh", "s": 20.0, "e": 20.2, "p": 0.2}]
    assert [w["w"] for w in unassigned_words(rows, ws)] == ["Come"]
    # tek kelimelik gerçek sözcük dublajlanır, ünlem orijinal kalır
    assert not is_nonverbal_text("Fuck.") and not is_nonverbal_text("Surprise!")
    assert is_nonverbal_text("Mm-hmm.") and is_nonverbal_text("-") and is_nonverbal_text("[grunts]")


def test_qa_penalizes_overlap_with_next_line():
    from dubpoc.stages.qa import score_candidate

    q = {"weights": {"wer": 0.45, "spk_sim": 0.2, "dur": 0.1}, "fail_wer": 0.5, "fail_cer": 0.25, "wer_max": 0.15,
         "cer_max": 0.1, "spk_sim_min_relative": 0.75, "dur_tolerance": 0.2}
    base = {"fit": "x.wav", "cer": 0.0, "wer": 0.0, "dur_ratio": 1.5, "dur_score_ratio": 1.5}
    long_take = {**base, "spk_sim_rel": 0.83, "slot_overflow_s": 1.13}   # üstüne biniyor, tını daha benzer
    short_take = {**base, "spk_sim_rel": 0.55, "slot_overflow_s": 0.05}  # sığıyor
    assert score_candidate(short_take, q, 1)[0] > score_candidate(long_take, q, 1)[0]


def test_qa_prefers_turkish_sounding_take():
    from dubpoc.stages.qa import score_candidate

    q = {"weights": {"wer": 0.45, "spk_sim": 0.2, "dur": 0.1, "lang": 0.15}, "fail_wer": 0.5, "fail_cer": 0.25,
         "wer_max": 0.15, "cer_max": 0.1, "spk_sim_min_relative": 0.75, "dur_tolerance": 0.2}
    base = {"fit": "x.wav", "cer": 0.0, "wer": 0.0, "dur_ratio": 1.0}
    en_accent = {**base, "spk_sim_rel": 0.70, "lang_tr": 0.05}
    tr_accent = {**base, "spk_sim_rel": 0.62, "lang_tr": 0.99}
    assert score_candidate(tr_accent, q, 2)[0] > score_candidate(en_accent, q, 2)[0]


def test_pick_bg_speech_filters_hallucination_and_dialog_leak():
    from dubpoc.stages.diarize import pick_bg_speech
    cfg = {"min_logprob": -0.8, "max_no_speech": 0.6, "min_word_p": 0.45, "max_compression": 2.4,
           "min_rms_dbfs": -50.0, "max_dialog_overlap": 0.5}

    def seg(a, b, text, lp=-0.4, nsp=0.2, p=0.8):
        ws = text.split()
        step = (b - a) / len(ws)
        return {"start": a, "end": b, "text": text, "avg_logprob": lp, "no_speech_prob": nsp, "compression_ratio": 1.2,
                "words": [{"w": w, "s": a + i * step, "e": a + (i + 1) * step, "p": p} for i, w in enumerate(ws)]}

    segs = [seg(84.2, 87.5, "Get out of the vehicle with your hands in the air!"),  # megafon anonsu
            seg(9.7, 9.9, "Thank you.", nsp=0.89),                                    # müzik üstünde uydurma
            seg(193.9, 196.9, "Two, one, jump."),                                     # diyaloğun sızıntısı
            seg(120.0, 121.0, "Hey there.", p=0.2),                                   # güvensiz kelimeler
            seg(300.0, 301.0, "Over here."),                                          # duyulmayacak kadar sessiz
            seg(400.0, 401.0, "Twenty. Twenty. Twenty. Twenty.")]                     # tekrar döngüsü
    keep, leaks = pick_bg_speech(segs, [(193.3, 195.35), (195.4, 197.0)], cfg,
                                 rms_of=lambda a, b: -70.0 if a >= 300 and a < 302 else -30.0)
    assert [k["text"] for k in keep] == ["Get out of the vehicle with your hands in the air!"]
    assert [x["text"] for x in leaks] == ["Two, one, jump."]


def test_region_mask_and_match_spectrum():
    sr = 16000
    m = audio.region_mask(sr * 2, sr, [(0.5, 1.0), (0.9, 1.2)], fade_s=0.02)
    assert m[int(0.75 * sr)] == 1.0 and m[int(1.1 * sr)] == 1.0 and m[int(0.2 * sr)] == 0.0 and m[int(1.5 * sr)] == 0.0
    assert 0.0 < m[int(0.5 * sr)] < 1.0  # kenarda yumuşak geçiş
    from scipy.signal import butter, sosfilt, welch
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.1, size=sr * 2).astype(np.float32)
    ref = sosfilt(butter(6, [400, 3000], btype="band", fs=sr, output="sos"), rng.normal(scale=0.1, size=sr * 2))
    y = audio.match_spectrum(x, sr, ref.astype(np.float32), sr)
    f, py = welch(y, sr, nperseg=1024)
    hi = py[f > 5000].mean() / py[(f > 800) & (f < 2500)].mean()
    assert 10 * np.log10(hi) < -20  # telsiz/megafon gibi bant sınırlı oldu


def _burst_scene(sr, n, a, b):
    t = np.arange(n) / sr
    music = 0.05 * np.sin(2 * np.pi * 200 * t)
    env = ((t >= a) & (t < b)).astype(float)
    speech = env * sum(0.08 / k * np.sin(2 * np.pi * 150 * k * t) for k in range(3, 12))  # harmonik "konuşma"
    return music.astype(np.float32), speech.astype(np.float32)


def _band_db(x, sr, a, b, lo, hi):
    seg = x[int(a * sr): int(b * sr)]
    spec = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1 / sr)
    return 10 * np.log10(spec[(f >= lo) & (f <= hi)].sum() + 1e-20)


def test_suppress_speech_keeps_music():
    sr = 16000
    music, speech = _burst_scene(sr, sr * 6, 2.5, 3.5)
    y, s = audio.suppress_speech(music + speech, sr, [(2.4, 3.6)])
    assert _band_db(y, sr, 2.6, 3.4, 400, 1700) < _band_db(music + speech, sr, 2.6, 3.4, 400, 1700) - 15
    assert abs(_band_db(y, sr, 2.6, 3.4, 190, 210) - _band_db(music, sr, 2.6, 3.4, 190, 210)) < 2  # müzik kaldı
    assert np.allclose(y[: sr], (music + speech)[: sr], atol=1e-4)  # bölge dışına dokunulmadı
    assert _band_db(s, sr, 2.6, 3.4, 400, 1700) > _band_db(speech, sr, 2.6, 3.4, 400, 1700) - 3  # konuşma tahmini


def test_bg_duplicates_marks_leak_of_dialog():
    from dubpoc.stages.asr import bg_duplicates
    rows = [{"seg_id": "f001_s25", "start": 112.99, "end": 113.52, "text_en": "Fuck.", "status": {}},
            {"seg_id": "f001_b03", "layer": "bg", "start": 111.5, "end": 112.9, "text_en": "Fuck.", "status": {}},
            {"seg_id": "f001_b01", "layer": "bg", "start": 66.0, "end": 67.6, "text_en": "Fuck.", "status": {}}]
    assert [(r["seg_id"], d["seg_id"]) for r, d in bg_duplicates(rows)] == [("f001_b03", "f001_s25")]


def test_mix_suppresses_leak_and_skips_undubbed_bg_line(mini_project):
    p = mini_project
    sr, n = 48000, int(2.5 * 48000)
    music, speech = _burst_scene(sr, n, 1.0, 1.6)
    sf.write(p.root / "work" / "seg" / "bg.wav", np.stack([music + speech] * 2, axis=1), sr)
    sf.write(p.root / "work" / "seg" / "f001_b01.wav", speech[: sr], sr)
    ctx = make_ctx(p)
    fit.run(ctx)
    r = ctx.manifest.rows[0]
    r.update({"best": {"A": 0}, "bg_path": "work/seg/bg.wav"})
    r["status"]["qa"] = "ok"
    leak = json.loads(json.dumps(r))
    leak.update({"seg_id": "f001_b01", "layer": "bg", "start": 0.9, "end": 1.7, "dur_src": 0.8, "bg_region": [0.95, 1.65],
                 "bg_leak_of": "f001_s01", "voice_path": "work/seg/f001_b01.wav", "candidates": {}, "best": {}})
    leak["status"]["diarize"] = "skip"
    undubbed = json.loads(json.dumps(leak))
    undubbed.update({"seg_id": "f001_b02", "bg_leak_of": None, "start": 2.0, "end": 2.4, "bg_region": [2.0, 2.4]})
    undubbed["status"]["diarize"] = "ok"
    ctx.manifest.rows += [leak, undubbed]
    ctx.save()
    mix.run(ctx)
    out, _ = audio.read(p.out / "A" / "f001.wav", mono=True)
    dub, _ = audio.read(p.abs(r["candidates"]["A"][0]["fit"]), mono=True)
    rest = out[: len(dub)] - dub[: len(out)]  # mikste dublaj dışında kalan = işlenmiş arka plan
    assert _band_db(rest, sr, 1.1, 1.5, 400, 1700) < _band_db(music + speech, sr, 1.1, 1.5, 400, 1700) - 15
    rows = {x["seg_id"]: x for x in ctx.manifest.rows}
    assert rows["f001_b02"]["mix_source"]["A"] == "background"  # dublajı olmayan bg satırının orijinali eklenmedi
    assert rows["f001_b01"]["mix_source"]["A"] == "background"


def test_bg_word_timing_refinement():
    from dubpoc.stages.diarize import matching_words, tighten_words
    ws = [{"w": "Shit!", "s": 63.15, "e": 64.01, "p": 0.4}, {"w": "Fuck!", "s": 67.02, "e": 67.57, "p": 0.8}]
    assert [w["w"] for w in matching_words("Fuck.", ws)] == ["Fuck!"]
    assert matching_words("Get out of the vehicle", ws) == []
    t = tighten_words([{"w": "Shit.", "s": 61.65, "e": 64.01, "p": 0.7}])  # DTW önceki müziği kelimeye katmış
    assert 63.4 < t[0]["s"] < 63.6 and t[0]["e"] == 64.01
    t = tighten_words([{"w": "Get", "s": 84.77, "e": 85.21}, {"w": "out", "s": 85.21, "e": 85.41}])
    assert t[0]["s"] == 84.77  # makul süreli kelimeye dokunulmaz


def _harmonic_voice(sr, dur, f0=140.0, seed=0):
    """Basit "konuşma": 3-4 Hz hece zarflı, geniş bantlı harmonik dizi."""
    t = np.arange(int(sr * dur)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k ** 0.7 for k in range(1, int(7000 / f0)))
    env = 0.5 * (1 + np.sin(2 * np.pi * 3.5 * t)) ** 2
    env *= np.clip(np.minimum(t, t[-1] - t) / 0.05, 0, 1)  # başta/sonda yumuşak giriş-çıkış (gerçek konuşma gibi)
    return (0.05 * x * env).astype(np.float32)


def _hz_db(x, sr, lo, hi, ref=(400, 3000)):
    from scipy.signal import welch
    f, p = welch(x, sr, nperseg=2048)
    return 10 * np.log10(p[(f >= lo) & (f < hi)].mean() / p[(f >= ref[0]) & (f < ref[1])].mean())


def test_channel_detection_phone_and_hint_gating():
    from dubpoc import fx
    sr = 48000
    clean = _harmonic_voice(sr, 2.0)
    phone = fx.bandpass(clean, sr, 320, 3300, order=8)
    pc = fx.band_profile(clean, sr)
    pp = fx.band_profile(phone, sr)
    assert fx.choose_kind(pc, None)[0] is None
    assert fx.choose_kind(pp, None)[0] == "phone" and 180 < pp["f_lo"] < 420 and 3000 < pp["f_hi"] < 4200
    assert fx.choose_kind(pp, "bg")[0] is None  # bg ses tahmininde alt bant güvenilmez → otomatik telefon yok
    # style notu türü verir ama ancak orijinal gerçekten dar bantsa (telsize konuşan adamın kendisi tam bant)
    assert fx.choose_kind(pc, None, style="police radio dispatcher, flat tone")[0] is None
    assert fx.choose_kind(pp, "bg", style="shouting through a megaphone")[0] == "megaphone"
    assert fx.choose_kind(pc, None, explicit="megaphone")[0] == "megaphone"
    assert fx.choose_kind(pp, None, explicit="none")[0] is None


def test_channel_effect_chains():
    from dubpoc import fx
    sr = 48000
    x = _harmonic_voice(sr, 1.5)
    ph = fx.apply(x, sr, "phone", {"f_lo": 300, "f_hi": 3400})
    assert len(ph) == len(x)
    assert _hz_db(ph, sr, 4500, 8000) < _hz_db(x, sr, 4500, 8000) - 25
    assert _hz_db(ph, sr, 60, 200) < _hz_db(x, sr, 60, 200) - 15
    mg = fx.apply(x, sr, "megaphone", {"f_hi": 4000})
    assert len(mg) > len(x) + int(0.5 * sr)  # dış mekân yankı kuyruğu
    assert _hz_db(mg, sr, 6000, 10000) < _hz_db(x, sr, 6000, 10000) - 20
    tail = mg[len(x) + int(0.05 * sr): len(x) + int(0.3 * sr)]
    assert audio.rms_dbfs(tail) > audio.rms_dbfs(mg[: len(x)]) - 30  # konuşma bitince yankı sürüyor
    # distorsiyon: harmonik olmayan bileşen / tepe-ortalama oranı düşer (sıkıştırılmış, "boğuk bağırma")
    crest = lambda z: 20 * np.log10(np.max(np.abs(z)) / np.sqrt(np.mean(z ** 2)))
    assert crest(mg[: len(x)]) < crest(fx.bandpass(x, sr, 450, 4000))


def test_fit_applies_phone_effect_and_keeps_dry_copy(mini_project):
    p = mini_project
    sr = 48000
    clean = _harmonic_voice(sr, 2.0)
    orig = 0.25 * np.concatenate([np.zeros(int(0.25 * sr)), fx_bp := __import__("dubpoc.fx", fromlist=["x"]).bandpass(
        clean, sr, 320, 3300, order=8), np.zeros(int(0.25 * sr))]).astype(np.float32)
    sf.write(p.root / "work" / "seg" / "f001_s01.wav", orig, sr)
    sf.write(p.root / "work" / "tts" / "f001_s01" / "cb1.wav", audio.resample(_harmonic_voice(sr, 2.1, f0=170), sr, 24000), 24000)
    ctx = make_ctx(p)
    fit.run(ctx)
    r = ctx.manifest.rows[0]
    c = r["candidates"]["A"][0]
    assert r["fx"]["kind"] == "phone" and c["fx"] == "phone" and c["fit_dry"]
    wet, _ = audio.read(p.abs(c["fit"]), mono=True)
    dry, _ = audio.read(p.abs(c["fit_dry"]), mono=True)
    assert _hz_db(wet, sr, 4500, 8000) < _hz_db(dry, sr, 4500, 8000) - 20
    assert abs(audio.rms_dbfs(audio.trim(wet, sr)) - audio.rms_dbfs(audio.trim(dry, sr))) < 1.5  # ikisi de orijinal seviyede


def test_leak_extra_words_split_off():
    from dubpoc.stages.asr import leak_extra_words
    ws = [{"w": "Two,", "s": 194.6, "e": 195.08, "p": 0.9}, {"w": "one,", "s": 195.1, "e": 195.96, "p": 0.95},
          {"w": "jump.", "s": 196.04, "e": 196.9, "p": 0.94}]
    assert [w["w"] for w in leak_extra_words(ws, "Three, two, one.", 193.3, 195.35)] == ["jump."]
    assert leak_extra_words(ws[:2], "Three, two, one.", 193.3, 195.35) == []  # hepsi diyalogda
    assert [w["w"] for w in leak_extra_words(ws, "Jump!", 196.0, 196.9)] == ["Two,", "one,"]  # baştaki fazlalık
    head = [{"w": "Hey!", "s": 10.0, "e": 10.3, "p": 0.8}, {"w": "Stop", "s": 10.6, "e": 10.9, "p": 0.9}]
    assert [w["w"] for w in leak_extra_words(head, "Stop right there.", 10.55, 11.8)] == ["Hey!"]
    assert leak_extra_words([{"w": "Oh,", "s": 9.0, "e": 9.2, "p": 0.9}, *head[1:]], "Stop.", 10.55, 11.0) == []


def test_vad_chunk_merge(monkeypatch):
    import types
    from dubpoc.stages import diarize
    calls = []

    def fake_ts(y, opts):
        calls.append(len(y))
        return [{"start": 0, "end": 16000}] if len(calls) == 1 else [{"start": 2 * 16000 - 8000, "end": 2 * 16000}]
    mod = types.ModuleType("faster_whisper.vad")
    mod.VadOptions = lambda **k: k
    mod.get_speech_timestamps = fake_ts
    monkeypatch.setitem(__import__("sys").modules, "faster_whisper.vad", mod)
    out = diarize._vad_regions(np.zeros(16000 * 30, dtype=np.float32), 16000)
    assert len(calls) == 2 and out == [(0.0, 1.0), (19.5, 20.0)]


def test_fit_loudness_includes_leaked_part(mini_project):
    """Replik ses ve arka plan stem'ine bölünmüşse dublaj seviyesi ikisinin toplamına eşlenir."""
    p = mini_project
    sr, n = 48000, int(6.0 * 48000)  # bölge çevresinde müzik bağlamı olsun (kenetleme medyanı)
    music, speech = _burst_scene(sr, n, 0.3, 2.2)
    sf.write(p.root / "work" / "seg" / "bg.wav", np.stack([music + speech * 8] * 2, axis=1), sr)  # asıl gövde arka planda
    ctx = make_ctx(p)
    fit.run(ctx)
    r = ctx.manifest.rows[0]
    quiet, _ = audio.read(p.abs(r["candidates"]["A"][0]["fit"]), mono=True)
    leak = json.loads(json.dumps(r))
    leak.update({"seg_id": "f001_b01", "layer": "bg", "bg_leak_of": "f001_s01", "bg_region": [0.2, 2.3],
                 "candidates": {}, "best": {}})
    r["bg_path"] = "work/seg/bg.wav"
    ctx.manifest.rows.append(leak)
    ctx.save()
    ctx = make_ctx(p, force=True, only="f001_s01")
    fit.run(ctx)
    loud, _ = audio.read(p.abs(ctx.manifest.get("f001_s01")["candidates"]["A"][0]["fit"]), mono=True)
    assert audio.rms_dbfs(audio.trim(loud, sr)) > audio.rms_dbfs(audio.trim(quiet, sr)) + 3
