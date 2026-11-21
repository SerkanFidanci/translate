"""Ses G/Ç, loudness, sessizlik kırpma, zaman esnetme ve limiter yardımcıları.

Tasarım notları:
- Dosya okuma soundfile ile (wav/flac/ogg/mp3); diğer formatlar ffmpeg ile wav'a çevrilir.
- Zaman esnetme varsayılan olarak pedalboard'un gömülü Rubber Band R3 motoruyla yapılır
  (harici rubberband.exe gerekmez). Sırayla: pedalboard → ffmpeg rubberband filtresi →
  rubberband CLI → ffmpeg atempo.
- True-peak limiter: 4x oversample edilmiş sinyal üzerinde tepe ölçümü + pedalboard Limiter.
"""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SF_EXT = {".wav", ".flac", ".ogg", ".mp3", ".aiff", ".aif"}


# ---- G/Ç --------------------------------------------------------------------------
def read(path: Path | str, mono: bool = False) -> tuple[np.ndarray, int]:
    """(samples, channels) float32 döndürür. mono=True ise (samples,)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if mono:
        data = data.mean(axis=1)
    return data, sr


def write(path: Path | str, data: np.ndarray, sr: int, subtype: str = "FLOAT") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sf.write(str(tmp), data, sr, subtype=subtype)
    tmp.replace(path)
    return path


def info(path: Path | str) -> dict:
    i = sf.info(str(path))
    return {"sr": i.samplerate, "channels": i.channels, "frames": i.frames,
            "dur": i.frames / i.samplerate, "subtype": i.subtype}


def duration(path: Path | str) -> float:
    return info(path)["dur"]


def resample(data: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return data
    return soxr.resample(data, sr_in, sr_out, quality="HQ").astype(np.float32)


def to_channels(data: np.ndarray, channels: int) -> np.ndarray:
    if data.ndim == 1:
        data = data[:, None]
    if data.shape[1] == channels:
        return data
    mono = data.mean(axis=1, keepdims=True)
    return np.repeat(mono, channels, axis=1)


def to_mono_16k(path: Path | str) -> tuple[np.ndarray, int]:
    x, sr = read(path, mono=True)
    return resample(x, sr, 16000), 16000


# ---- dönüştürme ----------------------------------------------------------------------
def run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(f"Komut başarısız ({p.returncode}): {' '.join(map(str, cmd))}\n{p.stderr[-2000:]}")
    return p


def convert_to_wav(src: Path, dst: Path, ffmpeg: str | None, vgmstream: str | None = None) -> Path:
    """Kaynağı orijinal SR ve kanal sayısıyla float wav'a çevirir. Kaynak dosyaya dokunmaz."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ext = src.suffix.lower()
    if ext == ".wem":
        if not vgmstream:
            raise RuntimeError(".wem girdisi için vgmstream-cli gerekli (config paths.vgmstream)")
        tmp = dst.with_suffix(".vgm.wav")
        run([vgmstream, "-o", str(tmp), str(src)])
        data, sr = read(tmp)
        write(dst, data, sr)
        tmp.unlink(missing_ok=True)
        return dst
    if ext in SF_EXT:
        try:
            data, sr = read(src)
            write(dst, data, sr)
            return dst
        except Exception:
            if not ffmpeg:
                raise
    if not ffmpeg:
        raise RuntimeError(f"{ext} formatı için ffmpeg gerekli")
    run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-c:a", "pcm_f32le", str(dst)])
    return dst


# ---- ölçümler -------------------------------------------------------------------------
def rms_dbfs(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    r = float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0
    return 20 * np.log10(max(r, 1e-10))


def lufs(x: np.ndarray, sr: int) -> float | None:
    """BS.1770 integrated loudness; < 0.4 sn ya da tamamen sessiz ise None."""
    import pyloudnorm as pyln

    if x.shape[0] < int(0.4 * sr) + 1:
        return None
    meter = pyln.Meter(sr)
    try:
        val = float(meter.integrated_loudness(x.astype(np.float64)))
    except Exception:
        return None
    if not np.isfinite(val) or val < -70:
        return None
    return val


def loudness(x: np.ndarray, sr: int) -> tuple[float, str]:
    """(değer, tür) – LUFS ölçülebiliyorsa LUFS, yoksa RMS dBFS."""
    v = lufs(x, sr)
    return (v, "lufs") if v is not None else (rms_dbfs(x), "rms")


def true_peak_db(x: np.ndarray, sr: int) -> float:
    up = soxr.resample(x, sr, sr * 4) if x.size else x
    peak = float(np.max(np.abs(up))) if up.size else 0.0
    return 20 * np.log10(max(peak, 1e-10))


def speech_bounds(x: np.ndarray, sr: int, thresh_db: float = -45.0, frame_ms: float = 10.0,
                  min_speech_ms: float = 30.0) -> tuple[float, float]:
    """Baştaki/sondaki sessizliği atlayan konuşma aralığı (sn). Sessizse (0, 0)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = max(1, int(sr * frame_ms / 1000))
    n = len(x) // hop
    if n == 0:
        return 0.0, 0.0
    frames = x[: n * hop].reshape(n, hop)
    db = 20 * np.log10(np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1)) + 1e-10)
    active = db > thresh_db
    k = max(1, int(min_speech_ms / frame_ms))
    # kısa tıkırtıları yok saymak için hareketli pencere
    smooth = np.convolve(active.astype(int), np.ones(k, dtype=int), mode="same") >= max(1, k // 2 + 1)
    idx = np.flatnonzero(smooth if smooth.any() else active)
    if idx.size == 0:
        return 0.0, 0.0
    start = idx[0] * hop / sr
    end = min(len(x), (idx[-1] + 1) * hop) / sr
    return float(start), float(end)


def split_on_pauses(x: np.ndarray, sr: int, min_pause_s: float = 0.35, min_seg_s: float = 0.3,
                    rel_db: float = 28.0, floor_db: float = -50.0) -> list[tuple[float, float]]:
    """Duraklamalardan konuşma parçaları (sn). Eşik göreli: p95 seviyesinin rel_db altı (müzik sızıntısına dayanıklı)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(sr * 0.01)
    n = len(x) // hop
    if n == 0:
        return []
    db = 20 * np.log10(np.sqrt(np.mean(x[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1)) + 1e-10)
    p95, p10 = float(np.percentile(db, 95)), float(np.percentile(db, 10))
    # eşik: konuşma tepesinin rel_db altı ya da gürültü tabanının (örn. müzik sızıntısı) biraz üstü
    thr = max(floor_db, p95 - rel_db, p10 + 0.35 * (p95 - p10))
    active = np.convolve((db > thr).astype(int), np.ones(5, dtype=int), mode="same") >= 3
    segs, start, quiet = [], None, 0
    min_quiet = int(min_pause_s / 0.01)
    for i, a in enumerate(active):
        if a:
            if start is None:
                start = i
            quiet = 0
        elif start is not None:
            quiet += 1
            if quiet >= min_quiet:
                segs.append((start, i - quiet + 1))
                start, quiet = None, 0
    if start is not None:
        segs.append((start, n - quiet))
    return [(s * hop / sr, e * hop / sr) for s, e in segs if (e - s) * hop / sr >= min_seg_s]


def trim(x: np.ndarray, sr: int, thresh_db: float = -45.0, keep_ms: float = 15.0) -> np.ndarray:
    s, e = speech_bounds(x, sr, thresh_db)
    if e <= s:
        return x[:0]
    pad = keep_ms / 1000
    a = max(0, int((s - pad) * sr))
    b = min(len(x), int((e + pad) * sr))
    return x[a:b]


def snr_estimate(x: np.ndarray, sr: int) -> float:
    """Basit enerji tabanlı SNR tahmini: konuşma çerçeveleri (p95) / gürültü tabanı (p10)."""
    if x.ndim > 1:
        x = x.mean(axis=1)
    hop = int(sr * 0.02)
    n = len(x) // hop
    if n < 5:
        return 0.0
    e = np.mean(x[: n * hop].reshape(n, hop).astype(np.float64) ** 2, axis=1) + 1e-12
    return float(10 * np.log10(np.percentile(e, 95) / np.percentile(e, 10)))


# ---- zaman esnetme -------------------------------------------------------------------
def available_stretch_engines(ffmpeg: str | None, rubberband: str | None) -> list[str]:
    engines = []
    try:
        import pedalboard  # noqa: F401
        if hasattr(pedalboard, "time_stretch"):
            engines.append("pedalboard")
    except ImportError:
        pass
    if ffmpeg and ffmpeg_has_filter(ffmpeg, "rubberband"):
        engines.append("ffmpeg_rubberband")
    if rubberband:
        engines.append("rubberband_cli")
    if ffmpeg:
        engines.append("atempo")
    return engines


_FILTER_CACHE: dict[tuple[str, str], bool] = {}


def ffmpeg_has_filter(ffmpeg: str, name: str) -> bool:
    key = (ffmpeg, name)
    if key not in _FILTER_CACHE:
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-filters"], capture_output=True, text=True,
                                 encoding="utf-8", errors="replace", timeout=30).stdout
            _FILTER_CACHE[key] = any(line.split()[1:2] == [name] for line in out.splitlines() if line.strip())
        except Exception:
            _FILTER_CACHE[key] = False
    return _FILTER_CACHE[key]


def time_stretch(x: np.ndarray, sr: int, speed: float, engine: str, ffmpeg: str | None = None,
                 rubberband: str | None = None) -> np.ndarray:
    """speed > 1 → hızlandır (kısalt), speed < 1 → yavaşlat. Perde korunur. x: mono."""
    if abs(speed - 1.0) < 1e-3 or x.size == 0:
        return x
    if engine == "pedalboard":
        from pedalboard import time_stretch as pb_stretch
        y = pb_stretch(x[None, :].astype(np.float32), float(sr), stretch_factor=float(speed),
                       high_quality=True, transient_mode="crisp")
        return np.asarray(y[0], dtype=np.float32)
    with tempfile.TemporaryDirectory() as td:
        src, dst = Path(td) / "in.wav", Path(td) / "out.wav"
        sf.write(str(src), x, sr, subtype="FLOAT")
        if engine == "ffmpeg_rubberband":
            run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af",
                 f"rubberband=tempo={speed:.6f}:pitchq=quality:window=standard", str(dst)])
        elif engine == "rubberband_cli":
            run([rubberband, "-3", "--tempo", f"{speed:.6f}", str(src), str(dst)])
        elif engine == "atempo":
            run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-af",
                 f"atempo={speed:.6f}", str(dst)])
        else:
            raise ValueError(f"Bilinmeyen esnetme motoru: {engine}")
        y, _ = read(dst, mono=True)
        return y


# ---- dinamik ---------------------------------------------------------------------------
def limit_true_peak(x: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """True-peak tavanı aşılırsa önce limiter, gerekirse kalan farkı kazançla düzelt."""
    tp = true_peak_db(x, sr)
    if tp <= ceiling_db:
        return x
    try:
        from pedalboard import Limiter, Pedalboard
        board = Pedalboard([Limiter(threshold_db=ceiling_db - 0.3, release_ms=60.0)])
        arr = x.T if x.ndim > 1 else x[None, :]
        y = board(arr.astype(np.float32), sr)
        y = y.T if x.ndim > 1 else y[0]
    except ImportError:
        y = x
    tp2 = true_peak_db(y, sr)
    if tp2 > ceiling_db:  # limiter intersample tepeleri tam yakalayamazsa güvenli kazanç
        y = y * (10 ** ((ceiling_db - tp2) / 20))
    return y.astype(np.float32)


def match_spectrum(x: np.ndarray, sr: int, ref: np.ndarray, ref_sr: int, n_fft: int = 2048,
                   smooth_oct: float = 1 / 3, max_boost_db: float = 12.0, max_cut_db: float = 36.0) -> np.ndarray:
    """x'in uzun dönem spektrumunu ref'inkine benzet: megafon/telsiz/uzaktan gelen sesin rengi (bant sınırı,
    rezonans) dublaja aktarılır. Toplam seviye ayrıca eşlenmeli (kazanç konuşma bandına göre göreli)."""
    from scipy.signal import istft, stft, welch

    if x.size < 256 or ref.size < 256:
        return x
    ref = resample(ref, ref_sr, sr)
    nper = min(n_fft, len(x), len(ref))
    f, px = welch(x, sr, nperseg=nper)
    _, pr = welch(ref, sr, nperseg=nper)

    def smooth(p):
        out = np.empty_like(p)
        for i, fi in enumerate(f):
            m = (f >= fi * 2 ** (-smooth_oct / 2)) & (f <= fi * 2 ** (smooth_oct / 2))
            out[i] = p[m].mean() if m.any() else p[i]
        return out

    g = 10 * np.log10(smooth(pr) + 1e-20) - 10 * np.log10(smooth(px) + 1e-20)
    band = (f >= 300) & (f <= 3400)
    g = g - (np.max(g[band]) if band.any() else np.max(g))  # en güçlü konuşma bandı 0 dB: kesme ağırlıklı
    g = np.clip(g, -max_cut_db, max_boost_db)
    fz, _, z = stft(x, sr, nperseg=nper)
    gz = np.interp(fz, f, g)
    _, y = istft(z * (10 ** (gz / 20))[:, None], sr, nperseg=nper)
    y = y[: len(x)] if len(y) >= len(x) else np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def region_mask(n: int, sr: int, regions: list[tuple[float, float]], fade_s: float = 0.06) -> np.ndarray:
    """0/1 zaman maskesi (bölgelerin birleşimi), kenarlarda fade_s'lik doğrusal geçiş."""
    m = np.zeros(n, dtype=np.float32)
    for a, b in regions:
        i0, i1 = max(0, int(a * sr)), min(n, int(b * sr))
        if i1 > i0:
            m[i0:i1] = 1.0
    f = int(fade_s * sr / 2)
    if f > 0 and m.any():
        w = 2 * f + 1
        c = np.cumsum(np.concatenate([[0.0], np.pad(m.astype(np.float64), (f, f))]))
        m = ((c[w:] - c[:-w]) / w).astype(np.float32)
    return m


def suppress_speech(x: np.ndarray, sr: int, regions: list[tuple[float, float]], k_db: float = 3.0,
                    context_s: float = 1.5, fmin: float = 120.0, fmax: float = 7000.0, fade_s: float = 0.08,
                    n_fft: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Arka plandaki konuşmayı bölge içinde spektral kenetlemeyle bastır → (temiz, konuşma tahmini).

    Ayrıştırma modelleri (DnR, vocals RoFormer, gürültü giderici) megafon/telsiz anonsunu ve uzaktan
    bağırmayı konuşma saymayıp efekt/müzikte bıraktı (GTA testi). Burada her frekans bandı, bölgenin
    çevresindeki (±context_s) arka planın medyan seviyesinin k_db üstüne kırpılır: müzik/efekt dokusu
    kalır, ondan yükselen konuşma harmonikleri gider. Faz korunur; stereo'da kazanç orta kanaldan.
    """
    from scipy.ndimage import uniform_filter
    from scipy.signal import istft, stft

    x2 = (x if x.ndim > 1 else x[:, None]).astype(np.float32)
    y = x2.copy()
    hop = n_fft // 4
    for a, b in sorted(regions):
        c0, c1 = max(0, int((a - context_s) * sr)), min(len(x2), int((b + context_s) * sr))
        if c1 - c0 < 2 * n_fft:
            continue
        seg = y[c0:c1].copy()
        f, t, z = stft(seg.T, sr, nperseg=n_fft, noverlap=n_fft - hop)
        mag = np.abs(z.mean(axis=0))
        tt = t + c0 / sr
        inside = (tt >= a) & (tt <= b)
        ref = mag[:, ~inside] if (~inside).sum() >= 4 else mag
        cap = np.median(ref, axis=1) * 10 ** (k_db / 20)
        g = np.minimum(1.0, cap[:, None] / np.maximum(mag, 1e-12))
        g[(f < fmin) | (f > fmax)] = 1.0
        g[:, ~inside] = 1.0
        g = uniform_filter(g, size=(3, 3))
        _, out = istft(z * g[None], sr, nperseg=n_fft, noverlap=n_fft - hop)
        out = out.T[: c1 - c0].astype(np.float32)
        if len(out) < c1 - c0:
            out = np.pad(out, ((0, c1 - c0 - len(out)), (0, 0)))
        m = region_mask(c1 - c0, sr, [(a - c0 / sr, b - c0 / sr)], fade_s)
        y[c0:c1] = seg + m[:, None] * (out - seg)
    s = x2 - y
    return (y, s) if x.ndim > 1 else (y[:, 0], s[:, 0])


def match_loudness(x: np.ndarray, sr: int, target: float, kind: str) -> np.ndarray:
    cur = lufs(x, sr) if kind == "lufs" else None
    if cur is None:
        cur = rms_dbfs(x)
        if kind == "lufs":  # hedef LUFS ama kısa klip: RMS ile yaklaşık eşitle
            kind = "rms"
    gain_db = float(np.clip(target - cur, -30, 30))
    return (x * (10 ** (gain_db / 20))).astype(np.float32)


def which(name: str) -> str | None:
    return shutil.which(name)
