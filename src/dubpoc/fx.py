"""Kanal efektleri: dublaja orijinal repliğin "cihaz" sesini ver (telefon, megafon, telsiz).

GTA testi (kullanıcı: "polisler megafonla bağırıyor, aynı efekti veremiyor", "telefonla konuşmanın ses efekti
olmamış"): uzun dönem spektrum eşleme (audio.match_spectrum) yalnızca ortalama tınıyı aktarıyordu. Megafonu
megafon yapan bozulma (distorsiyon), sıkıştırma ve dış mekân yankısı; telefonu telefon yapan 8 kHz'lik dar bant
ve codec'ti. Dublaj stüdyolarındaki "futz" eklentileri gibi sabit, açıklanabilir zincirler kullanılır; bant
sınırları orijinal repliğin spektrumundan ölçülür, sonda orijinalin tınısı yine match_spectrum ile aktarılır.

Tür seçimi (fit aşaması): translations.csv'deki "fx" sütunu ya da fx.overrides > style notundaki anahtar kelime
("megaphone", "on the phone", "radio" ...) > otomatik tespit (bant profili telefon gibiyse "phone").
"""
from __future__ import annotations

import re

import numpy as np

KINDS = ("phone", "megaphone", "radio")
# style notundaki (VoxCPM2 oyunculuk talimatı) ipuçları
_HINTS = [
    ("megaphone", r"megaphone|bullhorn|loudspeaker|loud ?hailer|\bpa system|megafon|anons"),
    ("phone", r"on the phone|over the phone|through the phone|telephone|phone call|handset|telefon"),
    ("radio", r"\bradio\b|walkie|intercom|telsiz|interkom"),
]


def band_profile(x: np.ndarray, sr: int) -> dict:
    """Konuşma bandına (400-3000 Hz) göre alt/üst bant seviyeleri (dB, Hz başına) ve ölçülen bant kenarları."""
    from scipy.signal import welch

    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.size < 1024 or not np.any(x):
        return {}
    f, p = welch(x, sr, nperseg=min(2048, len(x)))

    def lv(a: float, b: float) -> float | None:
        m = (f >= a) & (f < b)
        return float(10 * np.log10(p[m].mean() + 1e-20)) if m.any() else None

    mid = lv(400, 3000)
    if mid is None:
        return {}
    # 1/3 oktav yumuşatılmış seviye (konuşma bandına göre)
    lsm = np.empty_like(p)
    for i, fi in enumerate(f):
        m = (f >= fi * 2 ** (-1 / 6)) & (f <= fi * 2 ** (1 / 6))
        lsm[i] = p[m].mean() if m.any() else p[i]
    lev = 10 * np.log10(lsm + 1e-20) - mid
    top = min(8000.0, sr / 2 * 0.95)
    f_hi = None
    for fi in f[(f >= 2500) & (f <= top)]:
        if np.max(lev[(f >= fi) & (f <= top)]) < -20:
            f_hi = float(fi)
            break
    f_lo = None
    for fi in f[(f >= 120) & (f <= 700)][::-1]:
        if np.max(lev[(f >= 70) & (f <= fi)]) < -12:
            f_lo = float(fi)
            break
    lo, hi = lv(60, 250), lv(4500, 6500)
    return {"lo_db": None if lo is None else round(lo - mid, 1), "hi_db": None if hi is None else round(hi - mid, 1),
            "f_lo": None if f_lo is None else round(f_lo), "f_hi": None if f_hi is None else round(f_hi)}


def hint_from_text(text: str | None) -> str | None:
    t = (text or "").lower()
    for kind, pat in _HINTS:
        if re.search(pat, t):
            return kind
    return None


def choose_kind(prof: dict, layer: str | None, explicit: str | None = None, style: str | None = None,
                auto: bool = True, phone_lo_db: float = -6.0, phone_hi_db: float = -26.0,
                hint_hi_db: float = -24.0) -> tuple[str | None, str]:
    """(tür, neden). explicit: "phone|megaphone|radio|none|auto" (translations.csv fx / fx.overrides)."""
    e = (explicit or "").strip().lower()
    if e in KINDS:
        return e, "elle"
    if e in ("none", "yok", "-"):
        return None, "elle"
    if not auto or not prof or prof.get("lo_db") is None or prof.get("hi_db") is None:
        return None, "-"
    # style notu türü söyler, ölçüm "gerçekten cihazdan mı duyuluyor"u: GTA'da "police radio dispatcher" notlu
    # replik telsize konuşan adamın kendisi (tam bant); efekt verilseydi yanlış olurdu
    h = hint_from_text(style)
    if h and prof["hi_db"] < hint_hi_db:
        return h, f"style notu + bant profili (üst {prof['hi_db']} dB)"
    # Telefon/cihaz hoparlörü: hem alçaklar (60-250 Hz) hem tizler (4,5-6,5 kHz) belirgin kesik.
    # Arka plan satırlarında ses tahmini kenetlemeyle çıkarıldığı için alt bant güvenilmez (müzik bastırıyor).
    if layer != "bg" and prof["lo_db"] < phone_lo_db and prof["hi_db"] < phone_hi_db:
        return "phone", f"bant profili (alt {prof['lo_db']} dB, üst {prof['hi_db']} dB)"
    return None, "-"


# --- yapı taşları -------------------------------------------------------------------------------------------

def bandpass(x: np.ndarray, sr: int, lo: float | None, hi: float | None, order: int = 4) -> np.ndarray:
    from scipy.signal import butter, sosfilt

    nyq = sr / 2
    if lo and hi and hi < nyq * 0.98:
        sos = butter(order, [lo, hi], "bandpass", fs=sr, output="sos")
    elif hi and hi < nyq * 0.98:
        sos = butter(order, hi, "lowpass", fs=sr, output="sos")
    elif lo:
        sos = butter(order, lo, "highpass", fs=sr, output="sos")
    else:
        return x
    return sosfilt(sos, x).astype(np.float32)


def drive(x: np.ndarray, gain_db: float) -> np.ndarray:
    """tanh yumuşak kırpma; giriş tepe 1'e normalize edilir, çıkış seviyesi sonra eşlenir."""
    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk <= 0:
        return x
    g = 10 ** (gain_db / 20)
    return (np.tanh(g * x / pk) / np.tanh(g)).astype(np.float32)


def compress(x: np.ndarray, sr: int, thr_db: float = -20.0, ratio: float = 4.0, win_s: float = 0.01,
             smooth_s: float = 0.006) -> np.ndarray:
    """RMS zarflı basit kompresör; eşik zarf tepesine göre (seviye bağımsız)."""
    from scipy.ndimage import uniform_filter1d

    if x.size == 0:
        return x
    env = np.sqrt(uniform_filter1d(x.astype(np.float64) ** 2, max(1, int(win_s * sr))) + 1e-12)
    lev = 20 * np.log10(env / (np.max(env) + 1e-12))
    gdb = np.minimum(0.0, (thr_db - lev) * (1 - 1 / ratio))
    gdb = uniform_filter1d(gdb, max(1, int(smooth_s * sr)))
    return (x * 10 ** (gdb / 20)).astype(np.float32)


def telephone_codec(x: np.ndarray, sr: int, rate: int | None = 8000, mu: float = 255.0) -> np.ndarray:
    """G.711 benzeri 8 bit μ-law nicemleme (codec pürüzü). rate=8000: önce 8 kHz'e inilir (4 kHz üstü tamamen
    gider); None: örnekleme hızı korunur, bant şekli önceki filtre/match_spectrum'a kalır."""
    from .audio import resample

    pk = float(np.max(np.abs(x))) if x.size else 0.0
    if pk <= 0:
        return x
    y = resample(x / pk * 0.9, sr, rate) if rate else (x / pk * 0.9)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    q = np.round(c * 127) / 127
    y = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    if rate:
        y = resample(y.astype(np.float32), rate, sr)
    n = len(x)
    y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
    return (y * pk / 0.9).astype(np.float32)


def echo(x: np.ndarray, sr: int, taps: list[tuple[float, float]], rt60: float = 0.8, wet_db: float = -13.0,
         band: tuple[float, float] = (300.0, 4000.0), seed: int = 0) -> np.ndarray:
    """Dış mekân: binalardan dönen ayrık yansımalar (taps: (gecikme sn, dB)) + üstel sönümlü dağınık kuyruk.
    Çıkış girişten uzundur (kuyruk)."""
    from scipy.signal import fftconvolve

    pre = int(0.02 * sr)
    n_tail = int(rt60 * sr)
    n = max([int(t * sr) for t, _ in taps] + [pre + n_tail]) + 1
    ir = np.zeros(n, dtype=np.float64)
    ir[0] = 1.0
    for t, g in taps:
        ir[int(t * sr)] += 10 ** (g / 20)
    if n_tail > 0 and wet_db > -90:
        rng = np.random.default_rng(seed)
        tail = rng.standard_normal(n_tail) * np.exp(-6.91 * np.arange(n_tail) / n_tail)
        tail = bandpass(tail.astype(np.float32), sr, band[0], band[1], order=2).astype(np.float64)
        tail *= 10 ** (wet_db / 20) / (np.sqrt(np.sum(tail ** 2)) + 1e-12)
        ir[pre: pre + n_tail] += tail
    return fftconvolve(x.astype(np.float64), ir).astype(np.float32)


# --- zincirler ----------------------------------------------------------------------------------------------

def apply(x: np.ndarray, sr: int, kind: str, prof: dict | None = None, ref: np.ndarray | None = None,
          ref_sr: int | None = None) -> np.ndarray:
    """Kanal zinciri; ref (orijinal replik) verilirse sonunda uzun dönem tınısı aktarılır. Çıkış seviyesi ham,
    loudness sonra eşlenir; megafonda yankı kuyruğu yüzünden çıkış girişten uzun olabilir."""
    from .audio import match_spectrum

    prof = prof or {}
    y = np.asarray(x, dtype=np.float32)
    if kind == "phone":
        lo = float(np.clip(prof.get("f_lo") or 300, 200, 500))
        hi = float(np.clip(prof.get("f_hi") or 3400, 2800, 3800))
        y = bandpass(y, sr, lo, hi)
        y = drive(y, 4.0)
        # gerçek 8 kHz hat 4 kHz üstünü sıfırlar; oyunlardaki telefon/cihaz sesi çoğu zaman daha yumuşak eğimli
        # (GTA "Surprise!": 4,5-6,5 kHz konuşma bandının ~35 dB altında, sıfır değil) → orada hız korunur
        narrow = prof.get("hi_db") is not None and float(prof["hi_db"]) < -55
        y = telephone_codec(y, sr, rate=8000 if narrow else None)
        y = compress(y, sr, thr_db=-22, ratio=3)
        y = bandpass(y, sr, lo, hi)
    elif kind == "megaphone":
        lo = float(np.clip(prof.get("f_lo") or 450, 300, 800))
        hi = float(np.clip(prof.get("f_hi") or 4200, 2500, 5000))
        y = bandpass(y, sr, lo, hi, order=2)
        y = drive(y, 14.0)  # aşırı sürülmüş küçük hoparlör
        y = bandpass(y, sr, lo, hi)
        y = compress(y, sr, thr_db=-18, ratio=4)
        y = echo(y, sr, taps=[(0.14, -9.0), (0.29, -15.0)], rt60=0.8, wet_db=-13.0)
    elif kind == "radio":
        lo = float(np.clip(prof.get("f_lo") or 350, 250, 700))
        hi = float(np.clip(prof.get("f_hi") or 3000, 2400, 3800))
        y = bandpass(y, sr, lo, hi)
        y = drive(y, 18.0)
        y = compress(y, sr, thr_db=-16, ratio=6)
        y = bandpass(y, sr, lo, hi)
        rng = np.random.default_rng(1)
        hiss = bandpass(rng.standard_normal(len(y)).astype(np.float32), sr, 1000, hi, order=2)
        y = y + hiss * (np.sqrt(np.mean(y ** 2)) / (np.sqrt(np.mean(hiss ** 2)) + 1e-12)) * 10 ** (-30 / 20)
    else:
        raise ValueError(f"bilinmeyen kanal efekti: {kind}")
    if ref is not None and ref_sr and ref.size >= 1024:
        y = match_spectrum(y, sr, ref, ref_sr)
    # seviyeyi girişin RMS'ine getir (zincir 30-40 dB düşürebiliyor; fit'in ±30 dB'lik loudness eşlemesi yetmezdi)
    n = min(len(x), len(y))
    rin, rout = float(np.sqrt(np.mean(np.asarray(x[:n], np.float64) ** 2))), float(np.sqrt(np.mean(y[:n].astype(np.float64) ** 2)))
    if rin > 0 and rout > 0:
        y = y * (rin / rout)
    f = min(int(0.005 * sr), len(y) // 4)  # STFT/kırpma kenarında tık olmasın
    if f > 1:
        ramp = np.linspace(0.0, 1.0, f, dtype=np.float32)
        y[:f] *= ramp
        y[-f:] *= ramp[::-1]
    return y.astype(np.float32)
