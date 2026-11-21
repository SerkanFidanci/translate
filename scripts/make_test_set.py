"""Sentetik test seti üretir (sadece altyapıyı denemek için; gerçek oyun sesi yerine geçmez).

Windows'un yerleşik SAPI sesiyle İngilizce replikler sentezlenir, ikinci karakter perde kaydırmayla
türetilir, bir "ara sahne" dosyasına müzik + efekt yatağı eklenir.

    envs\\core\\Scripts\\python.exe scripts\\make_test_set.py            # input/ altına yazar
    envs\\core\\Scripts\\python.exe scripts\\make_test_set.py --out X    # başka klasöre

Çıktı: input/kaptan/*.wav, input/ajan/*.wav (klasör = karakter), input/sahne_01.wav (müzikli, 2 konuşmacı)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

SR = 48000

LINES = {
    "kaptan": [
        ("emir", "We move at dawn. Nobody leaves this camp without my order.", {}),
        ("kacis", "Get down! They're right behind us! Move, move!", {"rate": "+25%", "pitch": "+15%", "volume": "loud"}),
        ("kayip", "I lost good men out there today... too many.", {"rate": "-25%", "pitch": "-10%", "volume": "soft"}),
        ("hmm", "Hmm.", {}),
    ],
    "ajan": [
        ("kasa", "The vault code is hidden somewhere in the old lighthouse.", {}),
        ("buldum", "This is incredible! We actually found it!", {"rate": "+15%", "pitch": "+20%", "volume": "loud"}),
        ("sessiz", "Stay quiet and follow me. The guards change shifts in two minutes.", {"rate": "-10%", "volume": "soft"}),
        ("berlin", "I don't trust him. Not after what happened in Berlin.", {}),
    ],
}
SCENE = [("kaptan", "Where is the package?", {}), ("ajan", "Safe. For now. But we need to move, fast.", {}),
         ("kaptan", "Then let's go. Stay close to me.", {})]
PITCH = {"kaptan": -5.0, "ajan": 0.0}  # tek SAPI sesinden iki karakter


def synth_all(jobs: list[dict], tmp: Path) -> None:
    ps = r"""
Add-Type -AssemblyName System.Speech
$jobs = Get-Content -Raw -Encoding UTF8 '%s' | ConvertFrom-Json
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
foreach ($j in $jobs) {
  $fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(48000, [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen, [System.Speech.AudioFormat.AudioChannel]::Mono)
  $s.SetOutputToWaveFile($j.path, $fmt)
  $s.SpeakSsml($j.ssml)
  $s.SetOutputToNull()
}
$s.Dispose()
"""
    jf = tmp / "jobs.json"
    jf.write_text(json.dumps(jobs), encoding="utf-8")
    script = tmp / "synth.ps1"
    script.write_text(ps % jf, encoding="utf-8-sig")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script)], check=True)


def ssml(text: str, prosody: dict) -> str:
    attrs = " ".join(f'{k}="{v}"' for k, v in prosody.items())
    body = f"<prosody {attrs}>{text}</prosody>" if attrs else text
    return ('<speak version="1.0" xmlns="http://www.w3.org/2001/10/synthesis" xml:lang="en-US">'
            f"{body}</speak>")


def load(path: Path, char: str) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32")
    if sr != SR:
        x = soxr.resample(x, sr, SR)
    if PITCH[char]:
        from pedalboard import PitchShift
        x = PitchShift(semitones=PITCH[char])(x[None, :].astype(np.float32), SR)[0]
    return x.astype(np.float32)


def stereo(x: np.ndarray, pad_s: float = 0.25) -> np.ndarray:
    pad = np.zeros(int(pad_s * SR), np.float32)
    x = np.concatenate([pad, x, pad])
    return np.stack([x, x * 0.97], axis=1)


def music_bed(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n) / SR
    chords = [(110, 138.6, 164.8), (98, 123.5, 146.8), (87.3, 110, 130.8), (98, 123.5, 146.8)]
    out = np.zeros(n, np.float32)
    seg = int(2.0 * SR)
    for i in range(0, n, seg):
        f = chords[(i // seg) % len(chords)]
        tt = t[i: i + seg]
        env = np.minimum(1, np.minimum((tt - tt[0]) * 4, (tt[-1] - tt + 1e-3) * 4))
        out[i: i + seg] = env * sum(np.sin(2 * np.pi * fr * tt) for fr in f) / 3
    # "efekt": iki patlama benzeri gürültü darbesi
    for at in (0.3, 0.7):
        a = int(at * n)
        k = int(0.4 * SR)
        burst = rng.normal(size=k) * np.exp(-np.linspace(0, 8, k))
        out[a: a + k] += burst[: max(0, min(k, n - a))] * 0.8
    return np.stack([out, np.roll(out, 240)], axis=1).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "input"))
    args = ap.parse_args()
    if sys.platform != "win32":
        sys.exit("Bu üretici Windows SAPI sesini kullanır.")
    out = Path(args.out)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        jobs = []
        for ch, lines in LINES.items():
            for name, text, pros in lines:
                jobs.append({"path": str(tmp / f"{ch}_{name}.wav"), "ssml": ssml(text, pros)})
        for i, (ch, text, pros) in enumerate(SCENE):
            jobs.append({"path": str(tmp / f"scene_{i}_{ch}.wav"), "ssml": ssml(text, pros)})
        synth_all(jobs, tmp)
        for ch, lines in LINES.items():
            (out / ch).mkdir(parents=True, exist_ok=True)
            for name, _, _ in lines:
                x = stereo(load(tmp / f"{ch}_{name}.wav", ch))
                sf.write(str(out / ch / f"{name}.wav"), x * 0.8, SR, subtype="PCM_16")
        parts = []
        for i, (ch, _, _) in enumerate(SCENE):
            parts += [load(tmp / f"scene_{i}_{ch}.wav", ch), np.zeros(int(0.6 * SR), np.float32)]
        dia = stereo(np.concatenate(parts), pad_s=0.8)
        bed = music_bed(len(dia))
        mix = dia * 0.8 + bed * (10 ** (-16 / 20))
        mix /= max(1.0, float(np.max(np.abs(mix))) / 0.95)
        sf.write(str(out / "sahne_01.wav"), mix, SR, subtype="PCM_16")
    n = sum(1 for _ in out.rglob("*.wav"))
    print(f"{n} test dosyası yazıldı -> {out}")


if __name__ == "__main__":
    main()
