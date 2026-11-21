#!/usr/bin/env bash
# dubpoc ortam kurulumu (Linux / macOS-benzeri)
#   bash scripts/setup_envs.sh            # hepsi
#   ONLY="core tts" bash scripts/setup_envs.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ONLY="${ONLY:-core dnr sep diar emo tts vc omni}"
TORCH="${TORCH:-cu126}"
SEED_VC_COMMIT="${SEED_VC_COMMIT:-51383efd921027683c89e5348211d93ff12ac2a8}"
MSST_COMMIT="${MSST_COMMIT:-1f700c386e0dd763fe0a57242bb9420dce987354}"

command -v uv >/dev/null || { echo "uv bulunamadı: https://docs.astral.sh/uv/"; exit 1; }

for name in $ONLY; do
  echo "==> [$name] venv"
  py="envs/$name/bin/python"
  [ -x "$py" ] || uv venv "envs/$name" --python 3.11 --clear
  case "$name" in
    core) uv pip install --python "$py" -e ".[asr,dev]" ;;
    dnr)
      [ -d tools/msst/.git ] || git clone https://github.com/ZFTurbo/Music-Source-Separation-Training.git tools/msst
      git -C tools/msst fetch --depth 1 origin "$MSST_COMMIT" || true
      git -C tools/msst checkout -q "$MSST_COMMIT" || true
      uv pip install --python "$py" --torch-backend "$TORCH" -r requirements/dnr.txt
      ck=tools/models/bandit_v2/checkpoint-multi.ckpt
      [ -f "$ck" ] || { mkdir -p tools/models/bandit_v2; curl -L --retry 3 -o "$ck" "https://zenodo.org/records/12701995/files/checkpoint-multi.ckpt?download=1"; }
      echo "fea2868787551b0cff36cfcf7c3622a3  $ck" | md5sum -c - ;;
    vc)
      [ -d tools/seed-vc/.git ] || git clone https://github.com/Plachtaa/seed-vc.git tools/seed-vc
      git -C tools/seed-vc fetch --depth 1 origin "$SEED_VC_COMMIT" || true
      git -C tools/seed-vc checkout -q "$SEED_VC_COMMIT" || true
      uv pip install --python "$py" --torch-backend "$TORCH" -r requirements/vc.txt ;;
    *) uv pip install --python "$py" --torch-backend "$TORCH" -r "requirements/$name.txt" ;;
  esac
  echo "==> [$name] tamam"
done

command -v ffmpeg >/dev/null || echo "UYARI: ffmpeg PATH'te yok (apt install ffmpeg rubberband-cli)"
echo "Kontrol: envs/core/bin/dubpoc doctor"
