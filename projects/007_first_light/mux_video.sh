#!/usr/bin/env bash
# 007 First Light fragmanı: Türkçe dublajı videoya koy (1080p kopya, 2 ses izi) + telefon için tek parça 720p (<30 MB)
set -euo pipefail
cd "$(dirname "$0")"
V=downloads/007_first_light_trailer.mp4
FF=../../tools/bin/ffmpeg.exe
TAG=${1:-v1}
mkdir -p out/video
OUT=out/video/007_first_light_tr_${TAG}_1080p.mp4
$FF -hide_banner -loglevel error -y -i $V -i out/A/f001.wav -i input/007_trailer.wav \
  -map 0:v:0 -map 1:a:0 -map 2:a:0 -c:v copy -c:a aac -b:a 192k \
  -metadata:s:a:0 language=tur -metadata:s:a:0 title="Turkce dublaj (VoxCPM2, ${TAG})" \
  -metadata:s:a:1 language=eng -metadata:s:a:1 title="Orijinal (EN)" \
  -disposition:a:0 default -disposition:a:1 0 -shortest -movflags +faststart $OUT
$FF -hide_banner -loglevel error -y -i $OUT -map 0:v:0 -map 0:a:0 -vf "scale=-2:720,fps=30" \
  -c:v libx264 -preset medium -b:v 900k -maxrate 1200k -bufsize 2400k -c:a aac -b:a 112k -movflags +faststart \
  out/video/007_first_light_tr_${TAG}_720p.mp4
ls -la out/video/ | grep "$TAG"
