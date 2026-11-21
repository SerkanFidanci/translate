#!/usr/bin/env bash
# Tam videoya yeni Türkçe dublajı koy (1080p kopya) + telefona 2 parça 720p (<30 MB)
set -euo pipefail
cd "$(dirname "$0")/.."
V=downloads/yt_DYDz_cIgoAw/gta6_ending_tam_video.mp4
FF=tools/bin/ffmpeg.exe
TAG=${1:-v2}
OUT=out/video/gta6_tam_tr_${TAG}_1080p.mp4
$FF -hide_banner -loglevel error -y -i $V -i out/A/f001.wav -i input/gta6_ending_tam.wav \
  -map 0:v:0 -map 1:a:0 -map 2:a:0 -c:v copy -c:a aac -b:a 192k \
  -metadata:s:a:0 language=tur -metadata:s:a:0 title="Turkce dublaj (VoxCPM2, ${TAG})" \
  -metadata:s:a:1 language=eng -metadata:s:a:1 title="Orijinal (EN)" \
  -disposition:a:0 default -disposition:a:1 0 -shortest -movflags +faststart $OUT
$FF -hide_banner -loglevel error -y -ss 0 -t 200 -i $OUT -map 0:v:0 -map 0:a:0 -vf "scale=-2:720,fps=30" \
  -c:v libx264 -preset medium -b:v 650k -maxrate 850k -bufsize 1700k -c:a aac -b:a 96k -movflags +faststart \
  out/video/gta6_tam_tr_${TAG}_parca1_0000-0320.mp4
$FF -hide_banner -loglevel error -y -ss 200 -i $OUT -map 0:v:0 -map 0:a:0 -vf "scale=-2:720,fps=30" \
  -c:v libx264 -preset medium -b:v 480k -maxrate 650k -bufsize 1300k -c:a aac -b:a 96k -movflags +faststart \
  out/video/gta6_tam_tr_${TAG}_parca2_0320-0854.mp4
ls -la out/video/ | grep "$TAG"
