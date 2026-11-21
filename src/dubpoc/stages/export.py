"""Oyun dosyalarına geri paketleme — POC KAPSAMI DIŞI (spec 9).

TODO:
- out/A|B/<file_id>.wav → Wwise .wem dönüştürme (WwiseConsole.exe convert-external-source, Vorbis
  kalite ayarı orijinal .wem'den okunmalı).
- .bnk / .pck içine yeniden paketleme (ör. bnkextr / wwiser ile ID eşleme) ve orijinal
  dosya adlarına/ID'lere göre yerleştirme.
- Orijinal loudness/SR/kanal ve döngü (loop) metadatası korunmalı.
"""
from __future__ import annotations

from . import Ctx, console


def run(ctx: Ctx) -> None:  # pragma: no cover - stub
    console.print("[yellow]export: POC kapsamı dışında (Wwise .wem/.bnk paketleme). "
                  "Çıktılar out/A ve out/B altında .wav olarak duruyor.[/]")
