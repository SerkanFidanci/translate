# Yeni PC'de kurulum ve kaldığın yerden devam

Önce [YAPTIKLARIMIZ.md](YAPTIKLARIMIZ.md)'yi oku (ne yaptık, neden). Bu dosya komutları sırayla verir.
Komutlar PowerShell içindir ve repo kökünden (`translate/`) çalıştırılır.

## 1. Gereksinimler

- Windows 11, NVIDIA GPU (en az 6 GB; VoxCPM2 en çok 5,6 GB kullanıyor), güncel sürücü
- git, [uv](https://docs.astral.sh/uv/) (`winget install astral-sh.uv`). Python 3.11'i uv kendisi indirir.
- Disk: ortamlar + modeller ~25-30 GB
- İsteğe bağlı: [Ollama](https://ollama.com) (`ollama pull qwen3.5:9b`, sadece API anahtarı yoksa yedek çeviri)

## 2. Kodu al ve ortamları kur

```powershell
git clone https://github.com/SerkanFidanci/translate.git
```

```powershell
cd translate
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1
```

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1 -Only voxcpm
```

- İlk komut core, dnr, sep, diar, emo, tts, vc, omni ortamlarını kurar; MSST model kodunu (`tools/msst`) klonlar ve
  Bandit v2 DnR ağırlığını (447 MB, Zenodo) indirir. İkincisi kullandığımız tek TTS olan VoxCPM2'yi kurar.
- **RTX 50 serisi (5060 Ti vb.)** için iki komuta da `-Torch cu128` ekle (CUDA 12.6 yapıları bu kartları
  desteklemiyor). XTTS ortamı torch 2.6'ya sabit; RTX 50'de çalışmayabilir, zaten kullanmıyoruz.
- Model ağırlıkları (Whisper large-v3, emotion2vec, VoxCPM2) ilk çalıştırmada Hugging Face'ten iner.

Ek araçlar (bu makinede elle eklenmişti):

```powershell
uv pip install --python envs\core\Scripts\python.exe yt-dlp
```

```powershell
New-Item -ItemType Directory -Force tools\bin; envs\core\Scripts\python.exe -c "import imageio_ffmpeg, shutil; shutil.copy(imageio_ffmpeg.get_ffmpeg_exe(), r'tools\bin\ffmpeg.exe')"
```

(Video birleştirme betikleri `tools/bin/ffmpeg.exe` bekliyor.)

Kontrol:

```powershell
envs\core\Scripts\dubpoc.exe doctor
```

```powershell
envs\core\Scripts\python.exe -m pytest -q
```

(44 test geçmeli.)

## 3. Veriler git'te yok: iki yol

Ses/video ve ara çıktılar (`input/`, `downloads/`, `work/`, `out/`) kasıtlı olarak git'e konmadı (telifli oyun sesi,
sanatçı sesini taklit eden dublajlar; repo herkese açık). Git'te olanlar: kod, ayarlar, `speakers.yaml`,
`subtitles.csv`, `translations.csv` (elle yapılan konuşmacı ataması ve çeviriler).

### 3a. Önerilen: iş klasörlerini bu PC'den taşı (birebir aynı durum)

Bu PC'de (`Desktop\ttest\dubpoc`) şu klasörleri zipleyip USB/OneDrive ile taşı (GitHub'a değil):

| Klasör | Boyut | Ne |
|---|---|---|
| `work`, `input`, `downloads` | ~1,35 GB | GTA projesi (segmentler, ses bankaları, TTS adayları) |
| `projects\007_first_light\work`, `input`, `downloads` | ~340 MB | 007 fragmanı |
| `out` | ~1,4 GB | GTA miksleri ve videoları (isteğe bağlı; `mix` ile yeniden üretilir) |

```powershell
Compress-Archive -Path work, input, downloads -DestinationPath $HOME\Desktop\dubpoc_gta_veri.zip
```

```powershell
Compress-Archive -Path projects\007_first_light\work, projects\007_first_light\input, projects\007_first_light\downloads -DestinationPath $HOME\Desktop\dubpoc_007_veri.zip
```

Ev PC'sinde repo köküne aç (`Expand-Archive ... -DestinationPath .` ve 007 için `-DestinationPath projects\007_first_light`).
Manifest'teki yollar göreli olduğu için klasörün yeri değişse de çalışır.

### 3b. Sıfırdan (veri taşımadan)

```powershell
envs\core\Scripts\yt-dlp.exe -f "bv*[height<=1080][ext=mp4]+ba[ext=m4a]/b[height<=1080]" --merge-output-format mp4 --ffmpeg-location tools\bin -o "projects\007_first_light\downloads\007_first_light_trailer.%(ext)s" "https://www.youtube.com/watch?v=J4qY9DYE184"
```

```powershell
New-Item -ItemType Directory -Force projects\007_first_light\input; tools\bin\ffmpeg.exe -i projects\007_first_light\downloads\007_first_light_trailer.mp4 -vn -ac 2 -ar 48000 -c:a pcm_s16le projects\007_first_light\input\007_trailer.wav
```

GTA için aynısı: `https://www.youtube.com/watch?v=DYDz_cIgoAw` → `downloads\yt_DYDz_cIgoAw\gta6_ending_tam_video.mp4`
ve `input\gta6_ending_tam.wav`. **Dikkat:** GTA satırları eski kodla segmentlenip sonradan elle eklenen satırlarla
(`s44b`, `s55b`, `b09` ...) büyüdü; sıfırdan çalıştırınca `seg_id`'ler kayar ve `translations.csv` yanlış satırlara
düşer. GTA'yı sürdürmek için 3a yolunu kullan.

## 4. 007 fragmanı (bu PC'de bitti; evde yeniden üretmek ya da düzeltmek için)

`workers/` klasörü bir bağlantı (junction), git'e girmez; bir kez oluştur:

```powershell
New-Item -ItemType Junction -Path projects\007_first_light\workers -Target (Resolve-Path workers)
```

**3a ile taşıdıysan**, kaldığın aşamadan devam (durumu gör: `envs\core\Scripts\dubpoc.exe --root projects\007_first_light status`):

```powershell
envs\core\Scripts\dubpoc.exe --root projects\007_first_light run --from speakers --to qa
```

```powershell
envs\core\Scripts\dubpoc.exe --root projects\007_first_light run --from tts --to mix --force
```

İlki ilk TTS turunu (normal adaylar) ve QA'yı bitirir; ikincisi QA'dan geçen Türkçe okumalardan **istemli**
adaylar (`vt1/vt2`) ekler (eski adaylar yeniden üretilmez, `tts.reuse_existing`), yeniden seçer ve miksler.

**3b ile sıfırdan başladıysan:**

```powershell
envs\core\Scripts\dubpoc.exe --root projects\007_first_light run --to asr
```

```powershell
envs\core\Scripts\python.exe projects\007_first_light\fix_segments.py
```

Sonra yukarıdaki iki komut (ilkinde `--force` ekle). `fix_segments.py` segmentleme farklı çıkarsa durur; o zaman
satır zamanlarını kontrol etmek gerekir.

Video (1080p iki ses izli + telefona tek parça 720p):

```powershell
& "C:\Program Files\Gitinash.exe" projects/007_first_light/mux_video.sh v1
```

Çıktı: `projects\007_first_light\out\video\`.

## 5. GTA videosu

Ayarlar `config.local.yaml` (doğal mod, sadece VoxCPM2). Miksi yeniden üret ve videoyu çıkar:

```powershell
envs\core\Scripts\dubpoc.exe run --from mix --force
```

```powershell
& "C:\Program Files\Gitinash.exe" scripts/mux_gta_video.sh efekt2
```

## 6. Yeni bir video dublajlamak (tarif)

1. Proje klasörü: `projects\<ad>\` içine `config.yaml` (kökten kopya), `projects\007_first_light\config.local.yaml`
   kopyası (ilk satırlardaki yorumu düzelt), `input\` ve `workers` junction'ı.
2. Sesi `input\` altına WAV olarak koy (yukarıdaki yt-dlp + ffmpeg komutları).
3. `dubpoc --root projects\<ad> run --to asr` → `work\manifest.jsonl`'daki satırlara bak.
4. Konuşmacılar: HF token yoksa kümeleme kaba çıkar. Satır ortalarından kareler + ses perdesi (kadın ~170-350 Hz,
   erkek ~100-150 Hz) + metinle karakterleri belirle → `speakers.yaml` → `overrides: {seg_id: Karakter}`.
   İki kişinin karıştığı satırları `dialog_row` ile böl (örnek: `projects\007_first_light\fix_segments.py`).
5. Yanlış yazılan İngilizceyi `subtitles.csv` (`id,text_en`; `-` = orijinal kalsın) ile düzelt.
6. Çeviri: `ANTHROPIC_API_KEY` varsa `config.local.yaml`'daki `translate:` bölümünü sil (Claude çevirir); yoksa
   `translations.csv` (`seg_id,text_tr,style,fx,...`) elle/Claude ile yazılır. `style` = İngilizce kısa oyunculuk
   notu; `fx` = `phone|megaphone|radio|none` (boşsa otomatik).
7. `run --from speakers --to qa --force`, sonra `run --from tts --to mix --force` (iki TTS turu).
8. Dinle (`dubpoc serve` → rapor), yanlışları düzelt, `--only seg_id --force` ile sadece o satırları yenile.

## 7. Claude Code ile devam

Bu PC'deki asistan hafızası ev PC'sine taşınmaz. Yeni oturumda ilk mesajda şunu söylemen yeterli:
"`docs/YAPTIKLARIMIZ.md` ve `docs/KURULUM_VE_DEVAM.md`'yi oku, 007 fragmanından devam et."
