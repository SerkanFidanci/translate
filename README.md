# dubpoc — Oyun Türkçe AI Dublaj POC

İngilizce oyun repliklerini Türkçeye dublajlayan uçtan uca deneme hattı: ayrıştırma (Bandit v2 DnR:
konuşma / müzik / efekt) → konuşmacı →
ASR → duygu → hece bütçeli çeviri → TTS (karakter sesiyle) → VC → süre/ses eşleme → QA → miks → HTML rapor.
Spec: [docs/SPEC_oyun-tr-dublaj-poc.md](docs/SPEC_oyun-tr-dublaj-poc.md). Araştırma ve spec'e göre iyileştirmeler:
[docs/ARASTIRMA.md](docs/ARASTIRMA.md).

**Başka bir PC'de devam:** önce [docs/YAPTIKLARIMIZ.md](docs/YAPTIKLARIMIZ.md) (ne yaptık, nasıl), sonra
[docs/KURULUM_VE_DEVAM.md](docs/KURULUM_VE_DEVAM.md) (kurulum, verileri taşıma, 007 fragmanını bitirme).
Projeler: kök klasör = GTA 6 videosu; `projects/007_first_light/` = 007 First Light fragmanı (`--root` ile).

> Kişisel deneme amaçlıdır. Orijinal seslendirme sanatçılarının sesine benzetilmiş çıktılar dağıtılmamalıdır.
> Chatterbox çıktıları Perth filigranı içerir; OmniVoice ağırlıkları ticari olmayan kullanım (CC-BY-NC) içindir.

## Kurulum (Windows 11, NVIDIA GPU)

Gereksinimler: [uv](https://docs.astral.sh/uv/), git, NVIDIA sürücüsü. Python 3.11'i uv kendisi indirir.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1
```

Her ağır model kendi venv'ine kurulur (`envs/core, dnr, sep, diar, emo, tts, vc, omni`; toplam ~20 GB, model
ağırlıkları ilk çalıştırmada indirilir). `dnr` kurulumu ayrıca MSST model kodunu (`tools/msst`) klonlar ve
Bandit v2 DnR ağırlığını (Zenodo 12701995 `checkpoint-multi.ckpt`, 447 MB, CC BY-SA 4.0, MD5 doğrulamalı) indirir.

İsteğe bağlı üçüncü TTS: **XTTS-v2** (`-Only xtts`). Ağırlıkları Coqui Public Model License (ticari olmayan)
altında; ancak `tts.xtts.tos_agreed: true` ile (lisansı kabul ettiğinizde) indirilir. Etkinleştirmek için
`tts.backends` listesine `xtts` ekleyin.

İsteğe bağlı dördüncü TTS: **VoxCPM2** (`-Only voxcpm`, Apache-2.0, 6 GB kartta sınırda). Duygu talimatı
`translations.csv`'deki `style` sütunundan (İngilizce kısa not, örn. `shocked and alarmed, urgent`) alınır. Tek bir ortamı yeniden kurmak için: `-Only tts`.
Linux: `bash scripts/setup_envs.sh`.

- **ffmpeg**: sistemde yoksa `imageio-ffmpeg`'in paketlediği build (`tools/bin/ffmpeg.exe`) otomatik kullanılır.
- **vgmstream-cli**: sadece `.wem` girdisi için; `tools/vgmstream-cli.exe` olarak koyun.
- **HF_TOKEN**: çok konuşmacılı dosyalar için. HF'de `pyannote/speaker-diarization-community-1` koşullarını
  kabul edin. Yoksa duraklama tabanlı yedek bölme kullanılır.
- **ANTHROPIC_API_KEY**: çeviri için (varsayılan `claude-sonnet-5-5`). Yoksa `config.local.yaml` ile yerel
  Ollama kullanılabilir (bu makinede `qwen3.5:9b` ayarlı; anahtarı ayarlayınca dosyayı silin).

Kontrol:

```powershell
envs\core\Scripts\dubpoc.exe doctor
```

## Çalıştırma

1. Test dosyalarını `input/` altına koyun. `input/<karakter>/dosya.wav` yapısında klasör adı karakter olur
   (kümeleme atlanır). Kök dizindeki dosyalar (ara sahneler) kümelenir ya da klasör karakterlerine eşlenir.
   Sentetik deneme seti için: `envs\core\Scripts\python.exe scripts\make_test_set.py`
2. Hepsini çalıştır:

```powershell
envs\core\Scripts\dubpoc.exe run
```

3. Raporu dinlemek için `envs\core\Scripts\dubpoc.exe serve` çalıştırın: tarayıcıda
   `http://localhost:8765/out/report.html` açılır (sesler göreli yolla `work/` ve `out/` altından çalar).
   `out\report.html`'e çift tıklamak Chrome/Edge'de de çalışır; Firefox üst klasördeki sesleri engelleyebilir.

Diğer komutlar:

```
dubpoc doctor                          # ortam/binary/GPU/HF token/API key kontrolü
dubpoc ingest | separate | diarize | speakers | asr | emotion | translate | tts | vc | fit | qa | mix | report
dubpoc run [--from STAGE] [--to STAGE] [--only SEG_ID] [--force] [--path A|B|both]
dubpoc status [-s]                     # manifest özeti (-s: segment tablosu)
dubpoc clean [-y]                      # work/, out/, speakers.yaml sil (input/ ve record/ korunur)
dubpoc serve [--port 8765]             # raporu yerel sunucudan tarayıcıda aç
```

- Her aşama sadece bekleyen satırları işler; `--force` hepsini işler. Bir aşama bir satırı yeniden
  işlediğinde o satırın sonraki aşamaları otomatik "pending" olur.
- `speakers` sonrası `speakers.yaml`'daki küme adlarını doldurup `dubpoc run --from speakers` çalıştırın.
- **Yol B (insan):** `out/script.md` okuma senaryosudur. Kayıtları `record/<seg_id>.wav` olarak koyup
  `dubpoc run --from vc --path B` çalıştırın. Kaydı olmayan segmentlerde B miksi Yol A'yı kullanır.
- `subtitles.csv` (`id,text_en`; id = seg_id, file_id ya da dosya adı) varsa ASR yerine kullanılır.
- `translations.csv` (`seg_id,text_tr`) varsa o satırlar LLM yerine kullanılır (insan/elle çeviri).
- `config.local.yaml` (varsa) `config.yaml`'ın üstüne birleştirilir: makineye özel ayarlar için.
- **Arka plan konuşması** (megafon/telsiz anonsu, uzaktan bağırma): `diarize` müzik+efekt kanalını Whisper'la
  tarayıp bunları `<file>_bNN` satırları olarak ekler (`diarization.bg_speech`); dublajlanır, orijinali mix'te
  arka plandan bastırılır. Bu satırların sesi gürültülü olduğu için `speakers.yaml` overrides ile bir karaktere
  bağlamak daha temiz ses verir. Diyalogun arka plana sızan kopyaları (`bg_leak_of`) dublajlanmaz, sadece bastırılır;
  sızıntıda diyalogda olmayan baş/son kelimeler ("Two, one, **jump**") ayrı bir bg satırı olur.
- **Kanal efekti** (telefon, megafon, telsiz): `fit` orijinal replik bir cihazdan duyuluyorsa dublaja aynı zinciri
  uygular (`fx` modülü, `fx:` ayarları). Telefon/cihaz hoparlörü bant profilinden otomatik bulunur; megafon ve
  telsiz için `translations.csv`'ye `fx` sütunu ekleyin (`phone|megaphone|radio|none`) ya da `fx.overrides`
  kullanın. QA, efektsiz kopyayı (`fit_dry`) ölçer.

## Klasör yapısı

```
config.yaml  config.local.yaml  speakers.yaml  subtitles.csv
input/   record/   work/ (manifest.jsonl, ingest, sep, seg, emb, bank/<char>, tts, vc, fit, qa, logs, jobs)
out/A/<file_id>.wav  out/B/<file_id>.wav  out/report.html  out/script.md
src/dubpoc/  (orchestrator, typer CLI)   workers/ (her venv için worker)   scripts/   tests/   tools/
```

Worker sözleşmesi: `python workers/<x>.py in.json out.json`; stdout sadece log. Worker logları
`work/logs/<aşama>.log`. Worker hataları segmentin `errors` alanına yazılır, pipeline devam eder.
Aynı anda tek GPU worker çalışır (`work/.gpu.lock`).

## Testler

```powershell
envs\core\Scripts\python.exe -m pytest -q
```

GPU gerektirmeyen birim testleri + sentetik fit→mix→report uçtan uca testi.

## Milestone durumu

| # | İçerik | Durum |
|---|---|---|
| M1 | iskelet, config, manifest, doctor, ingest, asr, translate, tts, script.md | ✅ |
| M2 | speakers (klasör + kümeleme + bank), vc, fit, qa, aday seçimi, report | ✅ |
| M3 | separate (Bandit v2 DnR / RoFormer), diarize, mix | ✅ (diarize HF token olmadan yedek modda test edildi) |
| M4 | Yol B (record → vc → fit → qa → mix), A/B rapor | ✅ |

## Şu anki durum (2026-11-20)

- `input/` içinde YouTube'dan indirilen **GTA 6 "Good Ending" videosunun tamamı (8:54)** (kişisel test).
  1 dakikalık önceki çalışma `samples/gta_ilk60/`, sentetik set `samples/sentetik/` altında.
- Tam video: 65 konuşma parçası; 42 diyalog repliği VoxCPM2 ile dublajlandı (fail 0), müzik/aksiyon/çığlık
  parçaları orijinal kaldı. Konuşmacılar `speakers.yaml → overrides` ile (perde + bağlam), çeviri ve yönetmen
  notları `translations.csv`'de, ASR düzeltmeleri `subtitles.csv`'de. Video: `out/video/gta6_tam_tr_dublaj_*.mp4`.
- Sentetik setteki son koşu (Bandit v2 DnR ile): QA ok=8 / review=6 / fail=0; dublajlanan 14 replikte WER 0. Tek karakterli
  7 replik 7/7 ok (konuşmacı benzerliği oranı 0,85–1,09). Müzikli sahne HF token olmadan 7 parçaya bölündü,
  7/7 doğru karaktere atandı; review'lar kısa (<1,2 sn) parçaların benzerlik ölçümünden. Ayrıştırma ölçümü
  ve diğer bulgular: docs/ARASTIRMA.md §2.
- `config.local.yaml` çeviriyi yerel Ollama'ya yönlendiriyor (API anahtarı yok). `ANTHROPIC_API_KEY` ayarlayıp
  dosyayı silince Claude Sonnet 5.5 kullanılır; `HF_TOKEN` ile gerçek pyannote diarization açılır.

## Bilinen sorunlar

Bkz. [docs/BILINEN_SORUNLAR.md](docs/BILINEN_SORUNLAR.md).
