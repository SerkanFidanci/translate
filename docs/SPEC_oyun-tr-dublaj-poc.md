# Oyun Türkçe AI Dublaj — POC Spec

> Bu doküman bir kodlama ajanına (Claude Code / Fable) verilmek üzere yazıldı. Ajan bu spec'e göre projeyi sıfırdan kurar.
> Kapsam: **birkaç test ses dosyası** ile uçtan uca deneme. Oyun dosyalarına geri paketleme (Wwise/.wem repack) bu POC'nin dışında.

---

## 0. Amaç

Oyundan alınmış birkaç ses dosyasını alıp:

1. Konuşmayı arka plandan (müzik/efekt) ayırmak
2. Konuşmacıları ayırıp her konuşmacıyı bir karaktere eşlemek
3. İngilizce konuşmayı yazıya dökmek ve duygusunu ölçmek
4. Duygu ve süre bilgisini koruyarak Türkçeye çevirmek
5. Türkçe sesi, **karakterin kendi ses tonuyla** ve **aynı heyecanla** üretmek
6. Süreyi ve ses seviyesini orijinale eşlemek, arka planla tekrar birleştirmek
7. Otomatik kalite ölçümü yapıp sonuçları dinlenebilir bir HTML raporunda göstermek

İki üretim yolu yan yana karşılaştırılacak:

- **Yol A (TTS):** Türkçe metin → Chatterbox Multilingual TTS → Seed-VC (karakter tonu)
- **Yol B (İnsan):** Kullanıcı Türkçe repliği kendisi okur → Seed-VC (karakter tonu)

---

## 1. Ortam

- OS: Windows 11 (öncelik) ve Linux uyumlu. Yollar `pathlib` ile, shell'e özgü komut yok.
- GPU: NVIDIA, ≥ 8 GB VRAM, CUDA. Başlangıçta `torch.cuda.is_available()` kontrol edilsin, yoksa net bir hata verilsin.
- Python 3.11, paket yönetimi `uv`.
- Harici binary'ler `tools/` altında ya da PATH'te olmalı. Yolları `config.yaml`'dan okunur:
  - `ffmpeg` (zorunlu)
  - `vgmstream-cli` (sadece .wem girdisi varsa)
  - `rubberband` (zaman esnetme. Yoksa ffmpeg `atempo` ile devam edilir)

### 1.1 Ortam izolasyonu (ÖNEMLİ)

Chatterbox, Seed-VC, audio-separator, pyannote ve funasr'ın torch ve bağımlılık sürümleri çakışıyor. **Tek bir venv'e kurulmasın.**

```
envs/
  core/   # orchestrator, faster-whisper, pyloudnorm, jiwer, jinja2, typer, anthropic
  sep/    # audio-separator[gpu]
  diar/   # pyannote.audio (speaker diarization + embedding)
  emo/    # funasr + emotion2vec_plus_large
  tts/    # chatterbox-tts (multilingual)
  vc/     # seed-vc (git clone + requirements)
```

- Orchestrator `core` ortamında çalışır.
- Her ağır model `workers/<name>.py` olarak kendi venv'inin python'uyla `subprocess` üzerinden çağrılır.
- Worker sözleşmesi: girdi JSON dosyası yolu → çıktı JSON dosyası yolu. stdout sadece log için kullanılır.
- Her worker çalışıp kapanır. GPU belleğini tutmaz.
- `scripts/setup_envs.(ps1|sh)` tüm venv'leri kurar. `dubpoc doctor` komutu her ortamı ve binary'yi test eder.

---

## 2. Klasör yapısı

```
dubpoc/
  config.yaml
  speakers.yaml            # kullanıcı düzenler: cluster → karakter adı
  subtitles.csv            # opsiyonel: id,text_en (varsa ASR yerine kullanılır)
  input/                   # test dosyaları (.wav/.ogg/.mp3/.wem) — ASLA değiştirilmez
  record/                  # Yol B: kullanıcının okuduğu kayıtlar <seg_id>.wav
  work/
    manifest.jsonl
    ingest/  sep/  seg/  bank/<char>/  tts/<seg_id>/  vc/<seg_id>/  fit/  qa/
  out/
    A/<file_id>.wav        # Yol A final
    B/<file_id>.wav        # Yol B final
    report.html
    script.md              # Yol B için okuma senaryosu
  src/dubpoc/              # orchestrator paketi (typer CLI)
  workers/                 # her venv için worker scriptleri
  tools/
```

---

## 3. config.yaml (varsayılanlar)

```yaml
paths:
  ffmpeg: ffmpeg
  vgmstream: tools/vgmstream-cli.exe
  rubberband: tools/rubberband.exe

source_lang: en
target_lang: tr

separation:
  mode: auto            # auto | always | never
  model: ""             # boşsa: audio-separator --list_models içinden en yüksek SDR'li BS/Mel-Band RoFormer vocals modeli seçilip buraya yazılır
  clean_bg_threshold_dbfs: -45   # auto modda: bg RMS bunun altındaysa dosya "temiz" sayılır, bg atılır

diarization:
  mode: auto            # auto | always | never
  min_duration_s: 8     # auto modda bundan kısa dosyalar tek konuşmacı sayılır
  hf_token_env: HF_TOKEN

speakers:
  cluster_threshold: 0.55     # cosine distance, agglomerative
  bank_max_seconds: 20
  bank_min_clip_s: 1.5
  folder_name_as_character: true   # input/<karakter>/dosya.wav yapısı varsa clustering atlanır

asr:
  model: large-v3       # faster-whisper
  compute_type: float16

translate:
  provider: anthropic   # anthropic | openai_compat
  model: claude-sonnet-5-5
  api_key_env: ANTHROPIC_API_KEY
  base_url: ""          # openai_compat için (örn. Ollama)
  tr_syllables_per_sec: 6.0
  max_shorten_rounds: 2

tts:
  language_id: tr
  reference_mode: bank  # bank | original_line
  n_candidates: 3
  seeds: [11, 22, 33]
  exaggeration_range: [0.35, 0.9]  # intensity 0..1 → bu aralığa map
  cfg_weight_bank: 0.5
  cfg_weight_original_line: 0.0    # İngilizce referansla aksan sızmasın diye
  high_exaggeration_cfg: 0.3       # exaggeration ≥ 0.7 iken (bank modunda)

vc:
  backend: seed_vc      # seed_vc | chatterbox_vc
  diffusion_steps: 30
  inference_cfg_rate: 0.7

fit:
  max_stretch: 0.15     # ±%15'e kadar time-stretch
  shorten_trigger_ratio: 1.25   # tr_dur / src_dur bunu aşarsa çeviri kısaltılır
  pad_ms: 40

qa:
  wer_max: 0.15
  spk_sim_min_relative: 0.85   # karakterin kendi orijinal klipleri arası benzerliğe göre oran
  dur_tolerance: 0.10
  weights: {wer: 0.5, spk_sim: 0.3, dur: 0.2}
```

---

## 4. Manifest şeması (`work/manifest.jsonl`)

Her satır bir **segment**, yani bir replik. Tek konuşmacılı ve temiz bir dosya = 1 segment.

```json
{
  "seg_id": "f003_s02",
  "file_id": "f003",
  "src_path": "input/cutscene_03.wav",
  "src_sr": 48000, "src_channels": 2, "src_lufs": -23.1,
  "start": 4.21, "end": 7.88, "dur_src": 3.67,
  "voice_path": "work/seg/f003_s02.wav",
  "bg_path": "work/sep/f003_bg.wav",
  "cluster": "c1", "char": "trevor",
  "text_en": "...", "words": [{"w":"...", "s":0.0, "e":0.3}],
  "emotion": "angry", "emotion_score": 0.81, "intensity": 0.74,
  "syll_target": 22,
  "text_tr": "...", "tr_syllables": 23, "shorten_round": 0,
  "candidates": {
    "A": [{"tts":"...","vc":"...","fit":"...","wer":0.06,"spk_sim":0.71,"dur_ratio":1.03,"score":0.88}],
    "B": [{"rec":"record/f003_s02.wav","vc":"...","fit":"...","wer":0.02,"spk_sim":0.74,"dur_ratio":1.08,"score":0.91}]
  },
  "best": {"A": 0, "B": 0},
  "status": {"ingest":"ok","separate":"ok","diarize":"ok","speakers":"ok","asr":"ok",
             "emotion":"ok","translate":"ok","tts":"ok","vc":"ok","fit":"ok","qa":"review","mix":"pending"},
  "errors": []
}
```

Kurallar:

- Her stage sadece kendi alanlarını yazar.
- Her stage sadece `status[stage] != "ok"` olan satırları işler. `--force` verilirse hepsini tekrar işler.
- Yazma atomik olmalı: önce temp dosyaya yaz, sonra rename et.

---

## 5. Aşamalar

### 5.1 `ingest`

- `input/` altını tara. .wem dosyalarını `vgmstream-cli` ile wav'a çevir. Diğer formatları ffmpeg ile wav'a çevir.
- Orijinal sample rate, kanal sayısı ve LUFS'u (pyloudnorm) manifest'e yaz.
- Çalışma kopyası: `work/ingest/<file_id>.wav`, orijinal SR'de.
- `file_id` = `f001`, `f002`, … Orijinal yolu manifest'te tut.

### 5.2 `separate` (env: sep)

- `audio-separator` ile vocals ve instrumental ayrımı yap → `work/sep/<file_id>_voice.wav` ve `<file_id>_bg.wav`.
- `mode: auto` ise bg RMS `clean_bg_threshold_dbfs` altındaysa bg'yi at, `voice_path` olarak orijinali kullan. Oyun replik dosyalarının çoğu zaten temiz olduğu için bu durum sık görülür.
- Model boşsa ilk çalıştırmada `--list_models` çıktısından RoFormer tabanlı en yüksek SDR'li vocals modelini seç ve config'e yaz.

### 5.3 `diarize` (env: diar)

- `pyannote/speaker-diarization-community-1` kullan. HF token gerekli ve model koşulları kabul edilmiş olmalı.
- **`exclusive_speaker_diarization`** çıktısını kullan (çakışmasız).
- Segmentleri `voice` dosyasından kes → `work/seg/<seg_id>.wav`.
- 0.4 sn'den kısa segmentleri komşu aynı konuşmacı segmentiyle birleştir. Birleştirilemeyenleri `status=skip` yap.
- `mode: auto` ise `min_duration_s`'den kısa dosyalar tek segment sayılır.

### 5.4 `speakers` (env: diar)

- `folder_name_as_character: true` ise ve dosya `input/<karakter>/` altındaysa `char` = klasör adı olur, clustering atlanır.
- Değilse:
  1. Her segment için `pyannote/wespeaker-voxceleb-resnet34-LM` ile embedding çıkar.
  2. Tüm segmentleri **dosyalar arası global** olarak agglomerative clustering ile grupla (cosine).
  3. `speakers.yaml` dosyasını üret: `c1: {name: null, samples: [3 seg yolu]}`. Kullanıcı isimleri doldurur. Doldurulmamışsa `char = cluster`.
- **Ses bankası:** Her karakter için en uzun ve en temiz segmentlerden (`bank_min_clip_s` üstü, SNR'ye göre sıralı) toplam `bank_max_seconds` sürelik ses birleştir → `work/bank/<char>/ref.wav`. Ayrıca `ref_emb.npy` kaydet.
- **Baseline:** Aynı karakterin orijinal klipleri arasındaki ortalama cosine benzerliğini `bank/<char>/baseline_sim.txt` dosyasına yaz. QA bu değeri kullanır.

### 5.5 `asr` (env: core)

- `subtitles.csv` içinde ilgili satır varsa `text_en` oradan alınır.
- Yoksa faster-whisper `large-v3` çalıştır, `language=en`, `word_timestamps=True`.

### 5.6 `emotion` (env: emo)

- funasr ile `emotion2vec_plus_large` çalıştır → `emotion`, `emotion_score`.
- `intensity` (0..1) şöyle hesaplanır: `0.5*emotion_score(nötr değilse) + 0.25*norm(RMS) + 0.25*norm(pitch_std)`. Normalizasyon karakter bazında min-max ile yapılır.

### 5.7 `translate` (env: core)

- Hece bütçesi: `syll_target = round(dur_src * tr_syllables_per_sec)`.
- Türkçe hece sayısı = sesli harf sayısı (`aeıioöuüAEIİOÖUÜ`).
- Çeviri **dosya/sahne bazında toplu** yapılır, böylece bağlam korunur. Çıktı JSON şeması zorunlu.
- Prompt şablonu:

```
Sen bir oyun dublaj çevirmenisin. Aşağıdaki replikleri İngilizceden doğal, konuşma dili Türkçeye çevir.
Kurallar:
- Her replik verilen hece bütçesine ±%10 içinde sığmalı (Türkçede hece = sesli harf sayısı).
- Karakterin üslubunu ve duygusunu koru (emotion/intensity alanlarına bak). Argo ve küfür seviyesini koru.
- Dudak senkronu yok; anlamı ve süreyi önceliklendir. Kelime kelime çevirme.
- Özel isimleri, yer adlarını, oyun terimlerini çevirme.
- Ünlem, duraksama ("...") ve vurguyu noktalamayla ver; TTS bunları kullanır.
Sadece şu JSON'u döndür: [{"seg_id":"...","text_tr":"..."}]

Sahne bağlamı (sırayla):
[{"seg_id","char","emotion","intensity","syll_target","text_en"}, ...]
```

- Dönen çeviride `tr_syllables` hesapla. `> syll_target*1.15` ise kısaltma turu çalıştır ("şu repliği X heceye indir") (maks `max_shorten_rounds`).
- Aynı adımda `out/script.md` üret: Yol B için okuma senaryosu. Tablo kolonları: seg_id, karakter, duygu, hedef süre, EN, TR. Kayıt dosya adı = `record/<seg_id>.wav`.

### 5.8 `tts` (env: tts) — Yol A

- `ChatterboxMultilingualTTS`, `language_id="tr"`. Paketin en güncel multilingual checkpoint'i (v3 varsa v3) kullanılsın.
- Referans ses:
  - `reference_mode: bank` → `bank/<char>/ref.wav`, `cfg_weight = cfg_weight_bank`
  - `reference_mode: original_line` → segmentin kendi orijinal sesi, `cfg_weight = 0.0` (aksan sızmasın diye)
- `exaggeration = lerp(exaggeration_range, intensity)`. Bank modunda exaggeration ≥ 0.7 ise `cfg_weight = high_exaggeration_cfg`.
- `n_candidates` aday üret (farklı seed'lerle) → `work/tts/<seg_id>/cand_<k>.wav`.
- Model bir kez yüklenir, tüm segmentler batch olarak işlenir.

### 5.9 `vc` (env: vc) — Yol A ve B

- Kaynak: Yol A için her TTS adayı, Yol B için `record/<seg_id>.wav` (dosya varsa).
- Hedef: `bank/<char>/ref.wav`.
- Seed-VC zero-shot, **konuşma** checkpoint'i (singing değil). `diffusion_steps` ve `inference_cfg_rate` config'den alınır.
- Alternatif backend: `chatterbox_vc` (Chatterbox'ın kendi VC'si). Arayüz aynı kalsın.
- Çıktı → `work/vc/<seg_id>/{A_k|B}.wav`.

### 5.10 `fit` (env: core)

- Baştaki ve sondaki sessizliği kırp (ffmpeg `silenceremove`, -45 dB).
- `ratio = dur_tr / dur_src`:
  - `ratio > shorten_trigger_ratio` → `status.translate = "shorten"` yap. Pipeline `translate → tts → vc → fit` adımlarını sadece bu segment için tekrar çalıştırır (maks `max_shorten_rounds`).
  - `|ratio-1| ≤ max_stretch` → rubberband ile tam `dur_src`'ye esnet (pitch korunur).
  - Kısa kalıyorsa sona sessizlik ekle (`pad_ms` başa).
- Loudness'ı `src_lufs`'a eşitle (segment orijinalinin LUFS'u). Sonra true-peak -1 dBTP limiter uygula.
- Orijinal SR ve kanal sayısına geri çevir → `work/fit/<seg_id>_<A_k|B>.wav`.

### 5.11 `qa` (env: core + diar)

- **WER:** faster-whisper `language=tr` ile çıktıyı yazıya dök, `text_tr` ile karşılaştır (jiwer).
  - Normalizasyon Türkçeye uygun olmalı: lowercase'den önce `I→ı`, `İ→i` çevir, noktalamayı sil.
- **spk_sim:** Çıktının embedding'i ile `ref_emb` arasındaki cosine benzerlik. Raporlanan değer = `spk_sim / baseline_sim` oranı.
- **dur_ratio:** fit sonrası süre / `dur_src`.
- `score = Σ weights` (her metrik 0..1'e normalize edilir).
- Yol A'da en iyi aday `best.A` olarak seçilir.
- Durum:
  - `ok`: tüm eşikler geçti
  - `review`: bir eşik kaldı
  - `fail`: WER > 0.4 ya da çıktı yok

### 5.12 `mix` (env: core)

- Her dosya ve her yol (A, B) için orijinal süre uzunluğunda boş bir timeline oluştur. Segmentlerin `best` çıktılarını kendi `start` zamanlarına yerleştir.
  - Yol B'de kaydı olmayan segment varsa yerine Yol A'nın en iyi çıktısını koy ve rapora not düş.
- `bg_path` varsa topla. Clip olursa limiter uygula.
- Orijinal SR, kanal ve format ile `out/<A|B>/<file_id>.wav` olarak yaz.

### 5.13 `report` (env: core)

`out/report.html` tek dosya olmalı, audio dosyaları göreli yolla referans verilir. Jinja2 ile üretilir.

- Dosya bazında: orijinal tam dosya / Yol A final / Yol B final oynatıcıları.
- Segment bazında bir satır:
  - seg_id, karakter, duygu/intensity, EN metin, TR metin
  - Oynatıcılar: orijinal segment, A best, tüm A adayları (açılır panel), B
  - Metrikler: WER, spk_sim oranı, dur_ratio, durum rozeti
- Üstte özet: segment sayısı, ok/review/fail dağılımı, A ve B ortalama metrikleri.
- Karakter bazında ses bankası oynatıcısı ve baseline_sim.
- Her segmentte kullanıcının kendi notunu verebileceği 1–5 puan alanı. Puanlar localStorage'da (try/catch) tutulur, "JSON indir" butonuyla dışa aktarılır.

---

## 6. CLI

```
dubpoc doctor                      # ortam/binary/GPU/HF token/API key kontrolü
dubpoc ingest | separate | diarize | speakers | asr | emotion | translate | tts | vc | fit | qa | mix | report
dubpoc run [--from STAGE] [--to STAGE] [--only SEG_ID] [--force]
dubpoc run --path A|B|both         # varsayılan both; B sadece record/ dosyası olan segmentler için
dubpoc status                      # manifest özet tablosu (rich)
```

`run` adımları sırayla çalıştırır. `speakers` adımından sonra `speakers.yaml`'da isim yoksa uyarı verir ama durmaz.

---

## 7. Kabul kriterleri

1. 5–15 dosyalık test setinde `dubpoc run` manuel müdahale olmadan sonuna kadar çalışır. Tek istisna isteğe bağlı karakter isimlendirme.
2. Her segment için en az Yol A çıktısı oluşur ve QA durumu belirlenir.
3. `out/report.html` tarayıcıda açılır, tüm sesler çalar.
4. Bir stage'i tekrar çalıştırmak sadece bekleyen satırları işler. `--force` hepsini işler.
5. `input/` klasöründeki dosyalar hiçbir koşulda değişmez.
6. Hedef metrikler (ok durumundaki segmentlerde): WER ≤ 0.15, spk_sim oranı ≥ 0.85, dur_ratio 1 ± 0.10.
7. Her worker hatası manifest'teki `errors` alanına yazılır. Pipeline o segmenti atlayıp devam eder.

---

## 8. Bilinen tuzaklar

- **Bağımlılık çakışması:** Bkz. 1.1. Tek venv kesinlikle olmasın.
- **pyannote:** HF token ve model koşullarının kabulü gerekli. Girdi 16 kHz mono'ya çevrilir (pipeline kendisi yapar). ffmpeg/torchcodec yüklü olmalı.
- **Chatterbox:**
  - İngilizce dışında kalite değişken.
  - İngilizce referans klip kullanılırsa aksan sızar → `cfg_weight=0`.
  - Yüksek exaggeration konuşmayı hızlandırır → cfg düşürülmeli.
  - Çıktılar Perth watermark içerir.
  - Uzun metinde sınır artefaktları oluşur. 250 karakteri aşan replikleri cümle bazında böl, sonra birleştir.
- **Türkçe lowercase:** Python'un `lower()` fonksiyonu `I`'yı `i` yapar, Türkçede bu yanlış. Önce manuel `I→ı`, `İ→i` çevir.
- **Seed-VC:** Konuşma modeli ile şarkı modeli farklı checkpoint'ler. SR'ler farklı olabilir, çıktıyı fit aşamasında orijinal SR'ye çevir.
- **GPU belleği:** Her worker tek model yükler ve kapanır. Aynı anda iki GPU worker çalıştırma.
- **Sessiz/efor segmentleri:** ASR boş ya da 2 kelimeden kısa metin döndürür ve emotion "nonverbal" benzeri çıkarsa segmenti `status=keep_original` yap ve mix'te orijinalini kullan. Nefes, hırıltı ve gülme sesleri çevrilmez.

---

## 9. Kapsam dışı (POC'de yok)

- .wem/.bnk'ya geri dönüştürme ve oyuna paketleme (Wwise). Sadece `export/` için bir stub fonksiyonu ve TODO bırak.
- Karakter başına RVC modeli eğitimi. POC'de zero-shot Seed-VC yeterli.
- Dudak senkronu.
- Web arayüzü. Tek çıktı report.html.

---

## 10. Uygulama sırası (milestone)

| # | İçerik | Bitti sayılır |
|---|---|---|
| M1 | Proje iskeleti, config, manifest, `doctor`, `ingest`, `asr`, `translate`, `tts`; tek konuşmacılı temiz dosyalar | Temiz dosyalardan Türkçe TTS çıkıyor, script.md oluşuyor |
| M2 | `speakers` (klasör modu + clustering + bank), `vc`, `fit`, `qa`, aday seçimi, `report` | report.html'de A yolu metrikleriyle dinlenebiliyor |
| M3 | `separate`, `diarize`, `mix` | Müzikli ve çok konuşmacılı dosya uçtan uca dublajlanıyor |
| M4 | Yol B (record/ → vc → fit → qa → mix), A/B karşılaştırma raporu | İki yol yan yana dinlenip puanlanabiliyor |

Her milestone sonunda `README.md` güncellenir: kurulum, çalıştırma, bilinen sorunlar.

---

## 11. Test seti önerisi (kullanıcı hazırlar)

- 10 normal replik (tek karakter, temiz dosya). Mümkünse 2–3 farklı karakter, `input/<karakter>/` altında.
- 5 duygulu replik (bağırma, üzüntü, heyecan).
- 1–2 karışık dosya (ara sahne: müzik + efekt + 2 konuşmacı).

> Not: Bu POC kişisel deneme amaçlıdır. Orijinal seslendirme sanatçılarının sesine benzetilmiş çıktılar dağıtılmamalıdır.
