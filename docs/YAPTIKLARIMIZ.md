# Ne yaptık, nasıl yaptık (2026-11-20 → 2026-11-21)

Bu dosya projenin baştan sona hikâyesi: her adımda ne istendi, ne bulundu, koda/ayarlara ne girdi. Ölçümlerin
ayrıntısı [ARASTIRMA.md](ARASTIRMA.md), kalan sınırlar [BILINEN_SORUNLAR.md](BILINEN_SORUNLAR.md), yeni bir PC'de
kurulum ve kaldığın yerden devam [KURULUM_VE_DEVAM.md](KURULUM_VE_DEVAM.md).

> Kişisel deneme. Dublaj çıktıları orijinal seslendirme sanatçılarının sesini taklit eder; dağıtılmaz. Bu yüzden
> ses/video dosyaları ve ara çıktılar git'e konmadı (bkz. `.gitignore`).

## 0. Ne inşa ettik?

İngilizce oyun/ara sahne sesini Türkçeye dublajlayan bir hat (`dubpoc`). Spec: [SPEC_oyun-tr-dublaj-poc.md](SPEC_oyun-tr-dublaj-poc.md).
Aşamalar (her biri ayrı Python ortamında, `work/manifest.jsonl` üzerinden haberleşir):

| Aşama | Ne yapar | Kullanılan |
|---|---|---|
| ingest | Girdiyi 48 kHz WAV'a çevirir | ffmpeg / vgmstream |
| separate | Konuşma / müzik / efekt ayrıştırma | **Bandit v2 DnR** (MSST) |
| diarize | Replik segmentleri; arka plandaki konuşmalar (`bg` katmanı) | duraklama bölme + Silero VAD + Whisper tarama |
| speakers | Konuşmacı kümeleme, karakter ses bankası | ses embedding'i + `speakers.yaml` elle düzeltme |
| asr | İngilizce metin | faster-whisper large-v3 (dosya bağlamıyla) |
| emotion | Duygu ve yoğunluk | emotion2vec+ large |
| translate | Hece bütçeli Türkçe çeviri | Claude (anahtar varsa) / Ollama / **`translations.csv` (elle)** |
| tts | Karakter sesiyle Türkçe, replik başına çok aday | **VoxCPM2** (ayrıca Chatterbox, OmniVoice, XTTS-v2 hazır) |
| vc | Ses dönüştürme | Seed-VC (doğal modda kapalı) |
| fit | Süre/ses eşleme, kanal efekti (telefon/megafon) | pedalboard Rubber Band, `fx` modülü |
| qa | Adayları puanla, en iyisini seç | Whisper TR (CER), konuşmacı benzerliği, duygu, tonlama, Türkçe aksan |
| mix | Zaman çizelgesi + arka plan, orijinal konuşmayı bastırma | spektral kenetleme |

Çalıştırma: `envs\core\Scripts\dubpoc.exe run` (tümü) ya da `run --from X --to Y --only seg_id --force`.

## 1. Kurulum ve ilk araştırma (2026-11-20)

- **İstek:** "Dosyayı oku ve yap; yaparken daha iyi nasıl yaparız diye araştır."
- Spec'teki M1–M4 kuruldu. 10 sanal ortam (`scripts/setup_envs.ps1`, uv, CUDA 12.6): core, dnr, sep, diar, emo,
  tts, vc, omni (+ isteğe bağlı xtts, voxcpm). Ağır modeller ayrı worker süreçlerinde, tek tek yüklenip kapanıyor:
  6 GB'lık RTX 4050'de böyle sığıyor.
- Araştırmadan koda girenler (ARASTIRMA §1): müzik vocals modeli yerine **DnR 3-stem** ayrıştırma (konuşma SDR
  −1,1 → 20,5 dB), Whisper halüsinasyon yakalama, Türkçe için CER, duygu yoğunluğu = 1 − p(neutral), çok aday +
  QA seçimi, hece bütçesinin konuşma süresiyle hesaplanması, HF token yokken duraklama tabanlı bölme.
- Sentetik test seti (Windows SAPI sesi) ile uçtan uca çalıştırıldı.

## 2. Gerçek oyun sesi: GTA 6 "Good Ending" videosu (YouTube `DYDz_cIgoAw`)

1. **İlk 60 sn denemesi.** Kullanıcı: "çeviri çok kötü", "seslendirme kötü", "duygular kötü".
   - Yerel 9B model (Ollama `qwen3.5:9b`) çeviride kullanılamaz kalitede (küfürleri sansürledi, cümleleri bozdu).
     **Çözüm:** çeviriler Claude tarafından elle yazılıp `translations.csv`'ye kondu (`seg_id,text_tr,style,...`).
     `style` sütunu VoxCPM2'ye giden İngilizce oyunculuk notu ("shocked and alarmed, urgent").
   - ASR parça parça bağlamı kaybediyordu → `asr.file_context` (tüm dosya tek seferde yazılır).
   - HF token yok → konuşmacılar perde ölçümü + görüntüyle elle bağlandı (`speakers.yaml` → `overrides`).
   - TTS: XTTS-v2 (kullanıcı CPML ticari-olmayan lisansını kabul etti) ve **VoxCPM2** eklendi. VoxCPM2 metin
     talimatıyla duygu veriyor ve QA'da birinci çıktı. Kullanıcı: "Sadece VoxCPM2 ile hazırla."
2. **Tam video (8:54), sadece VoxCPM2.** Video iki parça 720p (<30 MB, telefona) olarak gönderildi:
   `scripts/mux_gta_video.sh <etiket>`.
3. **"Sesler robotikleşiyor, Türkçe bozuk."** → **Doğal mod** (`config.local.yaml`): Seed-VC kapalı, yavaşlatma yok,
   en fazla %8 sıkıştırma ve yalnızca sonraki repliğe çarparsa (`fit.use_slot`). **Türkçe devam istemi:** VoxCPM2'ye
   aynı karakterin daha önce iyi çıkmış bir Türkçe repliği istem olarak veriliyor (`tts.voxcpm.tr_prompt_candidates`,
   `vt1/vt2` adayları). Ölçüm: istemli adaylarda CER 0,009, eskilerinde 0,253. QA'ya tonlama (`prosody`) eklendi.
   Bu yüzden her projede TTS **iki tur**: önce normal adaylar → QA → sonra istemli adaylar.
4. **"Bazı kelimeler İngilizce kalmış."** Dublajı olmayan segmentte mix orijinal sesi çalıyordu. Sebepler ve
   düzeltmeler: kısa replikler artık atılmıyor, Silero VAD sessiz replikleri ekliyor, duygu modeli tek kelimelik
   sözcükleri "efor" saymıyor, Whisper'ın uzun dosyada halüsinasyon döngüsü için `asr.file_vad`. QA'ya Türkçe aksan
   ölçüsü (`lang`: Whisper dil tanıma) ve sonraki repliğe binme cezası (`slot_overflow_s`).
5. **"Polis helikopteri anonsları, arka plandaki konuşmalar çevrilmiyor."** DnR megafon/uzaktan bağırmayı efekt
   stem'ine koyuyor; 3 farklı ayrıştırma denemesi de ayıramadı. **Çözüm:** arka plan (müzik+efekt) 10 sn'lik
   örtüşen pencerelerle Whisper'la taranıyor (`diarization.bg_speech`), bulunan konuşmalar `bg` katmanı satırları
   (`f001_bNN`) olarak dublajlanıyor, orijinali mix'te **spektral kenetlemeyle** arka plandan bastırılıyor
   (`audio.suppress_speech`, medyanın 6 dB altı). Tek kelimelik ünlemlerde kayan zamanlar ±1 sn'lik yeniden yazımla
   düzeltildi. Diyaloğun arka plana sızan kopyaları (`bg_leak_of`) dublajlanmıyor, sadece bastırılıyor.
6. **"Kısık sesleri algılamıyor; megafon efekti yok; telefon efekti yok."**
   - Videonun gömülü altyazısı "gerçek replik listesi" olarak kullanıldı (altyazı bandında 4 fps metin pikseli
     sayımı). Kaçan tek altyazılı replik 8:19 "Huh?" (0,24 sn): tek geçişli Silero ve 0,3 sn alt sınırı kaçırıyordu
     → VAD 20 sn'lik örtüşen parçalarda, 0,15 sn'lik kısa ünlemler de segment. 3:16 "jump" sızıntı sayılıp
     bastırılmıştı → sızıntıda diyalogda olmayan baş/son kelimeler ayrı satır (`asr.split_leak_extras`).
   - **Kanal efektleri** (`src/dubpoc/fx.py`, `fit` aşamasında): telefon = bant geçiren + μ-law codec + kompresör;
     megafon = bant geçiren + sert distorsiyon + kompresör + dış mekân yankısı; telsiz. Bant kenarları ve son tını
     orijinal replikten ölçülür. Seçim: `translations.csv` `fx` sütunu (`phone|megaphone|radio|none`) > style
     notu (orijinal gerçekten dar bantsa) > otomatik (alçak+tiz kesikse telefon). QA efektsiz kopyayı ölçer.
   - Yanlış alarm notu: Türkçe miksi İngilizce zorlamalı Whisper'a verince Whisper Türkçeyi **çeviriyor**;
     İngilizce kalıntı testi miks üzerinde değil, arka plan katmanında yapılmalı.
   - Son video: `out/video/gta6_tam_tr_efekt_*` (bu makinede; git'te yok).

## 3. Donanım soruları

- **RTX 5060 Ti 16 GB:** ~2 kat hız (bellek bant genişliği), replik başına daha çok aday, Türkçe/karakter ince ayarı
  (fine-tune) mümkün. Ama kaliteyi asıl sınırlayan Türkçe destekli açık TTS modelleri; 6 GB'a sığmayan "daha iyi
  Türkçe model" yok. RTX 50 serisi için ortamlar `-Torch cu128` ile kurulmalı.
- **RX 6800:** CUDA yok; Windows'ta modellerimiz pratikte CPU'da çalışır. Linux + ROCm ile çoğu çalışır ama
  faster-whisper vb. değişmeli. Önerilmedi.

## 4. 007 First Light duyuru fragmanı (YouTube `J4qY9DYE184`, 3:14)

Ayrı proje kökü: `projects/007_first_light/` (ortamlar/modeller ana klasörden göreli yolla, `workers/` junction).
`dubpoc --root projects/007_first_light run ...` ile çalışır.

- Otomatik segmentleme 35 satır, kümeleme 20 "konuşmacı" verdi (HF token yok). Konuşmacılar görüntü + ses
  embedding'i + perde (f0) ile elle bağlandı: **M** (kadın, anlatıcı), **Greenway** (M'ye "Mum" = ma'am diyen),
  **Bond**, Resepsiyonist, Ajan (kadın), Yabanci (partide soru soran).
- İki kişinin tek segmente düştüğü iki satır bölündü, ikiye bölünmüş bir replik birleştirildi, "Lately," doğru
  satıra alındı: `projects/007_first_light/fix_segments.py`. Düzeltilmiş İngilizce: `subtitles.csv`.
- 32 replik Türkçeye çevrildi (`translations.csv`; M'ye hitap "efendim", "009" → "sıfır sıfır dokuz").
  "PlayStation" logosu ve iki nefes sesi orijinal bırakıldı (`subtitles.csv`'de `-`).
- İki TTS turu: 32 replik × 3 normal aday, sonra QA'dan geçen Türkçe okumalardan 2 istemli aday. Seçilenlerin
  12/32'si istemli; seçilen adayların hepsi kendi başına CER 0 (Greenway'in "009" satırı hariç, 0,14), fail 0.
- Mikste bulunan sorun: 1:25 "So you shouldn't have." ayrıştırıcıda ikiye bölünmüştü (ses stem'inde −34,7 LUFS,
  efekt stem'inde −21,7 LUFS); dublaj sadece ilk kısma eşlenip ikinci kısım da bastırıldığı için müziğin altında
  kayboluyordu. Düzeltme (genel): `fit.reference_speech` sızıntısı olan replikte referansa arka plandaki kısmı da
  ekliyor; dublaj −23 dB'e çıktı, Whisper mikste "Ah hiç zahmet etmeseydiniz" duyuyor. İşlenmiş arka planda
  gerçek İngilizce kalıntı yok (eşleşen 7 kelime müzik üstü Whisper uydurması).
- Kullanıcı: "bazı sesleri atlamış, özellikle baştaki M". Ölçümle atlanmış replik bulunamadı (M'nin tüm satırları
  mikste ve gönderilen 720p dosyada Whisper'la doğru okunuyor; satır başları ses stem'iyle hizalı; konuşmacı ve perde
  doğru). Bulunan gerçek kusur miks limiteriydi (BILINEN_SORUNLAR: miks limiteri); düzeltilip v2 videosu gönderildi.
- **Bitti (2026-11-21):** `projects/007_first_light/out/video/007_first_light_tr_v1_720p.mp4` (telefona gönderildi)
  ve `..._1080p.mp4` (iki ses izi: Türkçe + orijinal).

## 5. Kod haritası (en çok değişen yerler)

| Dosya | Ne var |
|---|---|
| `src/dubpoc/stages/diarize.py` | duraklama bölme, parçalı VAD, kısa ünlemler, `dialog_row`/`bg_row`, arka plan taraması |
| `src/dubpoc/stages/asr.py` | dosya bağlamlı ASR, sızıntı tespiti (`bg_duplicates`), `split_leak_extras` |
| `src/dubpoc/stages/tts.py` + `workers/voxcpm_worker.py` | VoxCPM2 adayları, style talimatı, Türkçe istem havuzu |
| `src/dubpoc/stages/fit.py` + `src/dubpoc/fx.py` | süre eşleme, kanal efekti, `fit_dry` |
| `src/dubpoc/stages/qa.py` | CER, konuşmacı, duygu, `prosody`, `lang`, taşma cezası |
| `src/dubpoc/stages/mix.py` + `src/dubpoc/audio.py` | yerleştirme, `suppress_speech`, `match_spectrum` |
| `config.yaml` / `config.local.yaml` | varsayılanlar / bu makinede kullanılan doğal mod + VoxCPM2 ayarları |
| `tests/test_core.py` | 44 test: `envs\core\Scripts\python.exe -m pytest -q` |
