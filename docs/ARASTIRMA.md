# Araştırma: "Daha iyi nasıl yaparız?" (2026-11-20)

Spec uygulanırken her aşama için güncel durum araştırıldı. Bu dosya bulguları ve **hangilerinin koda
uygulandığını** özetler. Kaynaklar en altta.

## 1. En önemli bulgular

| # | Bulgu | Etki | Durum |
|---|---|---|---|
| 1 | **Chatterbox Multilingual V3** (2026-06) sadece GitHub master'da; PyPI 0.1.7'de yok. Kütüphanenin varsayılanı hâlâ **v2**. | Spec "v3 varsa v3" diyor; PyPI'den kurulursa sessizce v2 kullanılırdı. | ✅ `requirements/tts.txt` git commit'ine sabitlendi, worker `t3_model="v3"` veriyor (`tts.chatterbox_t3_model`). |
| 2 | Chatterbox "retired research" (2026-08), **Seed-VC arşivlendi** (2025-11). İkisi de çalışıyor ama bakım yok. | Uzun vadede risk. | ✅ Commit'ler sabitlendi. Alternatif backend'ler eklendi. |
| 3 | **OmniVoice** (k2-fsa): Türkçe destekli, Türkçe WER 2,17 / benzerlik 0,851 (ElevenLabs Multilingual v2: 0,70 / 0,596), ve **`duration=` ile hedef sürede üretim**. | Dublajın asıl sorunu süre: sonradan esnetmek yerine doğru uzunlukta üretmek prozodiyi korur. | ✅ `omnivoice` backend'i (`envs/omni`). Varsayılan: Chatterbox 3 aday + OmniVoice 2 aday, QA en iyisini seçer. Ağırlıklar **CC-BY-NC** (kişisel POC için uygun). |
| 4 | VC Türkçe anlaşılırlığı düşürebilir (Seed-VC kendi değerlendirmesi: İngilizcede WER 8,0 → 12,0). | TTS→VC zinciri her zaman daha iyi değil. | ✅ `vc.also_score_without_vc`: ham TTS adayları da QA'da yarışır. |
| 5 | Seed-VC'nin `length_adjust` parametresi süreyi dönüşüm sırasında değiştirebilir. | Rubberband esnetmesi azalır. | ✅ `vc.length_adjust` (±%20), kalan fark fit'te esnetilir. |
| 6 | Müzik için eğitilmiş vocals modelleri bazı konuşmaları "enstrüman" sanabilir; ara sahneler için asıl doğru araç **DnR (diyalog/müzik/efekt) 3-stem** modelleri. audio-separator'da DnR modeli yok. | Ses stem'i boş kalır → dublaj sessiz/yanlış olur. | ✅ **Bandit v2 DnR** varsayılan motor (`separation.engine: dnr`, ayrı `envs/dnr`, MSST model kodu, çok dilli checkpoint). Ölçüm: konuşma SDR −1,1 → **20,5 dB** (bkz. §2). `voice_missing` koruması ve eski RoFormer yolu (`engine: audio_separator`, isteğe bağlı `dnr_refine_vocals` hibrit) duruyor. |
| 7 | Whisper müzik/sessizlikte "Thank you for watching" gibi halüsinasyonlar üretir; ayrıştırma artefaktları bunu tetikleyebilir. | Yanlış metin çevrilip dublajlanır. | ✅ Şüpheli sonuçlar (kalıp, kelime/sn çok düşük, no_speech yüksek) ayrıştırılmamış miks üzerinde tekrar denenir; düzelmezse hata olarak işaretlenir (test setinde iki kez yakaladı ve düzeltti). |
| 8 | Türkçe eklemeli: WER ek hatalarını iki kez cezalandırır; **CER birincil metrik** olmalı. | Spec'in WER ≤ 0,15 eşiği Türkçe için sert. | ✅ CER her adayda raporlanır; `qa.text_metric: cer` ile durum kararı CER'e geçer (varsayılan spec gereği `wer`). |
| 9 | Spec'teki yoğunluk formülü `emotion_score` = sınıflandırıcı güveni; emin bir "neutral" da yüksek puan alır. | Nötr replikler gereğinden heyecanlı üretilir. | ✅ `1 − p(neutral)` kullanılıyor (`emotion.intensity_source`). |
| 10 | Spec'in "aynı heyecanla" hedefinin QA metriği yok. emotion2vec embedding kosinüsü akustik benzerliği ödüllendiriyor (False Resonance, 2026). | Duygu aktarımı ölçülmüyordu. | ✅ `emo_sim` = 1 − JS ıraksaması (sınıf dağılımları, other/unknown hariç). Raporda; ağırlık `qa.weights.emo`. |
| 11 | İzokroni araştırması (EMNLP 2025, IWSLT 2025): LLM'ler ancak **uç kısa örnekler** gösterilince güvenilir kısaltır; birden çok aday üretip kodla seçmek en iyi denge. | Kısaltma turları daha az. | ✅ Kısaltma istemi 3 örnekli, 3 aday ister; hece sayımı kodda doğrulanır. Sayılar yazıyla istenir. |
| 12 | Hece bütçesi segment süresiyle (sessizlik dahil) hesaplanırsa şişer. | Çeviri gereğinden uzun olur. | ✅ Bütçe **konuşma süresi** (`dur_speech`) ile; dublaj orijinaldeki konuşma başlangıcına yerleştirilir. |
| 13 | pyannote 4'ün dosya okuması torchcodec+FFmpeg ister, Windows'ta sorunlu. | Kurulum kırılır. | ✅ Ses soundfile ile okunup waveform olarak veriliyor. |
| 14 | HF token yoksa diarization hiç çalışmıyordu. | Ara sahneler tek segment kalır. | ✅ Yedek: duraklama tabanlı bölme + embedding ile konuşmacı atama; klasör karakterleri "çapa" olarak kullanılır. |
| 15 | uv `--torch-backend`: doğru CUDA index'ini otomatik seçer; ortamlar arası aynı torch sürümü cache'ten hardlink'lenir. | Kurulum basit ve disk dostu. | ✅ `scripts/setup_envs.ps1`. |
| 16 | pedalboard (Spotify) Rubber Band R3'ü gömülü getiriyor. | Harici rubberband.exe gerekmez. | ✅ Varsayılan esnetme motoru. |
| 17 | Anthropic yapılandırılmış çıktı (`output_config.format` JSON şeması) + sunucu tarafı fallback. | JSON ayrıştırma hatası yok; reddedilen istek yedek modele gider. | ✅ `translate` (Claude Sonnet 5.5). Anahtar yoksa `openai_compat` ile yerel Ollama (Ollama modeli işten sonra VRAM'den boşaltılır). |

## 2. Bu projede ölçülenler (sentetik test seti, RTX 4050 6 GB)

Test seti: Windows SAPI sesiyle üretilmiş 8 tek replik (2 karakter, klasör modu) + 1 müzikli 2 konuşmacılı
sahne. Gerçek oyun sesi değil; eğilimler için bakın, mutlak sayılar için değil.

- **Süre hedefleme:** OmniVoice `duration=` ile konuşma süresi/hedef oranı **0,87–0,99**; Chatterbox'ta aynı
  repliğin seed'leri arasında **0,69–1,99** (bir aday kendini tekrar ederek 2 kat uzadı). Çoklu aday + QA seçimi
  bu yüzden şart; kısaltma döngüsüne hiç gerek kalmadı.
- **Ama:** OmniVoice'ta sıkı hedef süre bazı Türkçe repliklerde anlaşılırlığı bozdu ("Kasa kodu" → "Kudu") ve
  kötü referansla (müzikten zor ayrışmış ses) tamamen anlamsız çıktı verdi. Chatterbox bozuk referansa çok
  daha dayanıklı. → OmniVoice adayları artık karışık: biri hedef sürede, biri doğal sürede (`duration_modes`).
- **OmniVoice `ref_text` tuzağı:** referans ses kırpılıp metin kırpılmazsa çıktı bozuluyor. Banka artık tam
  kliplerden ≤12 sn'lik `ref_short.wav` + birebir metin üretiyor.
- **VC etkisi:** ham TTS ile VC sonrası arasında WER farkı koşudan koşuya değişti (ör. 0,04 → 0,15 ve
  0,08 → 0,07). Araştırmadaki "VC anlaşılırlığı düşürebilir" bulgusu kısmen doğrulandı; iki türün de QA'da
  yarışması (`also_score_without_vc`) doğru karar. İkinci tam koşuda kazananların 7/10'u VC'siz Chatterbox oldu.
- **Müzik vocals modeli + sentetik konuşma:** Mel-RoFormer, SAPI sesini temiz dosyalarda neredeyse tamamen
  "enstrüman"a koydu (ses stem'i orijinalden 34 dB düşük) → `voice_missing` koruması 8/8 dosyada devreye girdi.
- **DnR ile ölçüm (Bandit v2):** sahne dosyasının gerçek bileşenleri birebir yeniden üretilip (hata −70 dB)
  SDR ölçüldü:

  | Motor | Konuşma SDR | Arka plan SDR |
  |---|---|---|
  | Mel-RoFormer vocals | −1,1 dB | −6,2 dB |
  | **Bandit v2 DnR** | **20,5 dB** | **15,5 dB** |
  | ayırmamak | 5,0 dB | – |

  Temiz dosyalarda DnR arka planı −79…−93 dBFS (pratikte sessiz) → 8/8 doğru "clean". Sahnede sonuç zinciri:
  duraklama bölmesi 4 → 7 doğru parça, Whisper halüsinasyonu 1 → 0, karakter ataması 0/7 → **7/7**
  (HF token olmadan; ek olarak "küme → çapa" adımı: tek tek eşiği geçemeyen kısa replikler küme merkeziyle
  karaktere bağlanıyor). 6 GB'ta batch 2 ile 9 dosya ~13 sn.
- **Whisper halüsinasyonu:** ayrıştırılmış sahne sesinde "Thank you for watching." (EN) ve QA'da
  "Altyazı M.K." (TR) üretildi. Miks üzerinde tekrar deneme 4 şüpheli sonucun 3'ünü düzeltti; sadece müzik
  içeren son parça hata olarak işaretlendi ve miks orijinali kullandı.
- **Sayılar:** ASR "2", metin "iki" → WER cezası; Türkçe sayı→yazı normalizasyonu eklendi.
- **Büyük harf:** ASR bağırılan "SAFE"i büyük harfle yazdı, LLM "GÜVENLİ" döndürdü; Chatterbox bunu 6 adayın
  6'sında "Güvenliş" okudu. Çeviri ve TTS girişinde Türkçe kurallı büyük harf normalizasyonu eklendi.
- **Çok kısa hedef:** OmniVoice 0,58 sn hedefte ("Fast" → "Hızlıca") anlamsız ses üretti; 1 sn altı
  hedeflerde sadece doğal süre kullanılıyor (`min_target_s`).
- **Tek kelimelik repliklerde WER ikili:** tek harf hatası WER = 1 (fail) yapıyor, CER 0,14. Türkçe ve kısa
  replikler için `qa.text_metric: cer` daha anlamlı.
- **Kısa klip embedding'i:** 0,4 sn'lik "Hmm." klibi QA paydasını çökertip benzerlik oranını 3'e çıkardı;
  baseline artık sadece ≥1,5 sn kliplerle hesaplanıyor.

### Gerçek oyun sesiyle test (GTA 6 ara sahnesi, ilk 60 sn)

- **Yerel 9B LLM çevirisi kullanılamaz kalitede:** küfürleri sansürledi ("Fuck" → "Kral"), cümleleri bozdu.
  → `translations.csv` (seg_id,text_tr) ile insan/elle çeviri her zaman öncelikli; fit bu satırları LLM
  kısaltmasına geri göndermez. Asıl çözüm: Claude (API anahtarı) ya da insan çevirmen.
- **Parça parça ASR bağlamı kaybediyor:** 0,5–0,8 sn'lik parçalarda "Change of plans" → "Trash reports",
  "the Dodo" → "Madoto". → `asr.file_context`: tüm dosya tek seferde yazılır, kelimeler en yakın segmente
  dağıtılır (katı pencere ilk kelimeleri kaçırıyordu).
- **Konuşmacı ayırma (HF token yok):** 15 parça 12 "konuşmacıya" bölündü. Perde ölçümü net ayrım
  gösterdi (kadın 177–258 Hz, erkek 110–140 Hz); `speakers.num_speakers` ipucu eklendi.
- **Ses bankası boyutu belirleyici:** tek 1,7 sn klipten klonlanan ses kötüydü → `bank_target_min_s`
  (kısa kliplerle ≥6 sn'ye tamamla). Yine de 1 dakikalık klipte karakter başına 3–8 sn gürültülü ses var;
  konuşmacı benzerliği 0,17–0,87 (eşik 0,85). **Daha uzun kaynak ya da karakterin temiz örnekleri gerekli.**
- **Duygu referansı:** Chatterbox adaylarının bir kısmı "orijinal replik + banka" referansıyla üretiliyor;
  QA skoruna duygu benzerliği eklendi (ağırlık 0,15). Kazananların 7/15'i bu adaylardan.
- **Windows dosya kilidi:** worker'ın her öğeden sonra sonucu yeniden yazması antivirüs kilidine çarptı
  (WinError 5) ve worker yarıda kaldı; orchestrator bunu "hata: 0" sanıyordu. Yazma tekrar denemeli ve
  seyrek, eksik öğeler artık hata olarak işaretleniyor.

- **XTTS-v2 eklendi (3. TTS, `envs/xtts`, CPML lisansı kullanıcı onayıyla):** GTA klibinde 3 modelden replik
  başına en iyi aday seçildiğinde kazananlar: Chatterbox 8, XTTS-v2 6, OmniVoice 1. XTTS süre sapınca
  `speed` ile bir kez yeniden üretiliyor. Windows notu: torch ≥2.9 ile coqui-tts torchcodec istiyor → ortam
  torch 2.6'ya sabitlendi.

- **VoxCPM2 eklendi (4. TTS, `envs/voxcpm`, Apache-2.0):** 6 GB kartta sığıyor (tepe 5,64 GB), 15 replik ~5 dk.
  `voxcpm` paketi bağımlılıksız kuruluyor (wetext → kaldifst Windows'ta derlenmiyor; Türkçede gereksiz).
  Duygu, metnin başına "(talimat)" ile veriliyor; `translations.csv`'nin `style` sütunu replik başına yönetmen
  notu taşıyor. Ortalama QA skorunda birinci (0,596 – diğerleri 0,51–0,57); ama tek kelimelik repliklerde
  telaffuz kayabiliyor ("Gidelim" → "Kitilim").
- **Sadece VoxCPM2, replik başına 3 aday (2 talimatlı + 1 talimatsız):** 15/15 replik doğru okunuyor, fail 0.
  Kısa repliklerde talimatsız aday, uzun/duygulu repliklerde talimatlı aday kazandı. Kısa (≤2 kelime)
  repliklerde durum/skor artık CER ile (`qa.short_ref_words`); WER tek kelimede ikili olduğu için doğru okunan
  "Biri mi?" ("BİRİMİ" diye yazıldı) fail sayılıyordu.

- **Doğal mod (tam GTA videosu, 42 replik; kullanıcı: "robotikleşiyor"):** Seçilenlerin 20/42'si Seed-VC'den
  geçmiş, 20 replik %8-15 esnetilmişti (VC length_adjust ±%20 ile çift süre oynaması). VoxCPM2 tınıyı zaten
  klonladığı için `vc.enabled: false`; `fit.max_slowdown: 0` (kısa dublaj yavaşlatılmaz, sessizlikle dolar),
  `fit.use_slot: true` (uzun aday sonraki repliğe kadar taşabilir, çarparsa en fazla %8 sıkıştırılır).
- **Türkçe devam istemi (vurgu/tonlama):** VoxCPM2'nin birleşik modu: `reference_wav_path` (karakter tınısı,
  yalıtılmış) + `prompt_wav_path`/`prompt_text` = aynı karakterin daha önce iyi çıkmış bir **Türkçe** repliği
  (duygu/yoğunluğu yakın olan seçilir). Model İngilizce referansın vurgusunu değil Türkçe konuşmanın akışını
  sürdürüyor. Ölçüm (aynı 42 replik): istemli adaylar 82'de 1 fail, ort. CER 0,009, süre sapması 0,26;
  talimatlı/talimatsız eski adaylar 138'de 44 fail, ort. CER 0,253, süre sapması 0,60. Seçilenlerin 25/42'si
  istemli; seçilen fail 0, esnetme 0. Eskiden her adayda bozuk okunan "Silah çekme" ("Sayıla çekme") düzeldi.
  QA'ya perde canlılığı (`prosody`: dublaj/orijinal perde sapması, tekdüze okuma cezalı) eklendi.

### İngilizce kalan yerler ve İngilizce aksan (GTA tam video, 2026-11-21)

- Kullanıcı: "bazı kelimeleri atlıyor, İngilizce kalmış". Ölçüm: ses kanalını Silero VAD'lı Whisper ile baştan
  yazdırıp dublajlı segmentlerle karşılaştırdık. 23 segment orijinal (İngilizce) sesle kalmıştı, 4 sessiz replik
  hiç segmentlenmemişti. Arka plan stem'inde konuşma sızıntısı yoktu (DnR temiz).
- Silero VAD'ı enerji tabanlı bölmeye "yalnızca eksik bölge ekle" kuralıyla eklemek bu videoda tam olarak kaçan
  4 yeri buldu, sahte segment eklemedi.
- Whisper dil tanıma (dil zorlanmadan) aynı metnin adayları arasında İngilizce aksanlı okumayı ayırıyor: CER'i 0
  olan adaylar arasında p(tr) %1 ile %98 arasında değişiyor ("Herkes yapıyor", "Biri mi?", "Hadi"). QA'ya
  `lang` ağırlığı (p(tr)/(p(tr)+p(en))) eklendi. Tek hecelik ünlemlerde ("Hey, Duval!", "Sürpriz!") tüm adaylar
  İngilizce görünüyor; orada ölçü gürültüdür ve seçimi değiştirmez.
- Doğal modda bir aday sonraki repliğin üstüne binebiliyordu ("Nihayet." 1,75 sn uzatılmış okuma 0,62 sn'lik
  boşluğa seçildi). `slot_overflow_s` ile ağır ceza eklendi.

### Arka plandaki konuşmalar: megafon, telsiz, uzaktan bağırma (GTA tam video, 2026-11-21)
Kullanıcı ilk bölümde polis anonslarının ve arka plandaki konuşmaların çevrilmediğini duydu. Whisper bu replikleri
orijinal miks ve arka plan (müzik+efekt) üzerinde net yazıyor (1:24 anons lp -0,32; 2:41 lp -0,34), ama:

| Denenen ayırma | Sonuç (Whisper, 80-92 sn ve 158-170 sn) |
|---|---|
| Bandit v2 DnR (mevcut) | Anons efekt stem'inde, 2:41 kısmen müzik stem'inde |
| Vocals Mel-RoFormer, efekt stem'inde | Anons "enstrümantal" tarafta kaldı |
| Vocals Mel-RoFormer, tüm arka planda | Aynı |
| İkinci DnR geçişi (arka plana) | Konuşma stem'i -60…-93 dB (boş) |
| ZipEnhancer (VoxCPM'in gürültü gidericisi) | Çıktı -73 dB: anonsu gürültü saydı |
| **Spektral kenetleme**, çevredeki medyan + 0 dB | Anons ve bağırmalar gitti (Whisper uydurmaya düştü, nsp 0,6-0,8); müzikle aynı seviyedeki kısa küfürler kaldı ("Oh, shit") |
| **Spektral kenetleme**, medyan − 6 dB (mix'te kullanılan) | Küfürler de gitti; son miksin işlenmiş arka planında güvenilir İngilizce konuşma kalmadı |

Spektral kenetleme ayrıştırma değil, bölgesel bir bastırma: konuşmanın bulunduğu 2-4 sn'de her frekans bandı,
bölgenin ±1,5 sn çevresindeki arka planın medyan seviyesine kırpılıyor (faz korunuyor, 120 Hz-7 kHz). Müzik/efekt
dokusu kalıyor, ondan yükselen konuşma harmonikleri gidiyor. Tespit için tek geçişli Whisper kısa küfürleri
kaçırdı ("Shit." 1:02); 10 sn pencere / 5 sn adım hepsini buldu, "Thanks for watching" uydurmaları no_speech /
logprob / halüsinasyon listesiyle elendi. Tek kelimelik ünlemlerde Whisper'ın kelime zamanları 1-2 sn kayıyordu
("Fuck!" 65,0 dendi, gerçekte ~67): bulunan her replik ±1 sn'lik pencerede yeniden yazdırılıyor ve ilk kelimenin
süresi harf sayısıyla sınırlanıyor (DTW önceki müziği ilk kelimeye katıyor). "Kahretsin!" VoxCPM2'de 5 adayda da
doğru okunmadı (CER ≥ 0,56); "Lanet olsun!" 5/5 doğru. Dublajın tınısı temiz bir karakter bankasından, sahnedeki rengi (megafon
bant sınırı) orijinalin uzun dönem spektrumundan aktarılıyor (`fit.bg_match_spectrum`).

### Kısık sesler, megafon ve telefon efekti (GTA tam video, 2026-11-21)
Kullanıcı: "bazı kısık sesli şeyleri algılayamıyor", "polisler megafonla bağırıyor, aynı efekti veremiyor",
"telefonla konuşmanın ses efekti olmamış".

**Kaçan replikleri bulmak.** Videodaki gömülü altyazı "gerçek replik listesi" olarak kullanıldı: altyazı bandı
4 fps örneklendi, metin pikseli sayısı (parlak + yerel kontrast) ile altyazılı aralıklar çıkarıldı ve satırlarla
karşılaştırıldı. Altyazılı ama satırı olmayan tek replik 8:19 "Huh?" çıktı (0,24 sn; `min_seg_s` 0,3 sn ve tek
geçişli Silero kaçırıyordu). Altyazısız adaylar farklı pencere başlangıç/uzunluklarında yeniden yazdırıldı:
tutarlı çıkan tek gerçek kelime 3:16 "jump" (orijinalde p 0,7-0,98) idi; o da sızıntı sayılıp bastırılmıştı.
Gevşek eşikli tarama çoğunlukla "Thanks for watching!" uydurdu (müzikli 4-8. dakikada 30+ isabet).

**Yanlış alarm (not).** Mikste kalan İngilizceyi bulmak için miksi İngilizce zorlamalı Whisper'a vermek işe
yaramıyor: Whisper Türkçe konuşmayı İngilizceye *çeviriyor* ("suspects are in the area", "of course"). Miks −
dublaj farkı da limiter/yeniden örnekleme yüzünden Türkçe kalıntı taşıyor. Doğru test ham arka plan katmanında:
dublajlanan diyalogların 229 İngilizce kelimesinden 1'i tanındı (diyalog sızıntısı yok denecek kadar az).

**Kanal profili.** Orijinal konuşmada 60-250 Hz ve 4,5-6,5 kHz bantlarının 400-3000 Hz'e göre seviyesi (dB, Hz başına):

| Satır | alt | üst | Ne |
|---|---|---|---|
| Normal diyalog (çoğunluk) | +5…+22 | −3…−29 | tam bant |
| s55 "Surprise!" (7:42, cezaevi telefonu) | −24,2 | −34,7 | telefon |
| s58 "Hey, what's up, guy?" (8:31) | −9,6 | −31,8 | cihaz hoparlörü |
| b03/b05/b06 polis (1:24, 2:41, 2:46) | −3…−9 | −27…−40 | megafon (üst kenar 3,6-4 kHz) |
| s07 "Unit 2, this is Command" (telsize konuşan adam) | +6,3 | −11,6 | tam bant, efekt yok |

Spektrum eşleme (match_spectrum) yalnızca bu ortalama tınıyı aktarıyordu; megafonu megafon yapan distorsiyon,
sıkıştırma ve dış mekân yankısı, telefonu telefon yapan codec pürüzü eksikti. `fx` modülü dublaj stüdyolarının
"futz" zincirlerini uygular: telefon = bant geçiren + hafif sürme + μ-law (orijinal 4 kHz üstünü tamamen
kesiyorsa 8 kHz'e inerek) + kompresör; megafon = bant geçiren + sert tanh sürme (+14 dB) + kompresör +
0,14/0,29 sn yansıma ve 0,8 sn dağınık kuyruk (orijinal anonsların zarf öz ilintisinde 0,145 sn'de 0,48'lik
tepe vardı); sonda orijinalin tınısı match_spectrum ile aktarılır. Ölçüm (QA'nın seçtiği dublaj, alt/üst dB,
parantezde orijinal): telefon s55 −31,9/−34,1 (−24,2/−34,7), s58 −8,4/−32,3 (−9,6/−31,8); megafon b03
−14,9/−40,0 (−3,3/−39,5), b05 −3,6/−31,8 (−3,0/−31,1), b06 −17,9/−28,0 (−9,1/−26,9). Üst bant her satırda
1 dB içinde; megafonda alt bant orijinalden daha kesik çıkıyor, ama arka plan satırlarında orijinalin alt bant
ölçümü zaten güvenilmez (müzik bastırılırken konuşma tahmininden de gidiyor). İlk denemede telefon zinciri her
zaman 8 kHz'e iniyordu, üst bant −97 dB'e düştü; GTA'nın telefon sesi o kadar dar değil, bu yüzden 8 kHz'e
yalnızca orijinal gerçekten dar bantsa iniliyor. Efektli mikste Whisper "Araçtan inin! Eller havaya!"yı
"Araçlar ileri eller havaya", "Sürpriz!"i ve "Hey, naber dostum? Çekil yolumdan!"ı doğru yazıyor; 3:16'da
"3, 2, 1. Atla." duyuluyor, işlenmiş arka planda İngilizce "jump" kalmadı.

**Uygulanmayan (ileride):** öğrenilmiş efekt eşleme. DeepAFx-ST (Steinmetz ve ark., 2022) ve ST-ITO
(Steinmetz ve ark., ISMIR 2024) bir referans kliğin efekt zincirini (EQ, kompresör, distorsiyon, yankı)
parametre aramasıyla taklit ediyor; distorsiyon ve yankı miktarını da orijinalden öğrenmek için doğal sonraki adım.

## 3. Henüz uygulanmayan öneriler (öncelik sırasıyla)

1. **Ayrı stem'leri kullan:** Oyun VO/müzik/SFX'i ayrı dosyalarda veriyorsa (Wwise bank'ları) ayrıştırmayı
   tamamen atlayın (`separation.mode: never`); hiçbir model orijinal stem kadar temiz değildir. DnR'nin müzik ve
   efekt stem'leri ayrı kaydediliyor (`music_path`, `sfx_path`): ileride miks'te diyalog altında müzik "ducking"
   için kullanılabilir.
2. **QA konuşmacı benzerliği için ikinci model:** TTS makalelerinin standardı WavLM-large SV (Seed-TTS-eval,
   SIM-o). Diller arası (EN referans → TR çıktı) kosinüs ~0,08–0,10 düşer; bu yüzden mutlak eşik değil
   baseline oranı kullanılıyor (mevcut tasarım doğru).
3. **Duygu yoğunluğu için boyutsal model:** `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
   (arousal/valence, CC BY-NC-SA) ile intensity = arousal; QA'da |Δarousal|.
4. **Konuşma hızı kalibrasyonu:** `tr_syllables_per_sec` sabit 6,0. TTS çıktılarından karakter/duygu bazında
   gerçek hece/sn ölçülüp bir sonraki çeviride kullanılabilir.
5. **Komşu boşluğu ödünç alma (SoniTranslate):** taşan replik bir sonraki repliğe kadar olan sessizliğe
   uzayabilsin; manifest'te `slot_dur` hazır.
6. **Doğallık metrikleri:** UTMOSv2 / Audiobox Aesthetics sadece adayları birbirine göre sıralamak için
   (Türkçe için mutlak eşik olarak doğrulanmadı).
7. **Türkçe sayı normalizasyonu:** `trnorm` (GitHub) ile WER/CER öncesi; şimdilik LLM'den sayıları yazıyla
   yazması isteniyor.
8. **Ticari karşılaştırma tabanı:** ElevenLabs v3/v4 (Türkçe, ses etiketleri, Dubbing API ~2,20 $/dk),
   MiniMax Speech 2.8 (Türkçe `language_boost`). İkisinde de süre hedefi yok.
9. **Diğer açık TTS adayları:** XTTS-v2 (`coqui-tts` 0.28, Türkçe dinleme testinde en iyi MOS 3,82,
   2,2 GB VRAM), VoxCPM2 (metin talimatıyla duygu, 6 GB'a zor sığar), MOS-TTS-Local (token=süre).
   IndexTTS2, Qwen3-TTS, CosyVoice3: **Türkçe yok**.

## 4. Donanım notu (RTX 4050 Laptop, 6 GB)

Spec ≥ 8 GB varsayıyor. Her worker tek model yükleyip kapandığı için 6 GB yetiyor: Chatterbox ~3,5–4 GB,
faster-whisper large-v3 fp16 ~3 GB (sorun olursa `asr.compute_type: int8_float16`), Seed-VC küçük model,
OmniVoice fp16. Ollama ile çeviri yapılıyorsa model işten sonra boşaltılır (`keep_alive: 0`).

## Kaynaklar

- Chatterbox V3: https://www.resemble.ai/resources/chatterbox-multilingual-v3-tts-with-embedded-watermarking-for-25-languages · https://github.com/resemble-ai/chatterbox · https://pypi.org/project/chatterbox-tts/
- OmniVoice: https://github.com/k2-fsa/OmniVoice · https://arxiv.org/html/2604.00688v2
- Türkçe TTS karşılaştırması (XTTS, F5, VoxCPM): https://arxiv.org/html/2607.09530
- Seed-VC (arşiv) ve değerlendirme: https://github.com/Plachtaa/seed-vc · https://raw.githubusercontent.com/Plachtaa/seed-vc/main/EVAL.md
- MeanVC: https://arxiv.org/html/2510.08392v2
- audio-separator: https://github.com/nomadkaraoke/python-audio-separator · MVSEP DnR: https://mvsep.com/algorithms/64 · Bandit v2: https://zenodo.org/records/12701995
- pyannote community-1: https://huggingface.co/pyannote/speaker-diarization-community-1 · DiariZen: arXiv 2509.26177
- Seed-TTS-eval (SIM-o): https://github.com/BytedanceSpeech/seed-tts-eval
- Türkçe WER/CER: arXiv 2410.07400 · trnorm: https://github.com/ysdede/trnorm
- Duygu metrikleri: "The False Resonance" arXiv 2604.26347 · audeering msp-dim: https://huggingface.co/audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
- TTSDS2 (metrik–insan korelasyonu): arXiv 2506.19441
- İzokroni: https://aclanthology.org/2025.emnlp-demos.37 · IWSLT 2025 arXiv 2506.04855
- ElevenLabs: https://elevenlabs.io/docs/overview/models · https://elevenlabs.io/docs/capabilities/dubbing
- SoniTranslate / pyVideoTrans / open-dubbing / VideoLingo: ilgili GitHub depoları
