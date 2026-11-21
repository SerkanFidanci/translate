# Bilinen sorunlar ve sınırlamalar

## Ortam
- **6 GB VRAM** (spec ≥ 8 GB): modeller tek tek yüklendiği için çalışıyor. Bellek hatasında
  `asr.compute_type: int8_float16`, `tts.n_candidates` / `tts.omnivoice.n_candidates` düşürülebilir.
- `uv venv --allow-existing` Windows'ta mevcut `python.exe` varken "Dosya var (os error 80)" verir;
  kurulum scripti bu yüzden var olan venv'i yeniden oluşturmaz.
- Windows PowerShell 5.1, BOM'suz `.ps1` dosyalarını ANSI okur → kurulum scripti bilerek ASCII.
- İlk çalıştırmada model indirmeleri uzun sürer (Chatterbox ~3 GB, OmniVoice, Whisper large-v3 ~3 GB,
  emotion2vec, RoFormer). İndirmeler `%USERPROFILE%\.cache\huggingface` altına gider.

## Kalite
- **Sentetik test seti** (Windows SAPI sesi) gerçek oyun sesini temsil etmez. Müzik vocals modeli (eski
  `engine: audio_separator`) bu sesi "enstrüman" sanıyordu; varsayılan Bandit v2 DnR motoru bunu çözdü
  (konuşma SDR 20,5 dB). Mutlak QA sayıları yine de gerçek seslerle doğrulanmalı.
- **Bandit v2 DnR:** DnR v3 müzik stem'inden vokali çıkarır; şarkılı müzik/koro **diyalog** stem'ine düşebilir
  (dublajlanmaya çalışılır). Böyle sahnelerde `dnr_refine_vocals: true` ya da `subtitles.csv` ile kontrol edin.
  48 kHz çalışır; giriş otomatik yeniden örneklenir.
- **HF token yoksa** pyannote yerine duraklama tabanlı bölme kullanılır; tek replik içindeki uzun
  duraklamalar repliği bölebilir (test sahnesinde "Safe. For now." iki parçaya ayrıldı), üst üste konuşma
  ayrılamaz. Klasörsüz replikler klasör karakterlerine "çapa" ile (tek tek ya da küme merkeziyle) bağlanır;
  kısa (<1 sn) parçaların embedding'i zayıftır.
- **OmniVoice referans kalitesine hassas:** kötü ayrışmış/gürültülü referansla anlamsız çıktı verebiliyor; sıkı
  hedef süre bazı Türkçe repliklerde heceleri yutuyor. QA bu adayları eler, ama temiz bir ses bankası şart.
- QA'daki Türkçe Whisper bozuk sese "Altyazı M.K." gibi halüsinasyon üretebilir (WER = 1 → aday elenir).
- **Duygu tanıma** (emotion2vec) sentetik/monoton seste güvenilmez (ör. nötr okunan bir replik "sad").
- **Yerel Ollama çevirisi** (qwen3.5:9b) hece bütçesine uyuyor ama kalite Claude'un gerisinde
  ("Get down" → "Yere inin"). Varsayılan yapılandırma Claude Sonnet 5.5'tir.
- Whisper halüsinasyonları ("Thank you for watching") yakalanır; miks üzerinde tekrar denemeyle düzelmezse
  segment hata olarak işaretlenir ve miks orijinal sesi kullanır. Düzeltmek için `subtitles.csv`.
- QA WER/CER, Whisper'ın Türkçe çıktısına dayanır. Tam sayılar yazıya çevrilerek karşılaştırılır; sıra sayıları,
  ondalıklar, tarihler ve kısaltmalar normalize edilmez (LLM'den bunları yazıyla yazması istenir; tam çözüm: trnorm).
- **VoxCPM2 Türkçe vurgu:** İngilizce referansla klonlarken vurgu/tonlama İngilizceye kayabiliyor; Türkçe devam
  istemi (`tts.voxcpm.tr_prompt_candidates`) bunu büyük ölçüde düzeltiyor ama istem havuzu ilk turda üretilmiş
  iyi Türkçe repliklere dayanır (ilk tur olmadan istemli aday üretilemez). Kelime vurgusunu doğrudan ölçen bir
  metrik yok; `prosody` yalnızca tekdüzeliği yakalar, son karar kulakla verilmeli.
- **Doğal mod (`fit.use_slot`)** dudak/sahne senkronunu biraz gevşetir: dublaj orijinalden kısa ya da (sonraki
  repliğe çarpmadan) uzun kalabilir.
- **İngilizce kalan replikler (düzeltildi, 2026-11-21):** dublajı olmayan segmentte mix orijinal sesi çalar.
  GTA tam videosunda 23 segment bu yüzden İngilizce kalmıştı. Sebepler: (1) HF token yokken duraklama tabanlı
  bölmede kısa replikler ("Fuck.", "Take it.") skip ediliyordu, (2) duygu modeli tek kelimelik sözcükleri "efor"
  sanıyordu, (3) 9 dk'lık dosyada Whisper müzikli bölümlerde halüsinasyon döngüsüne giriyordu, (4) göreli enerji
  eşiğinin altındaki sessiz replikler ("Come on.") hiç segmentlenmiyordu (dublajda sessiz kalıyordu). Hepsi
  kodda düzeltildi (diarize: kısa segment korunur + Silero VAD; emotion: sadece gerçek ünlemler keep_original;
  asr: `file_vad`; segmente düşmeyen kelimeler `ctx.notes`'a uyarı). Kalan: belirsiz sesler ("Yeah?" p=0,27)
  ve ünlemler bilerek orijinal bırakılır; konuşmacı karışırsa `speakers.yaml` overrides ile düzeltin.
- **Arka plandaki konuşmalar (düzeltildi, 2026-11-21):** polis megafon anonsu, uzaktan bağırmalar ve bazı
  küfürler DnR'de "efekt/müzik" stem'ine düşüyor, dublaj sadece konuşma stem'ini çevirdiği için arka planda
  İngilizce kalıyordu (GTA: 1:24 "Get out of the vehicle...", 2:41 "Put your hands behind your head..."). Vocals
  RoFormer (efekt stem'inde ve tüm arka planda), ikinci DnR geçişi ve ZipEnhancer gürültü giderici bunları
  ayıramadı. Çözüm: arka plan 10 sn'lik örtüşen pencerelerle Whisper'la taranıyor, güvenilir konuşmalar ayrı
  "bg" katmanı satırları (`<file>_bNN`) olarak dublajlanıyor ve orijinal, spektral kenetlemeyle arka plandan
  bastırılıyor (Whisper bastırılmış sesi artık anlayamıyor). Sınırlar: (1) bastırılan 2-4 sn'de müzik/efekt orta
  frekanslarda biraz matlaşabilir; (2) kenetleme sadece çevresinden yüksek konuşmayı alır, müzikle aynı
  seviyedeki fısıltı/kalabalık uğultusu kalır; (3) bu kliplerden konuşmacı tanıma çalışmıyor (benzerlik ~0),
  varsayılan olarak kendi kümelerine (Arka1, ...) gürültülü klipten klonlanır, daha temiz ses için
  `speakers.yaml` overrides ile uygun bir karaktere bağlayın; (4) Whisper'ın arka plan kelime zamanları
  0,5-1 sn kayabildiğinden, yakındaki diyalogla aynı sözleri taşıyanlar sızıntı sayılır ve dublajlanmaz.
- **Kısa/kısık replikler (düzeltildi, 2026-11-21):** 8:19'daki "Huh?" (0,24 sn) enerji eşiğini geçtiği hâlde
  `min_seg_s` (0,3 sn) yüzünden atılıyordu; tüm dosyada tek geçişte çalışan Silero VAD da uzun sessizlikten sonra
  onu kaçırıyordu → mikste hiç duyulmuyordu (mix yalnızca satırları çalar). 3:16'daki "jump" ise arka plandaki
  "Two, one, jump." bütünüyle "Three, two, one." diyaloğunun sızıntısı sayıldığı için bastırılıp hiç
  dublajlanmamıştı. Düzeltme: VAD 20 sn'lik örtüşen parçalarda, 0,15-0,3 sn'lik kısa ünlemler de parça; sızıntıda
  diyalogda olmayan baş/son kelimeler ayrı satır. Bilerek orijinal bırakılanlar: Whisper'ın farklı pencerelerde
  tutarlı kelime bulamadığı iç çekme/mırıltılar (5:23, 7:34), düşük güvenli kalabalık "Yeah!" (5:10, p 0,3-0,5).
- **Kanal efektleri (telefon, megafon, telsiz; 2026-11-21):** zincirler sabit ön ayarlardır (dublaj stüdyolarının
  "futz" eklentileri gibi); bant kenarları ve son tını orijinalden ölçülür, ama distorsiyon miktarı ve yankı
  (megafon: 0,14/0,29 sn yansıma + 0,8 sn kuyruk) ölçülmez. Otomatik tespit sadece diyalogdaki telefon/cihaz
  sesini bulur (alçak ve tiz birlikte kesik); megafon/telsiz `translations.csv` `fx` sütunuyla ya da style notunda
  geçen kelimeyle (orijinal gerçekten dar bantsa) seçilir. Arka plan satırlarının ses tahmini gürültülü olduğundan
  bu satırlarda alt bant ölçümü güvenilmez. Son karar kulakla verilmeli.
- **İki stem'e bölünen replik (düzeltildi, 2026-11-21):** DnR bir repliğin bir kısmını efekt/müzik stem'ine
  koyabiliyor (007: "So you shouldn't have."). Dublaj seviyesi artık iki kısmın toplamına eşleniyor
  (`fit.leak_loudness`, varsayılan açık); arka plan kısmı kenetlemeyle tahmin edildiği için ±2-3 dB sapabilir.
- **İngilizce kalıntı ölçümü:** Türkçe miks Whisper'a İngilizce zorlanarak verilirse Whisper Türkçeyi İngilizceye
  çeviriyor ("Şüpheliler bölgede" → "suspects are in the area"); kalıntı testi miks üzerinde değil, dublajsız arka
  plan katmanında yapılmalı.

## Kapsam dışı (spec 9)
- `.wem/.bnk` geri paketleme (`dubpoc export` sadece TODO stub'ı), karakter başına RVC eğitimi, dudak senkronu,
  web arayüzü.
