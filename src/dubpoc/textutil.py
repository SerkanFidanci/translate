"""Türkçe metin yardımcıları: hece sayımı ve WER/CER normalizasyonu."""
from __future__ import annotations

import re
import unicodedata

# Spec: Türkçe hece = sesli harf sayısı. Şapkalı ünlüler (â, î, û) de hece çekirdeğidir.
TR_VOWELS = set("aeıioöuüAEIİOÖUÜâîûÂÎÛ")

# İngilizce/genel ünlem ve efor sesleri: bunlardan oluşan replikler çevrilmez (keep_original).
INTERJECTIONS = {
    "uh", "um", "umm", "hmm", "hm", "mm", "mmm", "mhm", "ah", "aah", "ahh", "oh", "ooh", "huh",
    "ugh", "argh", "agh", "gah", "ha", "haha", "hahaha", "heh", "hehe", "pfft", "phew", "tsk",
    "grr", "grunt", "grunts", "sigh", "sighs", "gasp", "gasps", "laugh", "laughs", "laughing",
    "groan", "groans", "pant", "panting", "breath", "breathing", "scream", "screams", "hah", "hngh", "nngh",
}


def count_syllables_tr(text: str) -> int:
    return sum(1 for ch in text if ch in TR_VOWELS)


def tr_lower(text: str) -> str:
    """Python lower() 'I'→'i' yapar; Türkçede 'I'→'ı', 'İ'→'i' olmalı."""
    return text.replace("I", "ı").replace("İ", "i").lower()


def tr_upper(text: str) -> str:
    return text.replace("i", "İ").replace("ı", "I").upper()


def fix_caps_tr(text: str) -> str:
    """TTS için büyük harf normalizasyonu. ASR bağırılan repliği "SAFE" diye yazınca LLM de "GÜVENLİ" döndürür;
    TTS büyük harfli metni yanlış okuyabiliyor (ölçüldü: "Güvenliş"). Tüm satır büyük harfse cümle düzenine,
    satır içinde 4+ harfli tamamen büyük kelimeler küçük harfe çevrilir (≤3 harfli kısaltmalar korunur)."""
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        low = tr_lower(text)
        # her cümlenin ilk harfini büyüt
        return re.sub(r"(^|[.!?…]\s+)(\w)", lambda m: m.group(1) + tr_upper(m.group(2)), low)
    return re.sub(r"\b[^\W\d_]{4,}\b", lambda m: tr_lower(m.group(0)) if m.group(0).isupper() else m.group(0), text)


_ONES = ["", "bir", "iki", "üç", "dört", "beş", "altı", "yedi", "sekiz", "dokuz"]
_TENS = ["", "on", "yirmi", "otuz", "kırk", "elli", "altmış", "yetmiş", "seksen", "doksan"]


def tr_num2words(n: int) -> str:
    """0..999 999 999 arası tam sayıyı Türkçe yazıya çevirir (boşluklu): 2025 → 'iki bin yirmi beş'."""
    if n == 0:
        return "sıfır"

    def below_1000(k: int) -> list[str]:
        h, rest = divmod(k, 100)
        t, o = divmod(rest, 10)
        out = []
        if h:
            out += ([] if h == 1 else [_ONES[h]]) + ["yüz"]
        out += [w for w in (_TENS[t], _ONES[o]) if w]
        return out

    words: list[str] = []
    for value, name in ((1_000_000, "milyon"), (1000, "bin")):
        q, n = divmod(n, value)
        if q:
            words += ([] if (q == 1 and name == "bin") else below_1000(q)) + [name]
    words += below_1000(n)
    return " ".join(words)


def tr_ordinal(n: int) -> str:
    """2 → 'ikinci', 4 → 'dördüncü', 10 → 'onuncu' (ünlü uyumu + t→d yumuşaması)."""
    words = tr_num2words(n).split()
    last = words[-1]
    if last == "dört":
        last = "dörd"
    vowels = [c for c in last if c in "aeıioöuü"]
    v = vowels[-1] if vowels else "i"
    suffix_v = {"a": "ı", "ı": "ı", "e": "i", "i": "i", "o": "u", "u": "u", "ö": "ü", "ü": "ü"}[v]
    suffix = ("nc" if last[-1] in "aeıioöuü" else suffix_v + "nc") + suffix_v
    return " ".join(words[:-1] + [last + suffix])


def speakable_tr(text: str) -> str:
    """TTS ve QA için okunabilir metin: büyük harf düzeltme + rakamları yazıya çevirme.
    "2. Birim" → "İkinci birim" (sıra sayısı: rakam + nokta + sonrasında kelime)."""
    t = fix_caps_tr(text)
    t = re.sub(r"\b(\d{1,6})\.(?=\s+\w)", lambda m: tr_ordinal(int(m.group(1))), t)
    t = expand_numbers_tr(t)
    return tr_upper(t[:1]) + t[1:] if t and t[0].islower() and text[:1].isdigit() else t


def expand_numbers_tr(text: str) -> str:
    """WER/CER için rakamları yazıya çevir (ASR "2" yazar, metin "iki" der). Basit tam sayılar."""
    def repl(m: re.Match) -> str:
        digits = m.group(0).replace(".", "")
        return tr_num2words(int(digits)) if len(digits) <= 9 else m.group(0)
    return re.sub(r"\d{1,3}(?:\.\d{3})+|\d+", repl, text)


def normalize_tr(text: str) -> str:
    """WER/CER için: rakam→yazı, Türkçe küçük harf, noktalama sil, boşlukları sadeleştir."""
    t = unicodedata.normalize("NFC", text)
    t = expand_numbers_tr(t)
    t = tr_lower(t)
    t = t.translate(str.maketrans({"â": "a", "î": "i", "û": "u"}))
    t = t.replace("’", "'").replace("‘", "'")
    # kesme işareti ('Ankara'ya) kelimeyi böler; birleştir
    t = t.replace("'", "")
    t = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def normalize_en(text: str) -> str:
    t = text.lower()
    t = "".join(ch if not unicodedata.category(ch).startswith(("P", "S")) else " " for ch in t)
    return re.sub(r"\s+", " ", t).strip()


def is_nonverbal_text(text: str) -> bool:
    """Boş, köşeli/parantezli açıklama ([laughs], (grunts), *sigh*) ya da sadece ünlemlerden oluşan metin."""
    t = text.strip()
    if not t:
        return True
    stripped = re.sub(r"[\[\(\*][^\]\)\*]*[\]\)\*]", " ", t)
    words = normalize_en(stripped).split()
    if not words:
        return True
    return all(w in INTERJECTIONS or re.fullmatch(r"(h+[aeiu]+)+h*|a+h+|o+h+|u+h+|m+h*m+", w) for w in words)


def word_count(text: str) -> int:
    return len(normalize_en(text).split())
