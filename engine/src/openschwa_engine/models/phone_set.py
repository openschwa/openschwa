"""The canonical phone inventory and per-model mapping tables.

`docs/architecture.md` calls this out as a top risk: a wrong mapping does not
crash, it quietly scores the wrong segment. So the mapping is an explicit,
committed table with round-trip tests rather than string munging at the call
site.

Two distinct alphabets meet here:

* **Canonical** — what exercise packs author and what the API returns. A broad
  General American inventory, IPA, no length marks, no stress marks.
* **Model tokens** — whatever the acoustic model's vocabulary happens to use.
  These are *not* interchangeable with canonical labels. The espeak model is
  multilingual and spells English long vowels with a length mark (`iː`, `uː`,
  `ɑː`), writes NURSE as `ɜː` with no `ɝ` in the vocabulary at all, and uses
  U+0261 ɡ (LATIN SMALL LETTER SCRIPT G) rather than ASCII `g`. Mapping
  `i` → `i` instead of `i` → `iː` would align against a token the model never
  emits for English.

Adding a model to the M1 bake-off means adding a table here and a
round-trip test — never touching pipeline code.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

#: Canonical General American inventory. Exercise packs may only use these
#: labels; the content loader rejects anything else at startup.
CANONICAL_EN: frozenset[str] = frozenset(
    {
        # obstruents
        "p",
        "b",
        "t",
        "d",
        "k",
        "ɡ",
        "tʃ",
        "dʒ",
        "f",
        "v",
        "θ",
        "ð",
        "s",
        "z",
        "ʃ",
        "ʒ",
        "h",
        # sonorants
        "m",
        "n",
        "ŋ",
        "l",
        "ɹ",
        "w",
        "j",
        # monophthongs
        "i",
        "ɪ",
        "ɛ",
        "æ",
        "ə",
        "ʌ",
        "ɑ",
        "ɔ",
        "ʊ",
        "u",
        # r-coloured vowels
        "ɚ",
        "ɝ",
        # diphthongs
        "eɪ",
        "aɪ",
        "ɔɪ",
        "oʊ",
        "aʊ",
        # vowel + /ɹ/ sequences that the model emits as single units
        "ɑɹ",
        "ɔɹ",
        "ɛɹ",
        "ɪɹ",
        "ʊɹ",
    }
)

#: Spellings an author may reasonably type, normalised to canonical form.
#: ASCII `g`/`r` are the common ones — they look identical to ɡ/ɹ in most fonts.
_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        "g": "ɡ",  # U+0067 -> U+0261
        "r": "ɹ",
        "y": "j",
        "ʧ": "tʃ",
        "ʤ": "dʒ",
        "ɜ": "ɝ",
        "ɜː": "ɝ",
        "ɐ": "ə",
        "e": "ɛ",
        "o": "ɔ",
        "a": "ɑ",
    }
)

_STRIPPED = ("ˈ", "ˌ", "ː", ".", "/", " ")


class PhoneSetError(ValueError):
    """A phone cannot be represented — raised at load, never mid-analysis."""


def normalize(phone: str) -> str:
    """Canonicalise an authored phone label, or raise.

    Stress and length marks are stripped first: `ˈiː` and `i` are the same
    canonical phone, and packs should not have to know which the model wants.
    """
    cleaned = phone.strip()
    for mark in _STRIPPED:
        cleaned = cleaned.replace(mark, "")
    if not cleaned:
        raise PhoneSetError(f"empty phone label (from {phone!r})")
    if cleaned in CANONICAL_EN:
        return cleaned
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    raise PhoneSetError(
        f"/{phone}/ is not in the canonical inventory — add it to CANONICAL_EN and to "
        "every model table in phone_set.py, or fix the exercise"
    )


#: Canonical -> espeak vocabulary token for `facebook/wav2vec2-lv-60-espeak-cv-ft`.
#: Every value is checked against the committed vocabulary snapshot by the
#: round-trip tests, and against the downloaded vocabulary at model load.
ESPEAK_EN: Mapping[str, str] = MappingProxyType(
    {
        "p": "p",
        "b": "b",
        "t": "t",
        "d": "d",
        "k": "k",
        "ɡ": "ɡ",
        "tʃ": "tʃ",
        "dʒ": "dʒ",
        "f": "f",
        "v": "v",
        "θ": "θ",
        "ð": "ð",
        "s": "s",
        "z": "z",
        "ʃ": "ʃ",
        "ʒ": "ʒ",
        "h": "h",
        "m": "m",
        "n": "n",
        "ŋ": "ŋ",
        "l": "l",
        "ɹ": "ɹ",
        "w": "w",
        "j": "j",
        # English long vowels carry the length mark in this vocabulary; the bare
        # tokens exist only because the model is multilingual.
        "i": "iː",
        "u": "uː",
        "ɑ": "ɑː",
        "ɔ": "ɔː",
        "ɪ": "ɪ",
        "ɛ": "ɛ",
        "æ": "æ",
        "ə": "ə",
        "ʌ": "ʌ",
        "ʊ": "ʊ",
        "ɚ": "ɚ",
        "ɝ": "ɜː",  # no ɝ in the vocabulary — espeak spells NURSE as ɜː
        "eɪ": "eɪ",
        "aɪ": "aɪ",
        "ɔɪ": "ɔɪ",
        "oʊ": "oʊ",
        "aʊ": "aʊ",
        "ɑɹ": "ɑːɹ",
        "ɔɹ": "ɔːɹ",
        "ɛɹ": "ɛɹ",
        "ɪɹ": "ɪɹ",
        "ʊɹ": "ʊɹ",
    }
)

#: Canonical -> charsiu token for charsiu/en_w2v2_ctc_libris_and_cv (M1 bake-off
#: candidate). The vocabulary is stressless ARPABET: 39 phones, no schwa, no
#: r-coloured vowel, no vowel+r digraphs. The lossy entries below (ə/ʌ -> AH,
#: ɚ/ɝ -> ER, sequences -> their first phone) exist so that alignment of any
#: utterance works; contrast scoring is only ever asked about phones whose
#: tokens are unique, which is what `required` pins in PhoneMap.build.
CHARSIU_EN: Mapping[str, str] = MappingProxyType(
    {
        "p": "P",
        "b": "B",
        "t": "T",
        "d": "D",
        "k": "K",
        "ɡ": "G",
        "tʃ": "CH",
        "dʒ": "JH",
        "f": "F",
        "v": "V",
        "θ": "TH",
        "ð": "DH",
        "s": "S",
        "z": "Z",
        "ʃ": "SH",
        "ʒ": "ZH",
        "h": "HH",
        "m": "M",
        "n": "N",
        "ŋ": "NG",
        "l": "L",
        "ɹ": "R",
        "w": "W",
        "j": "Y",
        "i": "IY",
        "ɪ": "IH",
        "ɛ": "EH",
        "æ": "AE",
        "ə": "AH",  # no schwa token: best-effort lossy
        "ʌ": "AH",
        "ɑ": "AA",
        "ɔ": "AO",
        "ʊ": "UH",
        "u": "UW",
        "ɚ": "ER",  # no r-coloured tokens: best-effort lossy
        "ɝ": "ER",
        "eɪ": "EY",
        "aɪ": "AY",
        "ɔɪ": "OY",
        "oʊ": "OW",
        "aʊ": "AW",
        "ɑɹ": "AA",  # sequences: best-effort lossy, first phone only
        "ɔɹ": "AO",
        "ɛɹ": "EH",
        "ɪɹ": "IH",
        "ʊɹ": "UH",
    }
)

#: Canonical -> token for the Option 3 contrast judge (dh-contrast-v1). Its
#: vocabulary is exactly {blank, unk, ð, z, d, v}, so only the drilled set maps.
DHZ_EN: Mapping[str, str] = MappingProxyType({"ð": "ð", "z": "z", "d": "d", "v": "v"})

#: Open-set variant (Stage 3): every non-ð/z/d realization folds into a single
#: "other" class, which exists only in the model's vocabulary - authored
#: content never names it (CANONICAL_EN is untouched).
DHZ_OPEN_EN: Mapping[str, str] = MappingProxyType(
    {"ð": "ð", "z": "z", "d": "d", "other": "other"}
)

#: Table name -> mapping. registry.MANIFEST names one of these per model.
TABLES: Mapping[str, Mapping[str, str]] = MappingProxyType(
    {
        "espeak_en": ESPEAK_EN,
        "charsiu_en": CHARSIU_EN,
        "dhz_en": DHZ_EN,
        "dhz_open_en": DHZ_OPEN_EN,
    }
)

#: The phones a model is allowed to *discriminate*: its tokens for these must
#: be unique. Alignment may be lossy outside this set; contrast scoring never is.
REQUIRED: Mapping[str, frozenset[str]] = MappingProxyType(
    {
        "espeak_en": CANONICAL_EN,
        "charsiu_en": frozenset({"ð", "z", "d", "v"}),
        "dhz_en": frozenset({"ð", "z", "d", "v"}),
        "dhz_open_en": frozenset({"ð", "z", "d", "other"}),
    }
)

#: Blank token spellings, in preference order (espeak uses '<pad>', charsiu '[PAD]').
_BLANK_CANDIDATES = ("<pad>", "[PAD]", "pad")


@dataclass(frozen=True)
class PhoneMap:
    """A validated canonical <-> model-token <-> vocabulary-index mapping."""

    model_id: str
    table_name: str
    token_of: Mapping[str, str]
    index_of: Mapping[str, int]
    blank_index: int
    #: Phones whose tokens must be unique: closed-set contrast scoring only ever
    #: runs inside this set. Outside it the table may be lossy for alignment.
    required: frozenset[str] = CANONICAL_EN

    @classmethod
    def build(cls, model_id: str, table_name: str, vocab: Mapping[str, int]) -> "PhoneMap":
        """Bind a table to a concrete vocabulary, failing loud on any gap."""
        try:
            table = TABLES[table_name]
        except KeyError as exc:
            raise PhoneSetError(f"unknown phone table '{table_name}'") from exc
        required = REQUIRED.get(table_name, CANONICAL_EN)

        missing_canonical = sorted(required - set(table))
        if missing_canonical:
            raise PhoneSetError(
                f"table '{table_name}' is missing required canonical phones: {missing_canonical}"
            )
        missing_tokens = sorted({t for t in table.values() if t not in vocab})
        if missing_tokens:
            raise PhoneSetError(
                f"table '{table_name}' maps to tokens absent from the {model_id} "
                f"vocabulary: {missing_tokens}"
            )
        # Only phones the engine may discriminate need unique tokens: a lossy
        # schwa mapping must not corrupt alignment, but a shared token between
        # /ð/ and /z/ would corrupt every verdict.
        required_tokens = [table[p] for p in required]
        collisions = sorted({t for t in required_tokens if required_tokens.count(t) > 1})
        if collisions:
            raise PhoneSetError(
                f"table '{table_name}' maps several required phones to {collisions}; "
                "discriminated phones must stay distinguishable"
            )

        blank: int | None = None
        for candidate in _BLANK_CANDIDATES:
            blank = vocab.get(candidate)
            if blank is not None:
                break
        if blank is None:
            blank = 0

        return cls(
            model_id=model_id,
            table_name=table_name,
            token_of=MappingProxyType(dict(table)),
            index_of=MappingProxyType({p: vocab[t] for p, t in table.items()}),
            blank_index=blank,
            required=required,
        )

    def to_index(self, canonical_phone: str) -> int:
        try:
            return self.index_of[canonical_phone]
        except KeyError as exc:
            raise PhoneSetError(
                f"/{canonical_phone}/ has no mapping for model '{self.model_id}'"
            ) from exc

    def to_indices(self, phones: "list[str] | tuple[str, ...]") -> list[int]:
        return [self.to_index(p) for p in phones]

    def canonical_of_token(self, token: str) -> str | None:
        """Inverse lookup; None for tokens outside the canonical inventory."""
        for canonical, mapped in self.token_of.items():
            if mapped == token:
                return canonical
        return None
