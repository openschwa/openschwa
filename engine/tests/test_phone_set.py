"""Phone-set mapping is a named top risk in docs/architecture.md: a wrong entry
does not crash, it silently scores the wrong segment. These tests run against
the committed vocabulary snapshot, so they catch mapping breakage in CI without
downloading any weights."""

import json

import pytest

from openschwa_engine.models.phone_set import (
    TABLES,
    PhoneMap,
    PhoneSetError,
    normalize,
)
from openschwa_engine.models.registry import MANIFEST, VOCAB_DIR


def _vocab(spec):
    return json.loads((VOCAB_DIR / spec.vocab_snapshot).read_text(encoding="utf-8"))


@pytest.fixture(params=list(MANIFEST.values()), ids=lambda s: s.id)
def phone_map(request):
    return PhoneMap.build(request.param.id, request.param.phone_table, _vocab(request.param))


def test_every_required_phone_maps(phone_map):
    """Every phone the model may *discriminate* must have a token. Outside the
    required set a table may be lossy (see CHARSIU_EN's schwa), but a missing
    drill phone would silently corrupt verdicts."""
    assert phone_map.required <= set(phone_map.token_of)


def test_round_trip_canonical_to_token_and_back(phone_map):
    """The mapping must be injective on the required set — two required phones
    sharing a token would be indistinguishable, which is fatal for closed-set
    contrast scoring."""
    for phone in sorted(phone_map.required):
        token = phone_map.token_of[phone]
        assert phone_map.canonical_of_token(token) == phone


def test_indices_are_real_vocabulary_positions(phone_map):
    vocab = _vocab(MANIFEST[phone_map.model_id])
    for phone, token in phone_map.token_of.items():
        assert phone_map.to_index(phone) == vocab[token]


def test_blank_is_the_pad_token(phone_map):
    vocab = _vocab(MANIFEST[phone_map.model_id])
    if any(c in vocab for c in ("<pad>", "[PAD]", "pad")):
        expected = next(vocab[c] for c in ("<pad>", "[PAD]", "pad") if c in vocab)
        assert phone_map.blank_index == expected


def test_script_g_is_not_ascii_g(phone_map):
    """U+0261 ɡ and ASCII g look identical in most fonts but are different
    characters; the espeak vocabulary contains only the former. Models with a
    closed-set-only vocabulary (dh-contrast-v1) never see either."""
    assert normalize("g") == "ɡ"
    if "ɡ" in phone_map.token_of:
        assert "g" not in phone_map.token_of


def test_charsiu_required_phones_map_to_unique_tokens():
    """The bake-off contrast lives inside the required set: /ð/ vs {z, d, v}
    must be four distinct charsiu tokens."""
    charsiu = TABLES["charsiu_en"]
    assert charsiu["ð"] == "DH"
    assert charsiu["z"] == "Z"
    assert charsiu["d"] == "D"
    assert charsiu["v"] == "V"


def test_charsiu_may_be_lossy_outside_the_required_set():
    """No schwa exists in the charsiu vocabulary, so ə and ʌ share AH: fine for
    alignment, and unreachable for contrast scoring (both are outside the
    required set)."""
    charsiu = TABLES["charsiu_en"]
    assert charsiu["ə"] == charsiu["ʌ"] == "AH"


def test_english_long_vowels_carry_the_length_mark():
    """The vocabulary is multilingual and holds both `i` and `iː`; espeak emits
    the long form for English FLEECE. Mapping to the bare token would align
    against something the model never predicts here."""
    espeak = TABLES["espeak_en"]
    assert espeak["i"] == "iː"
    assert espeak["u"] == "uː"
    assert espeak["ɝ"] == "ɜː"  # no ɝ exists in this vocabulary at all
    assert espeak["ɑɹ"] == "ɑːɹ"


@pytest.mark.parametrize(
    ("written", "expected"),
    [("ˈɪ", "ɪ"), ("iː", "i"), ("r", "ɹ"), ("ʧ", "tʃ"), (" ð ", "ð"), ("ɜː", "ɝ")],
)
def test_normalize_accepts_reasonable_author_spellings(written, expected):
    assert normalize(written) == expected


def test_normalize_rejects_unknown_phones():
    with pytest.raises(PhoneSetError, match="canonical inventory"):
        normalize("ʘ")


def test_build_rejects_a_vocabulary_missing_mapped_tokens():
    with pytest.raises(PhoneSetError, match="absent from"):
        PhoneMap.build("toy", "espeak_en", {"<pad>": 0, "ð": 1})
