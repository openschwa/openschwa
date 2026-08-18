"""ARPABET <-> canonical IPA mapping, shared by both corpus adapters.

The two corpora spell phones differently from each other AND from the engine's
canonical inventory (phone_set.py). speechocean762 uses classic ARPABET with
stress digits (DH, AH0); L2-ARCTIC mixes ARPABET with IPA symbols (ɹ, ə) and
marks stress as R vs R*. One explicit table per direction, with loud failures,
keeps a label mismatch from quietly relabelling evidence.
"""

import re
from types import MappingProxyType

from openschwa_engine.models.phone_set import PhoneSetError, normalize

#: ARPABET (stress stripped) -> canonical IPA. Covers the full CMU set.
ARPABET_TO_CANONICAL = MappingProxyType(
    {
        # vowels
        "AA": "ɑ",
        "AE": "æ",
        "AH": "ʌ",
        "AX": "ə",  # pre-1993 CMU symbol for schwa
        "AO": "ɔ",
        "AW": "aʊ",
        "AY": "aɪ",
        "EH": "ɛ",
        "ER": "ɝ",
        "EY": "eɪ",
        "IH": "ɪ",
        "IY": "i",
        "OW": "oʊ",
        "OY": "ɔɪ",
        "UH": "ʊ",
        "UW": "u",
        # consonants
        "B": "b",
        "CH": "tʃ",
        "D": "d",
        "DH": "ð",
        "F": "f",
        "G": "ɡ",
        "HH": "h",
        "JH": "dʒ",
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ŋ",
        "P": "p",
        "R": "ɹ",
        "S": "s",
        "SH": "ʃ",
        "T": "t",
        "TH": "θ",
        "V": "v",
        "W": "w",
        "Y": "j",
        "Z": "z",
        "ZH": "ʒ",
    }
)

_STRESS = re.compile(r"[012*]$")


class LabelMappingError(ValueError):
    """A corpus label cannot be mapped to the canonical inventory."""


def from_arpabet(label: str) -> str:
    """Map an ARPABET phone (stress digit stripped) to canonical IPA."""
    cleaned = label.strip()
    if cleaned in ("", "SIL", "SP", "SPN", "sil", "sp", "spn", "err"):
        raise LabelMappingError(f"'{label}' is not a phone label")
    base = _STRESS.sub("", cleaned)
    canonical = ARPABET_TO_CANONICAL.get(base)
    if canonical is None:
        raise LabelMappingError(f"unknown ARPABET phone '{label}'")
    return canonical


#: Annotation junk the hand-transcribers attached to real phones: stress
#: digits, Praat-ish markers, and plain typos. Stripped from the label tail
#: before mapping (AA*1 -> AA, ER) -> ER, V8 -> V, Z_ -> Z ...).
_JUNK_SUFFIX = "012*)_`8"


def from_l2arctic_label(label: str) -> str | None:
    """Map a label from an L2-ARCTIC phone tier to canonical IPA, or None.

    Returns None for non-phone markers (sil, sp, err). Raises for anything
    that looks like a phone but cannot be mapped: silent loss of evidence is
    the bug this module exists to prevent.
    """
    cleaned = label.strip()
    if not cleaned or cleaned.lower() in ("sil", "sp", "err", "spn"):
        return None
    while len(cleaned) > 1 and cleaned[-1] in _JUNK_SUFFIX:
        cleaned = cleaned[:-1]
    if not cleaned or cleaned.lower() in ("sil", "sp", "err", "spn"):
        return None
    if cleaned.startswith("R") and len(cleaned) > 1:
        # L2-ARCTIC spells the bunched-r variants (R, R*) differently from the
        # plain ARPABET R; both are the canonical /ɹ/.
        cleaned = "R"
    try:
        return from_arpabet(cleaned)
    except LabelMappingError:
        pass
    # Case drift (Ah, Uh): the ARPABET table is uppercase.
    try:
        return from_arpabet(cleaned.upper())
    except LabelMappingError:
        pass
    # Mixed IPA spellings the annotators used (z, ɹ, ə, dʒ, ɫ ...): run them
    # through the engine's own canonicalizer, which knows the inventory.
    try:
        return normalize(cleaned)
    except PhoneSetError:
        pass
    if cleaned == "ɫ":
        return "l"
    raise LabelMappingError(f"unmappable L2-ARCTIC label '{label}'")
