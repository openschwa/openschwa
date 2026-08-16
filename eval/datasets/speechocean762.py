"""speechocean762 adapter (https://www.openslr.org/101/). M1.

Phone-level accuracy scores from five experts; L1-Mandarin speakers. The
adapter maps its ARPAbet-style labels into the canonical IPA inventory and
treats expert accuracy below the corpus's documented threshold as
"substituted"/"deleted" per its annotation scheme.
"""

from collections.abc import Iterator

from .base import DatasetAdapter, LabeledToken


class SpeechOcean762(DatasetAdapter):
    def tokens(self, target_phone: str) -> Iterator[LabeledToken]:
        raise NotImplementedError("M1")
