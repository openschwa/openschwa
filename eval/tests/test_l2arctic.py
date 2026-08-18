"""L2-ARCTIC adapter, pinned on a synthetic speaker directory."""

import wave

import pytest

from openschwa_eval.datasets.l2arctic import AdapterError, L2Arctic

ANNOTATION = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.2
tiers? <exists>
size = 3
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.2
        intervals: size = 1
            intervals [1]:
                xmin = 0.1
                xmax = 0.7
                text = "then"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.2
        intervals: size = 5
            intervals [1]:
                xmin = 0
                xmax = 0.1
                text = "sil"
            intervals [2]:
                xmin = 0.1
                xmax = 0.25
                text = "DH,Z,s"
            intervals [3]:
                xmin = 0.25
                xmax = 0.5
                text = "EH1"
            intervals [4]:
                xmin = 0.5
                xmax = 0.7
                text = "N"
            intervals [5]:
                xmin = 0.7
                xmax = 1.2
                text = "sp"
"""

DELETION = ANNOTATION.replace('text = "DH,Z,s"', 'text = "DH,D,d"')


def _tiny_wav(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(44_100)
        handle.writeframes(b"\x00\x00" * 4410)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "l2arctic"
    speaker = root / "ABA"
    (speaker / "wav").mkdir(parents=True)
    (speaker / "transcript").mkdir()
    (speaker / "annotation").mkdir()
    _tiny_wav(speaker / "wav" / "arctic_a0001.wav")
    (speaker / "transcript" / "arctic_a0001.txt").write_text("then\n", encoding="utf-8")
    (speaker / "annotation" / "arctic_a0001.TextGrid").write_text(ANNOTATION, encoding="utf-8")
    return root


def test_parses_a_substitution(corpus):
    utterances = list(L2Arctic(corpus).utterances("ð"))
    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.l1 == "arabic"
    assert utterance.corpus == "l2arctic"
    assert [t.phone for t in utterance.phones] == ["ð", "ɛ", "n"]
    token = utterance.phones[0]
    assert token.label == "substituted"
    assert token.substituted_with == "z"
    assert token.start_s == 0.1


def test_parses_a_deletion(tmp_path, corpus):
    (corpus / "ABA" / "annotation" / "arctic_a0001.TextGrid").write_text(DELETION, encoding="utf-8")
    utterances = list(L2Arctic(corpus).utterances("ð"))
    token = utterances[0].phones[0]
    assert token.label == "deleted"
    assert token.substituted_with is None


def test_only_utterances_containing_the_target_are_yielded(corpus):
    assert list(L2Arctic(corpus).utterances("z")) == []


def test_unannotated_speakers_are_skipped(corpus):
    (corpus / "BWC").mkdir()
    assert len(list(L2Arctic(corpus).utterances("ð"))) == 1


def test_unknown_speaker_directory_fails_loud(corpus):
    (corpus / "UNKNOWN").mkdir()
    with pytest.raises(AdapterError, match="UNKNOWN"):
        list(L2Arctic(corpus).utterances("ð"))


def test_unparsable_tag_is_skipped_and_recorded(corpus):
    """A malformed tag must not kill the run, and must not become a fake
    'correct' label either: the token is dropped and the loss recorded."""
    adapter = L2Arctic(corpus)
    (corpus / "ABA" / "annotation" / "arctic_a0001.TextGrid").write_text(
        ANNOTATION.replace('text = "DH,Z,s"', 'text = "DH,Z,???"'), encoding="utf-8"
    )
    assert list(adapter.utterances("ð")) == []
    assert adapter.skipped_labels
