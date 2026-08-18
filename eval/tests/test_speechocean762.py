"""speechocean762 adapter, pinned on a synthetic corpus directory."""

import json
import wave

import pytest

from openschwa_eval.datasets.speechocean762 import SpeechOcean762

SCORES = {
    "000010011": {
        "words": [
            {"text": "THE", "phones": ["DH", "AH0"]},
            {"text": "ZEN", "phones": ["Z", "EH1", "N"]},
        ]
    },
    "000010035": {"words": [{"text": "ZERO", "phones": ["Z", "IY1", "R", "OW0"]}]},
}

DETAIL = {
    "000010011": {
        "words": [
            {"phones": ["{DH} AH0"] * 3 + ["DH AH0"] * 2},  # 3/5 experts -> error
            {"phones": ["Z EH1 N"] * 5},
        ]
    },
    "000010035": {"words": [{"phones": ["Z IY1 R OW0"] * 5}]},
}


def _tiny_wav(path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(b"\x00\x00" * 1600)


@pytest.fixture
def corpus(tmp_path):
    root = tmp_path / "so762"
    (root / "resource").mkdir(parents=True)
    wav_dir = root / "WAVE" / "SPEAKER0001"
    wav_dir.mkdir(parents=True)
    (root / "train").mkdir()
    (root / "test").mkdir()
    (root / "resource" / "scores.json").write_text(json.dumps(SCORES), encoding="utf-8")
    (root / "resource" / "scores-detail.json").write_text(json.dumps(DETAIL), encoding="utf-8")
    for utt in ("000010011", "000010035"):
        _tiny_wav(wav_dir / f"{utt}.WAV")
    (root / "train" / "wav.scp").write_text(
        "000010011 WAVE/SPEAKER0001/000010011.WAV\n", encoding="utf-8"
    )
    (root / "train" / "text").write_text("000010011 THE ZEN\n", encoding="utf-8")
    (root / "test" / "wav.scp").write_text(
        "000010035 WAVE/SPEAKER0001/000010035.WAV\n", encoding="utf-8"
    )
    (root / "test" / "text").write_text("000010035 ZERO\n", encoding="utf-8")
    return root


def test_majority_vote_marks_errors(corpus):
    utterances = list(SpeechOcean762(corpus).utterances("ð"))
    assert len(utterances) == 1
    utterance = utterances[0]
    assert utterance.split == "train"
    assert utterance.l1 == "mandarin"
    assert utterance.corpus == "so762"
    phones = utterance.phones
    assert phones[0].phone == "ð" and phones[0].label == "substituted"
    assert phones[1].phone == "ʌ" and phones[1].label == "correct"


def test_no_target_no_utterance(corpus):
    assert list(SpeechOcean762(corpus).utterances("θ")) == []


def test_missing_score_files_fail_loud(tmp_path):
    with pytest.raises(Exception, match="missing"):
        SpeechOcean762(tmp_path)
