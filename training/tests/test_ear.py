"""Ear pipeline maths, without g2p_en/datasets/models: vocab, token mapping,
speaker-disjoint split, and the feature batch/padding arithmetic."""

from pathlib import Path

import numpy as np
import pytest
import torch

from openschwa_training.ear_prep import TOKEN_INDEX, TOKEN_SET, sentence_to_tokens, vocab
from openschwa_training.ear_train import ClipRef, ShardIndex, _client_split, _clip_batch


class FakeG2p:
    """Minimal stand-in for g2p_en's G2p: returns phones with stress digits."""

    def __init__(self, phones: dict[str, list[str]] | None = None) -> None:
        self.phones = phones or {}

    def __call__(self, word: str) -> list[str]:
        if word not in self.phones:
            return ["UW0", "N1", "K0"]  # "unk": a harmless fallback
        return self.phones[word]


def test_vocab_is_pad_blank_plus_39_phones():
    vocabulary = vocab()
    assert vocabulary["[PAD]"] == 0
    assert len(vocabulary) == 40
    assert len(TOKEN_SET) == 39
    assert vocabulary["DH"] == TOKEN_INDEX["DH"]


def test_sentence_to_tokens_strips_stress_digits():
    g2p = FakeG2p({"this": ["DH0", "IH1", "S0"], "is": ["IH1", "Z0"]})
    assert sentence_to_tokens("this is", g2p) == ["DH", "IH", "S", "IH", "Z"]


def test_sentence_to_tokens_rejects_unmappable_phones_loudly():
    g2p = FakeG2p({"weird": ["DH0", "XQ1"]})
    with pytest.raises(ValueError, match="unmappable"):
        sentence_to_tokens("weird", g2p)


def test_client_split_is_deterministic_and_val_is_a_small_bucket():
    splits = {_client_split(f"client-{i}") for i in range(500)}
    assert splits <= {"train", "val"}
    # ~5% bucket; 500 clients is enough that both appear and val is the minority.
    val_count = sum(1 for i in range(500) if _client_split(f"client-{i}") == "val")
    assert 10 <= val_count <= 50


def test_clip_batch_pads_features_and_targets():
    shard = ShardIndex(
        path=Path("shard.npy"),
        clips=(
            {"id": "a", "start": 0, "length": 3, "tokens": "DH IH S"},
            {"id": "b", "start": 3, "length": 5, "tokens": "Z"},
        ),
    )
    array = np.arange(8 * 1024, dtype=np.float32).reshape(8, 1024)
    arrays = {shard.path: array}
    picks = [ClipRef(shard=shard, position=0), ClipRef(shard=shard, position=1)]
    padded, lengths, targets, target_lengths = _clip_batch(picks, arrays)
    assert padded.shape == (2, 5, 1024)
    assert lengths.tolist() == [3, 5]
    assert float(padded[0, 3].abs().sum()) == 0.0  # padding is zeros
    assert target_lengths.tolist() == [3, 1]
    assert targets[0, 0].item() == TOKEN_INDEX["DH"]
    assert targets[1, 0].item() == TOKEN_INDEX["Z"]
    assert targets[1, 1].item() == 0  # target padding is the blank token
    # The second clip's feature block starts at array row 3.
    assert bool(torch.allclose(padded[1, 0], torch.from_numpy(array[3])))
