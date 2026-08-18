"""Corpus adapters - see base.py for the interface."""

from openschwa_eval.datasets.base import DatasetAdapter, PhoneToken, Utterance
from openschwa_eval.datasets.l2arctic import L2Arctic
from openschwa_eval.datasets.speechocean762 import SpeechOcean762

__all__ = ["DatasetAdapter", "L2Arctic", "PhoneToken", "SpeechOcean762", "Utterance"]
