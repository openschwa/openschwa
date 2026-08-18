"""The minimal TextGrid parser, pinned on long-format fixtures."""

import pytest

from openschwa_eval.textgrid import TextGridError, parse

GRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 3
            intervals [1]:
                xmin = 0
                xmax = 0.1
                text = "sil"
            intervals [2]:
                xmin = 0.1
                xmax = 0.6
                text = "DH"
            intervals [3]:
                xmin = 0.6
                xmax = 1.0
                text = ""
    item [2]:
        class = "IntervalTier"
        name = "IPA"
        xmin = 0
        xmax = 1.0
        intervals: size = 2
            intervals [1]:
                xmin = 0.1
                xmax = 0.6
                text = "ð,z,s"
            intervals [2]:
                xmin = 0.6
                xmax = 1.0
                text = ""
"""


def test_parses_tiers_and_intervals():
    tiers = parse(GRID)
    assert [t.name for t in tiers] == ["phones", "IPA"]
    phones = tiers[0].intervals
    assert len(phones) == 3
    assert phones[1].text == "DH"
    assert phones[1].xmin == 0.1
    assert phones[1].xmax == 0.6


def test_overlap_fraction():
    tiers = parse(GRID)
    tag = tiers[1].intervals[0]
    phone = tiers[0].intervals[1]
    assert tag.overlaps(phone) == 1.0
    assert tiers[0].intervals[0].overlaps(tag) == 0.0


def test_rejects_short_format():
    with pytest.raises(TextGridError, match="long-format"):
        parse('File type = "ooTextFile"\nObject class = "TextGrid"\nxmin = 0')


def test_rejects_non_interval_tiers():
    bad = GRID.replace('name = "IPA"', 'name = "IPA2"').replace(
        'class = "IntervalTier"\n        name = "IPA2"', 'class = "TextTier"\n        name = "IPA2"'
    )
    with pytest.raises(TextGridError, match="not an IntervalTier"):
        parse(bad)
