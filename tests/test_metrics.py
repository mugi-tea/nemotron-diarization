import os

from livediar.diarizer import Segment
from livediar.metrics import cer, der, load_rttm, normalize_text


def test_cer_zero_for_identical_after_normalization():
    assert cer("お疲れ様です。今日は、来週。", "お疲れ様です 今日は来週") == 0.0


def test_cer_counts_substitutions():
    assert abs(cer("仕様書", "使用書") - 2 / 3) < 1e-9


def test_normalize_strips_punctuation_and_width():
    assert normalize_text("ＡＰＩの変更点は？") == "apiの変更点は"


def test_der_identical_is_zero(fixtures_dir):
    ref = load_rttm(os.path.join(fixtures_dir, "meeting_ja_3spk.ref.rttm"))
    rate, info = der(ref, ref)
    assert rate == 0.0 and info["confusion"] == 0.0


def test_der_finds_best_label_mapping():
    ref = [Segment(0, 5, "A"), Segment(6, 10, "B")]
    hyp = [Segment(0, 5, 2), Segment(6, 10, 1)]  # same segmentation, swapped labels
    rate, info = der(ref, hyp)
    assert rate < 0.01
    assert info["mapping"] == {2: "A", 1: "B"}


def test_der_penalizes_confusion():
    ref = [Segment(0, 5, "A"), Segment(6, 10, "B")]
    hyp = [Segment(0, 5, 1), Segment(6, 10, 1)]
    rate, _ = der(ref, hyp)
    assert 0.4 < rate < 0.5
