from livediar.textfilters import (clean_segment, collapse_repeats, compression_ratio, finalize_text,
                                  is_hallucination, plausible_length, trigram_similarity)


def test_collapse_repeats_folds_runaway_loops():
    assert collapse_repeats("お前" * 40) == "お前お前"
    assert collapse_repeats("どうしたどうした?いや別に") == "どうしたどうした?いや別に"  # 2x stays


def test_clean_segment_drops_mostly_repetitive_text():
    assert clean_segment("お前" * 40 + "だ") == ""
    assert clean_segment("反映済みです。ただ認証周りのテストがまだ残っています。") == "反映済みです。ただ認証周りのテストがまだ残っています。"


def test_compression_ratio_separates_natural_and_repetitive_text():
    assert compression_ratio("お疲れ様です。今日は来週のリリース計画について話しましょう。") < 2.4
    assert compression_ratio("お前、" * 30) > 2.4


def test_hallucination_boilerplate():
    assert is_hallucination("ご視聴ありがとうございました")
    assert not is_hallucination("ありがとうございました。では次回に。")


def test_plausible_length_and_finalize():
    assert plausible_length("承知しました。", 1.5)
    assert not plausible_length("あ" * 100, 1.0)
    assert finalize_text("あ" * 100, 1.0) == ""
    assert finalize_text("承知しました。", 1.5) == "承知しました。"


def test_trigram_similarity():
    assert trigram_similarity("言わせてもらいますよ", "じゃあいいですか?言わせてもらいますよ") == 1.0
    assert trigram_similarity("どら焼きだよ", "承知しました") == 0.0
    assert trigram_similarity("", "x") == 0.0
