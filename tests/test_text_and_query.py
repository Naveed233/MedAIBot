from medaibot.query import guess_section, normalise_query
from medaibot.text import normalize, numbers_in, split_sentences, tokenize


def test_japanese_runs_become_bigrams():
    assert tokenize("タグリッソ") == ["タグ", "グリ", "リッ", "ッソ"]


def test_half_width_katakana_is_folded_before_tokenising():
    assert tokenize("ﾀｸﾞﾘｯｿ") == tokenize("タグリッソ")


def test_english_tokens_drop_stopwords_and_keep_doses():
    assert tokenize("What is the dose of TAGRISSO 80 mg?") == ["dose", "tagrisso", "80", "mg"]


def test_full_width_ascii_is_normalised():
    assert normalize("ＴＡＧＲＩＳＳＯ　８０ｍｇ") == "tagrisso 80mg"


def test_sentence_split_handles_english_bullets_and_japanese():
    assert split_sentences("Take 80 mg daily. • Do not crush.") == ["Take 80 mg daily.", "• Do not crush."]
    assert split_sentences("用量は80mgです。食事に関係なく服用。") == ["用量は80mgです。", "食事に関係なく服用。"]


def test_numbers_in_finds_decimals():
    assert numbers_in("eGFR below 25 mL/min/1.73 m2") == {"25", "1.73", "2"}


def test_katakana_question_is_mapped_to_english_terms(catalog):
    nq = normalise_query("タグリッソの用量は？", catalog)
    assert nq.drug == "tagrisso"
    assert "TAGRISSO" in nq.text and "dosage" in nq.text
    assert nq.section == "dosage_and_administration"
    assert nq.translated_terms == ["用量"]


def test_english_question_is_left_unchanged(catalog):
    nq = normalise_query("What is the dose of Lynparza?", catalog)
    assert nq.text == "What is the dose of Lynparza?"
    assert nq.drug == "lynparza"


def test_no_drug_and_two_drugs_give_no_filter(catalog):
    assert normalise_query("What is the starting dose?", catalog).drug is None
    both = normalise_query("Can Tagrisso be taken with Farxiga?", catalog)
    assert both.drugs == ["tagrisso", "farxiga"] and both.drug is None


def test_section_guess():
    assert guess_section("Is it safe in pregnancy?") == "use_in_specific_populations"
    assert guess_section("禁忌は？") == "contraindications"
    assert guess_section("Hello") is None


def test_catalog_matches_aliases_and_respects_word_boundaries(catalog):
    assert [d.key for d in catalog.find_in_text("フォシーガ")] == ["farxiga"]
    assert [d.key for d in catalog.find_in_text("osimertinib mesylate")] == ["tagrisso"]
    assert catalog.find_in_text("imfinzified") == []


def test_a_latin_drug_name_inside_a_japanese_sentence_is_found(catalog):
    assert [d.key for d in catalog.find_in_text("TAGRISSOの副作用は？")] == ["tagrisso"]
    assert normalise_query("LYNPARZAの用量は？", catalog).drug == "lynparza"


def test_age_questions_point_to_specific_populations_but_dosage_does_not():
    assert guess_section("anything related to age?") == "use_in_specific_populations"
    assert guess_section("Is it safe for older patients?") == "use_in_specific_populations"
    assert guess_section("What is the recommended dosage?") == "dosage_and_administration"
