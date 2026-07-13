"""Bullet post-processing.

The bullet is a continuation of the entity name that the UI prints separately, so
the model's text must not restate it. The prompt says so; these tests cover the
model ignoring the prompt, using outputs it actually produced.
"""
from aiu_chat.nsr.draft import _clean_bullet


def test_leaked_pipe_separator_is_stripped():
    # Straight from a real run: the few-shot examples paired "Name | text" and the
    # model copied the pairing into its answer.
    got = _clean_bullet("Karlsruhe UAC | experienced weather-related regulations.",
                        "Karlsruhe UAC")
    assert got == "experienced weather-related regulations."


def test_restated_name_is_stripped():
    got = _clean_bullet("Reims ACC faced staffing shortages on Monday.", "Reims ACC")
    assert got == "faced staffing shortages on Monday."


def test_bracketed_name_is_stripped():
    got = _clean_bullet("[Barcelona ACC] recorded ATC capacity delays.", "Barcelona ACC")
    assert got == "recorded ATC capacity delays."


def test_name_with_added_suffix_is_stripped():
    # The model likes to write "Frankfurt airport experienced..." where the
    # ranking calls the entity just "Frankfurt".
    got = _clean_bullet("Frankfurt airport experienced high arrival delays.",
                        "Frankfurt")
    assert got == "experienced high arrival delays."


def test_hyphenated_variant_of_the_name_is_stripped():
    # The ranking says "Tel Aviv"; NOP (and then the model) says "Tel-Aviv".
    got = _clean_bullet("Tel-Aviv experienced daily ATC capacity regulations.",
                        "Tel Aviv")
    assert got == "experienced daily ATC capacity regulations."


def test_leading_icao_code_is_stripped():
    got = _clean_bullet("EHAM recovered from IT issues.", "Amsterdam")
    assert got == "recovered from IT issues."


def test_chained_restatements_are_all_stripped():
    # Seen in a real run: name, then ICAO, then name again.
    got = _clean_bullet("Maastricht UAC EDYY Maastricht experienced weather delays.",
                        "Maastricht UAC")
    assert got == "experienced weather delays."


def test_a_clean_bullet_is_left_alone():
    text = "recorded ATC capacity delays throughout the week."
    assert _clean_bullet(text, "Barcelona ACC") == text


def test_entity_named_mid_sentence_is_kept():
    # Only a leading restatement is noise; the name appearing later is legitimate.
    text = "suffered delays; traffic into Zurich was regulated."
    assert _clean_bullet(text, "Zurich") == text


def test_an_opening_word_is_not_mistaken_for_an_icao_code():
    # The ICAO strip must not eat a real four-letter first word.
    text = "saw regulations throughout the week."
    assert _clean_bullet(text, "Nice") == text
    text2 = "faced delays due to staffing."
    assert _clean_bullet(text2, "Vienna") == text2
