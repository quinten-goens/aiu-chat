"""Rendering and round-tripping of a draft.

The screen, the HTML export and the plain-text export must all say the same
thing, and an edited draft must still be checkable -- if the HTML the analyst
edits cannot be turned back into text, the verify gate silently stops applying to
exactly the version that gets published.
"""
import datetime as dt

import pytest

from aiu_chat.nsr.draft import Bullet, Draft, markdown_to_html, parse_headline
from aiu_chat.nsr.facts import Fact, Week, WeekFacts

HEADLINE = (
    "Traffic\n"
    "There were 33,890 daily flights in W22, 1.9% more than in W21.\n\n"
    "ATFM Delay\n"
    "ATFM delay was 2.6 min/flight on W22.\n\n"
    "Punctuality\n"
    "In W22 arrival punctuality decreased to 77%."
)


@pytest.fixture
def draft():
    week = Week(monday=dt.date(2026, 5, 25), sunday=dt.date(2026, 5, 31), iso_week=22)
    wf = WeekFacts(week=week)
    wf.add(Fact("traffic_daily", "33,890", 33890.0, "flights/day", "api", "avgValue"))
    return Draft(
        week=week, facts=wf, headline=HEADLINE,
        enroute=[Bullet("Barcelona ACC", "acc", 1, 2.12,
                        "recorded ATC capacity delays throughout the week.")],
        airport=[Bullet("Munich", "airport", 6, 2.18,
                        "suffered from thunderstorms on Sunday.")],
    )


def test_headline_parses_into_heading_body_pairs():
    # The model writes "Traffic\nThere were..." -- a line break, not a paragraph
    # break. Rendered raw into markdown the heading glues onto the body and the
    # report loses its bold headings, which is exactly what shipped.
    assert parse_headline(HEADLINE) == [
        ("Traffic", "There were 33,890 daily flights in W22, 1.9% more than in W21."),
        ("ATFM Delay", "ATFM delay was 2.6 min/flight on W22."),
        ("Punctuality", "In W22 arrival punctuality decreased to 77%."),
    ]


def test_markdown_separates_heading_from_body(draft):
    md = draft.to_markdown()
    # A blank line between them is what makes the heading render bold and alone.
    assert "**Traffic**\n\nThere were 33,890" in md
    assert "- **Barcelona ACC** recorded ATC capacity delays" in md


def test_html_emits_bold_headings_and_a_bullet_list(draft):
    html = draft.to_html()
    assert "<p><strong>Traffic</strong></p><p>There were 33,890" in html
    assert "<li><strong>Munich</strong> suffered from thunderstorms" in html
    assert html.count("<ul>") == 2  # en-route and airport


def test_plain_text_keeps_the_title_and_the_bullets(draft):
    text = draft.to_text()
    assert text.startswith("Network situation (Week 22: 25 - 31 May)")
    assert "  - Munich suffered from thunderstorms on Sunday." in text


def test_edited_markdown_round_trips_back_to_publishable_html():
    # The analyst edits; the export must reflect the edit, not the original draft.
    md = ("**Traffic**\n\nThere were 33,890 daily flights.\n\n"
          "**For en-route ATFM delay:**\n\n"
          "- **Barcelona ACC** recorded ATC capacity delays.")
    html = markdown_to_html(md)
    assert "<p><strong>Traffic</strong></p>" in html
    assert "<p>There were 33,890 daily flights.</p>" in html
    assert "<ul><li><strong>Barcelona ACC</strong> recorded ATC capacity delays.</li></ul>" in html


def test_a_heading_only_line_does_not_become_a_paragraph_of_text():
    assert markdown_to_html("**Punctuality**") == "<p><strong>Punctuality</strong></p>"
