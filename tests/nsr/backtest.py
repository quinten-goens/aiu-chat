"""Backtest the NSR fact layer against the reports EUROCONTROL actually published.

For every past week that has both a published NSR and Data App coverage, this
re-derives the numbers from the API and diffs them against the figures in the
published prose. It is the objective signal for whether the drafter can be
trusted: if we cannot reproduce last quarter's reports, we have no business
drafting next Monday's.

    .venv/bin/python tests/nsr/backtest.py            # all covered weeks
    .venv/bin/python tests/nsr/backtest.py --weeks 6  # just the most recent 6

Hits the live API + the published archive; not part of the offline test run.
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import sys

import requests

from aiu_chat import config
from aiu_chat.nsr import facts
from aiu_chat.sources import dataapp

# The claims we can check, and how to pull them out of the published prose.
# Each maps a fact key -> regex whose first group is the published literal.
CLAIMS: dict[str, str] = {
    "traffic_daily":       r"There were ([\d,]+) daily flights",
    "traffic_wow":         r"daily flights in W\d+, (-?[\d.]+)% (?:more|less)",
    "traffic_vs_prev_year": r"This is (-?\d+)% (?:higher|lower) than in 20\d\d",
    "delay_per_flight":    r"ATFM delay was ([\d.]+) min/flight",
    "delay_enroute":       r"min/flight \(([\d.]+) en-route",
    "delay_airport":       r"en-route / ([\d.]+) airport\)",
    "delay_wow":           r"on W\d+, (-?\d+)% (?:higher|lower) than in W\d+",
    # Anchor on "punctuality ... to N%": a looser pattern picks up the trailing
    # digits of the traffic sentence instead.
    "punctuality":         r"punctuality (?:decreased|increased|remained)[^.%]*?to (\d+)%",
    # Improvements are published unsigned ("3pp better"), regressions signed
    # ("-4pp worse"); the direction lives in the adjective, so read it from there.
    # House style changed partway through the archive: the older reports carry a
    # decimal ("10.1pp", "-0.1pp"), the recent ones round to whole pp -- so the
    # fractional part is optional, and must not be dropped (a `\d+`-only pattern
    # reads "10.1pp" as "1pp").
    "punctuality_wow":     r"(-?\d+(?:\.\d+)?)pp (?:better|worse)",
}

# Published literals whose sign is carried by a word rather than a minus sign.
SIGNED_BY_WORD = {"punctuality_wow": ("better", "worse")}


def _strip(html: str) -> str:
    text = re.sub(r"</li>|</p>", "\n", html)
    return re.sub(r"<[^>]+>", "", text).replace("\xa0", " ")


def published_reports(session: requests.Session) -> list[dict]:
    r = session.get(
        f"{config.DATAAPP_BASE}/situation_reports",
        params={"itemsPerPage": 100, "order[date]": "desc"},
        timeout=30, headers={"User-Agent": dataapp.USER_AGENT},
    )
    r.raise_for_status()
    return r.json()["data"]


def week_from_title(title: str, report_date: str) -> facts.Week | None:
    """'Network situation (Week 22: 25 - 31 May)' -> the Week it actually covers.

    Anchored on the *publication date*, not the week number in the title: the
    archive contains a mislabelled report (16-22 Feb 2026 is titled "Week 7",
    duplicating the real W7 of 9-15 Feb). Trusting the number there would diff a
    report against the wrong week's data. Reports go out in the days right after
    the week closes, so the last complete week before publication is the subject.
    """
    if not re.search(r"Week \d+", title):
        return None
    return facts.last_complete_week(dt.date.fromisoformat(report_date[:10]))


def _norm(text: str) -> str:
    """Compare published vs derived literals on their digits alone."""
    return re.sub(r"[^\d.-]", "", text.replace(",", ""))


def _agrees(published: str, derived: str) -> bool:
    """Do the published and derived literals state the same figure?

    Compared numerically at the precision the *report* chose, because house style
    is inconsistent across the archive: the same quantity appears as "10.1pp" in
    2025 and "3pp" in 2026. A derived -3.89 agrees with a published "-4" and also
    with a published "-3.9"; it does not agree with "-3".

    One rounding step of slack, because we are diffing a live API against numbers
    frozen at publication: the API now says 9.98pp where the report said 10.1pp.
    Figures get revised after the fact, so demanding bit-equality would flag data
    vintage as a defect. Anything larger than a rounding step is a real gap.
    """
    try:
        p = float(_norm(published))
        d = float(_norm(derived))
    except ValueError:
        return _norm(published) == _norm(derived)
    decimals = len(_norm(published).partition(".")[2])
    step = 10 ** -decimals
    # One full last-digit step: the live API has been revised since publication,
    # so a report saying 35,532 flights where the API now says 35,533, or 84%
    # where it now says 85%, is a data-vintage difference and not a defect.
    return abs(round(d, decimals) - p) <= step + 1e-9


def check_week(week: facts.Week, prose: str, session: requests.Session) -> dict:
    wf = facts.collect(week, session=session)
    result = {"week": week.label, "checks": [], "named": {}}

    for key, pattern in CLAIMS.items():
        m = re.search(pattern, prose, re.IGNORECASE)
        fact = wf.get(key)
        if not m:
            result["checks"].append((key, None, fact.text if fact else None, "no-claim"))
            continue
        pub = m.group(1)
        if key in SIGNED_BY_WORD:
            better, _worse = SIGNED_BY_WORD[key]
            # "3pp better" means +3; "-4pp worse" already carries its minus.
            if better in m.group(0).lower() and not pub.startswith("-"):
                pub = f"+{pub}"
        if fact is None:
            result["checks"].append((key, pub, None, "MISSING"))
            continue
        # Compare against the unrounded fact, at whatever precision the report
        # used -- fact.text is only one of the renderings it accepts.
        status = "ok" if _agrees(pub, str(fact.value)) else "MISMATCH"
        result["checks"].append((key, pub, fact.text, status))

    # Did the ranking surface every entity the analyst chose to write about?
    for kind, ranking in (("acc", wf.acc_ranking), ("airport", wf.airport_ranking)):
        top = [r["name"] for r in ranking]
        # Entities are bolded in the published bullets.
        bolded = set(re.findall(r"^\s*([A-Z][\w \-']+?(?: ACC| UAC)?)\s", prose, re.M))
        covered = [n for n in top if any(n.split()[0] in b for b in bolded)]
        result["named"][kind] = (len(covered), len(top))
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weeks", type=int, default=0, help="only the N most recent")
    args = ap.parse_args()

    session = requests.Session()
    reports = published_reports(session)
    if args.weeks:
        reports = reports[: args.weeks]

    rows, total, passed = [], 0, 0
    for rep in reports:
        week = week_from_title(rep["title"], rep["date"])
        if week is None:
            continue
        prose = _strip(rep["content"])
        try:
            res = check_week(week, prose, session)
        except Exception as exc:  # noqa: BLE001 - report, don't hide
            print(f"  {week.label}: ERROR {exc}", file=sys.stderr)
            continue

        checked = [c for c in res["checks"] if c[3] in ("ok", "MISMATCH", "MISSING")]
        ok = [c for c in checked if c[3] == "ok"]
        total += len(checked)
        passed += len(ok)
        rows.append((week, res, len(ok), len(checked)))

        flag = "OK  " if len(ok) == len(checked) else "FAIL"
        print(f"{flag} {week.label} ({week.span_text():>22})  "
              f"{len(ok)}/{len(checked)} numbers reproduced")
        for key, pub, got, status in res["checks"]:
            if status in ("MISMATCH", "MISSING"):
                print(f"       {status:8} {key:20} published={pub!r} derived={got!r}")

    print()
    if total:
        print(f"TOTAL: {passed}/{total} published numbers reproduced "
              f"({passed / total * 100:.1f}%) across {len(rows)} weeks")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
