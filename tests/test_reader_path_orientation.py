"""Test the compact Chinese reader-path orientation sentence added in Task 09.

The orientation sentence must:
- live in app/streamlit_app.py (rendered, not just source-of-truth elsewhere);
- explicitly direct visitors asking the Silverstone "what if" question to
  `比赛结论` or `自由实验`;
- explicitly direct visitors asking the wet-to-dry reliability question to
  `公开信号审计`;
- be a short, restrained piece of navigation help (not a hero / dashboard /
  modal / large block).
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "app" / "streamlit_app.py"
SOURCE = APP_PATH.read_text(encoding="utf-8")

# A short context window right after the two-module disclaimer (the
# orientation is supposed to sit between the disclaimer and the tabs).
DISCLAIMER_END = SOURCE.find("两块内容都不是通用实时决策系统")
assert DISCLAIMER_END != -1, "two-module disclaimer not found in streamlit_app.py"
# Anchor the right boundary to the st.tabs() call site, not to the tab
# labels themselves (which also appear inside the orientation sentence).
TAB_NAV = SOURCE.find("st.tabs(", DISCLAIMER_END)
assert TAB_NAV != -1, "st.tabs() call not found after disclaimer"

ORIENTATION_WINDOW = SOURCE[DISCLAIMER_END:TAB_NAV]


class ReaderPathOrientationTests(unittest.TestCase):
    def test_orientation_present_in_rendered_app(self):
        self.assertIn(
            "想探究",
            ORIENTATION_WINDOW,
            "reader-path orientation sentence not found between disclaimer and tabs",
        )

    def test_orientation_directs_to_silverstone_what_if_tabs(self):
        # The two tabs a Silverstone what-if visitor should open.
        self.assertIn("比赛结论", ORIENTATION_WINDOW)
        self.assertIn("自由实验", ORIENTATION_WINDOW)

    def test_orientation_directs_to_audit_tab(self):
        self.assertIn("公开信号审计", ORIENTATION_WINDOW)

    def test_orientation_mentions_wet_to_dry_signal_question(self):
        # The sentence should reference the audit's actual scope: whether
        # public evidence can support a wet-to-dry slick switch.
        self.assertTrue(
            re.search(r"公开.{0,12}湿转干", ORIENTATION_WINDOW)
            or "湿转干" in ORIENTATION_WINDOW,
            "orientation should reference the wet-to-dry signal question",
        )

    def test_orientation_is_restrained_not_a_hero_block(self):
        # Hard cap: the orientation sentence itself (anchored on its unique
        # opening phrase and a sentence-terminal period) should be well
        # below a hero / dashboard size. This guards against accidental
        # growth into a landing page.
        start = ORIENTATION_WINDOW.find("想探究")
        self.assertNotEqual(start, -1, "orientation opening phrase not found")
        end = ORIENTATION_WINDOW.find("。", start)
        self.assertNotEqual(end, -1, "orientation sentence not terminated")
        sentence = ORIENTATION_WINDOW[start : end + 1]
        self.assertLess(
            len(sentence),
            200,
            f"orientation sentence is {len(sentence)} chars, "
            "exceeds the restrained-help budget",
        )


if __name__ == "__main__":
    unittest.main()
