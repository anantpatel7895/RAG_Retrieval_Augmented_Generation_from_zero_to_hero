"""Tests for the format probes.

These exist because the measurement logic is load-bearing: `docs/PARSING.md` quotes
its output, and one of these metrics was measured wrong twice before it was right.
A regex that silently stops matching is exactly the kind of failure that produces a
plausible number nobody questions.
"""

from __future__ import annotations

import re

from rag.ingest.format_probe import (
    AMOUNT,
    cell_texts,
    strip_inline_aware,
    strip_naive,
)


class TestAmountPattern:
    """The pattern that took three attempts to get right."""

    def test_matches_plain_integer(self) -> None:
        assert re.fullmatch(AMOUNT, "1234")

    def test_matches_thousands_separators(self) -> None:
        assert re.fullmatch(AMOUNT, "794,433,813,496")

    def test_matches_decimals(self) -> None:
        # The original `[\d,]{4,}` missed these, which silently excluded most
        # amounts in the corpus and inflated the "split" count.
        assert re.fullmatch(AMOUNT, "4.4")
        assert re.fullmatch(AMOUNT, "362.4")

    def test_old_broken_pattern_would_miss_decimals(self) -> None:
        """Regression guard: documents *why* the pattern looks the way it does."""
        broken = r"[\d,]{4,}"
        assert re.fullmatch(broken, "794,433,813,496")
        assert not re.fullmatch(broken, "4.4")


class TestStripStrategies:
    def test_naive_separates_across_inline_tags(self) -> None:
        html = "<span>$</span><ix:nonFraction>1,234</ix:nonFraction>"
        assert "$ 1,234" in strip_naive(html)

    def test_inline_aware_keeps_adjacency(self) -> None:
        html = "<span>$</span><ix:nonFraction>1,234</ix:nonFraction>"
        assert "$1,234" in strip_inline_aware(html)

    def test_inline_aware_still_breaks_across_cells(self) -> None:
        """The finding that drives Phase 2: a cell boundary is a real separation.

        No text-level strategy can fix this, which is why grid-aware table parsing
        is required rather than optional.
        """
        html = "<tr><td>$</td><td>1,234</td></tr>"
        assert "$1,234" not in strip_inline_aware(html)

    def test_entities_become_whitespace(self) -> None:
        assert "a b" in strip_inline_aware("a&#160;b")


class TestCellTexts:
    def test_extracts_and_trims(self) -> None:
        html = "<tr><td> $ </td><td><span>1,234</span></td></tr>"
        assert cell_texts(html) == ["$", "1,234"]

    def test_ignores_self_closing_spacers(self) -> None:
        """Self-closing <td/> spacers are ~62% of cells; they are counted separately."""
        assert cell_texts("<tr><td/><td>x</td><td/></tr>") == ["x"]

    def test_dollar_only_cell_is_detectable(self) -> None:
        cells = cell_texts("<tr><td>$</td><td>1,234</td><td>Total</td></tr>")
        assert cells[0] == "$"
        assert re.fullmatch(AMOUNT, cells[1])
