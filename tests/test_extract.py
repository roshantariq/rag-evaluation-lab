"""Tests for PDF extraction.

Block geometry is fabricated rather than loaded from real PDFs, so these
run in milliseconds and do not depend on any file in data/.
"""

from rageval.ingest.extract import (
    _order_blocks,
    classify,
    normalize_text,
    collapse_paragraphs,
    score_quality,
    split_sections,
    strip_references,
    strip_encoded_blobs,
)

PAGE_WIDTH = 595.0


def blk(x0, y0, x1, y1, text, btype=0):
    """A PyMuPDF text block tuple."""
    return (x0, y0, x1, y1, text, 0, btype)


class TestColumnOrdering:
    def test_two_column_page_reads_down_each_column(self):
        page = [
            blk(60, 50, 535, 90, "TITLE"),
            blk(60, 120, 285, 200, "L1"), blk(310, 128, 535, 210, "R1"),
            blk(60, 215, 285, 300, "L2"), blk(310, 220, 535, 305, "R2"),
            blk(60, 310, 285, 400, "L3"), blk(310, 315, 535, 405, "R3"),
        ]
        assert [b[4] for b in _order_blocks(page, PAGE_WIDTH)] == [
            "TITLE", "L1", "L2", "L3", "R1", "R2", "R3"
        ]

    def test_single_column_order_is_preserved(self):
        page = [blk(60, 100 + i * 80, 535, 170 + i * 80, f"P{i}") for i in range(5)]
        assert [b[4] for b in _order_blocks(page, PAGE_WIDTH)] == [f"P{i}" for i in range(5)]

    def test_stray_sidebar_does_not_trigger_column_split(self):
        page = [
            blk(60, 100, 535, 200, "A"), blk(60, 210, 535, 300, "B"),
            blk(60, 310, 535, 400, "C"), blk(60, 410, 535, 500, "D"),
            blk(420, 510, 535, 560, "sidebar"),
        ]
        assert [b[4] for b in _order_blocks(page, PAGE_WIDTH)] == ["A", "B", "C", "D", "sidebar"]

    def test_images_and_blank_blocks_are_dropped(self):
        page = [blk(60, 100, 535, 200, "keep"), blk(60, 210, 535, 300, "", 1),
                blk(60, 310, 535, 400, "   ")]
        assert [b[4] for b in _order_blocks(page, PAGE_WIDTH)] == ["keep"]

    def test_empty_page_returns_empty(self):
        assert _order_blocks([], PAGE_WIDTH) == []


class TestCleaning:
    def test_ligatures_are_expanded(self):
        assert normalize_text("the classiﬁer uses aﬃne maps") == "the classifier uses affine maps"

    def test_line_break_hyphenation_is_rejoined(self):
        assert normalize_text("convo-\nlutional network") == "convolutional network"

    def test_normalize_preserves_line_structure(self):
        # Regression test for the bug that made every paper report one section:
        # collapsing newlines before parsing structure hides every heading.
        assert "\n" in normalize_text("1 Introduction\nBody text follows here.")

    def test_collapse_turns_soft_wraps_into_spaces(self):
        assert collapse_paragraphs("one line\nsame para") == "one line same para"

    def test_collapse_keeps_paragraph_breaks(self):
        assert collapse_paragraphs("para one\n\npara two") == "para one\n\npara two"


class TestPipelineOrder:
    PAPER = (
        "Abstract\nWe present a model.\n\n"
        "1 Introduction\nNowcasting matters a great deal in operations.\n\n"
        "2.1 Model Architecture\nWe use ConvLSTM layers throughout.\n\n"
        "3 Experiments\nWe evaluate on radar data from 2015 onward.\n\n"
        + "Body filler sentence for length. " * 100
        + "\nReferences\n[1] Shi et al. 2015. Convolutional LSTM Network."
    )

    def test_headings_survive_normalisation(self):
        body, _ = strip_references(normalize_text(self.PAPER))
        headings = [s.heading for s in split_sections(body)]
        assert "1 Introduction" in headings
        assert "2.1 Model Architecture" in headings

    def test_references_are_found_on_normalised_text(self):
        _, found = strip_references(normalize_text(self.PAPER))
        assert found is True


class TestReferenceStripping:
    def test_reference_list_is_removed(self):
        text = "Body text. " * 300 + "\nReferences\n[1] Shi et al. 2015."
        out, found = strip_references(text)
        assert found and "Shi et al" not in out

    def test_early_heading_is_not_treated_as_the_reference_list(self):
        # A "References" line in the first 40% is far more likely to be a
        # cross-reference than the actual bibliography.
        _, found = strip_references("References\n" + "body " * 500)
        assert found is False

    def test_appendix_survives_bibliography_removal(self):
        text = ("Body sentence with real content. " * 200
                + "\nReferences\n" + "[1] Gu and Dao. Mamba. arXiv 2312.00752, 2023.\n" * 40
                + "\nAppendix\nA Data Details\nWe collected ERA5 data for the region.")
        out, found = strip_references(text)
        assert found is True
        assert "arXiv 2312.00752" not in out      # bibliography gone
        assert "We collected ERA5 data" in out    # appendix kept

    def test_early_bibliography_accepted_when_appendix_follows(self):
        # A 45-page paper with 36 pages of figure appendices puts its
        # bibliography around 8% of the way through the text. A percentage
        # guard rejects that; an absolute body-length guard accepts it.
        body = "Short body sentence with real content. " * 200  # ~2 pages
        text = (body
                + "\nReferences\n" + "[1] Lam et al. GraphCast. 2022.\n" * 20
                + "\nAppendix\n" + "Figure A1 caption text here. " * 400)
        out, found = strip_references(text)
        assert found is True
        assert "GraphCast" not in out                  # bibliography gone
        assert "Figure A1 caption" in out              # appendix kept

    def test_bibliography_marker_with_no_real_body_is_rejected(self):
        # Guards against a "References" line in front matter or an abstract.
        text = ("Short body. " * 100
                + "\nReferences\n" + "[1] Lam et al. 2022.\n" * 20
                + "\nAppendix\n" + "caption. " * 400)
        _, found = strip_references(text)
        assert found is False


class TestSections:
    def test_numbered_and_named_headings_both_split(self):
        doc = (
            "Abstract\nWe present a model.\n\n"
            "1 Introduction\nNowcasting matters.\n\n"
            "2.1 Model Architecture\nWe use ConvLSTM layers.\n\n"
            "3 Experiments\nWe evaluate on radar data."
        )
        headings = [s.heading for s in split_sections(doc)]
        assert "Abstract" in headings
        assert "1 Introduction" in headings
        assert "2.1 Model Architecture" in headings

    def test_document_without_headings_returns_one_section(self):
        secs = split_sections("just some flowing prose with no structure at all")
        assert len(secs) == 1 and secs[0].heading == "(untitled)"


class TestQualityScoring:
    PROSE = ("The convolutional recurrent network predicts precipitation "
             "fields from radar reflectivity observations. ")

    def test_normal_paper_is_clean(self):
        assert classify(score_quality(self.PROSE * 260, 8), 6) == "clean"

    def test_sparse_extraction_is_degraded(self):
        assert classify(score_quality(self.PROSE * 40, 8), 6) == "degraded"

    def test_cid_garbage_is_failed(self):
        assert classify(score_quality("(cid:12)(cid:45) " * 400, 8), 6) == "failed"

    def test_near_empty_is_failed(self):
        assert classify(score_quality("short", 1), 1) == "failed"

    def test_equation_soup_is_degraded(self):
        eq = "where x t = f ( W h t - 1 + b ) and = 0 . 5 , [ 0 , 1 ] . " * 200
        assert classify(score_quality(eq, 8), 6) == "degraded"

class TestEncodedBlobs:
    BLOB = '1_base64="k7S9hMrk67oWzTDFSMw89rgny8=">AB7HicbVBNTwIxEJ3FL8Qv1KOX'

    def test_base64_image_data_is_removed(self):
        out = strip_encoded_blobs(f"See Figure 3 {self.BLOB} for details.")
        assert "base64" not in out
        assert out == "See Figure 3 for details."

    def test_png_header_is_removed(self):
        png = "iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAACXBIWXMAAA7EAAAOxAGVKw4b"
        assert strip_encoded_blobs(f"x {png} y") == "x y"

    def test_long_legitimate_tokens_survive(self):
        # Vowel frequency cannot separate these from base64; case-flip rate can.
        for token in (
            "https://doi.org/10.48550/arXiv.2304.02948",
            "https://www.ecmwf.int/en/forecasts/datasets/set-i",
            "ConvLSTMEncoderDecoderForecastingNetwork",
            "ERA5_reanalysis_500hPa_geopotential_1979_2018_daily",
            "Latitude-weighted-Root-Mean-Square-Error",
        ):
            assert strip_encoded_blobs(token) == token

    def test_line_structure_is_preserved(self):
        # Regression guard: collapsing newlines here would re-introduce the
        # bug that made every paper report a single section.
        assert "\n" in strip_encoded_blobs("1 Introduction\nBody text here.")
        assert strip_encoded_blobs(f"A\n{self.BLOB}\nB").count("\n") == 2

    def test_urls_with_random_looking_paths_survive(self):
        # Found in the corpus: a Google Forms link in FuXi (2306.12873v3) was
        # deleted because its random ID cleared both heuristics.
        url = "google.com/forms/d/e/1FAIpQLSfjwZLf6PmxRvRhIPMQ1WRLJ98iLxOq"
        assert strip_encoded_blobs(url) == url