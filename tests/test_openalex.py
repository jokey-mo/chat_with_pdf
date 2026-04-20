from backend.ingestion.sources.openalex import _inverted_to_text, _norm_doi, _pick_oa_pdf


def test_norm_doi_strips_prefix():
    assert _norm_doi("https://doi.org/10.1/Abc") == "10.1/abc"
    assert _norm_doi(None) is None


def test_pick_oa_pdf_prefers_best():
    work = {
        "best_oa_location": {"pdf_url": "https://a.pdf"},
        "primary_location": {"pdf_url": "https://b.pdf"},
    }
    assert _pick_oa_pdf(work) == "https://a.pdf"


def test_pick_oa_pdf_falls_back_to_locations():
    work = {"locations": [{"pdf_url": None}, {"pdf_url": "https://c.pdf"}]}
    assert _pick_oa_pdf(work) == "https://c.pdf"


def test_inverted_to_text():
    inverted = {"hello": [0], "world": [1]}
    assert _inverted_to_text(inverted) == "hello world"
    assert _inverted_to_text(None) is None
