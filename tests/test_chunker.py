from backend.parsing.chunker import section_aware_chunks
from backend.parsing.docling_parser import ParsedDoc, ParsedSection


def test_section_aware_chunks_basic():
    parsed = ParsedDoc(
        markdown="",
        sections=[
            ParsedSection(heading="Intro", text="Hello world " * 50),
            ParsedSection(heading="Methods", text="Procedure details " * 50),
        ],
    )
    chunks = section_aware_chunks(parsed, max_tokens=50, overlap=5)
    assert len(chunks) >= 2
    assert all(c.token_count > 0 for c in chunks)
    assert {c.section for c in chunks} == {"Intro", "Methods"}


def test_section_aware_chunks_splits_long():
    parsed = ParsedDoc(
        markdown="",
        sections=[ParsedSection(heading="Big", text="word " * 2000)],
    )
    chunks = section_aware_chunks(parsed, max_tokens=200, overlap=20)
    assert len(chunks) > 1
