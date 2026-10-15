"""Unit tests for text chunking module.

Verifies structure-aware chunking:
1. Default parameters (500 chunk size, 100 overlap).
2. Paragraph boundaries preferred.
3. Long paragraphs split at sentence boundaries.
4. Extremely long sentences split at word boundaries.
5. Heading and list preservation.
6. PDF page isolation and metadata fidelity.
7. Deterministic chunk IDs and content ordering.
8. Overlap correctness without excessive duplication.
9. Safe handling of empty, whitespace, and Unicode text.
"""

import pytest
from src.chunker import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    TextChunk,
    chunk_document,
    chunk_segment,
    chunk_text_content,
    validate_chunk_params,
)
from src.document_processor import ExtractedSegment


# 1. Default chunk size is 500 words and default overlap is 100 words.
def test_default_chunk_parameters():
    assert DEFAULT_CHUNK_SIZE == 500
    assert DEFAULT_CHUNK_OVERLAP == 100

    # Running with default arguments on >500 words splits with 500 max size
    words = [f"word{i}" for i in range(750)]
    text = " ".join(words)
    chunks = chunk_text_content(text)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= DEFAULT_CHUNK_SIZE


# 2. Paragraph boundaries are preferred when creating chunks.
def test_paragraph_boundaries_preferred():
    # P1 (180 words), P2 (180 words), P3 (200 words)
    # Total = 560 words. chunk_size = 500, overlap = 100
    p1 = " ".join(["FirstParaWord"] * 179) + " FirstParaEnd."
    p2 = " ".join(["SecondParaWord"] * 179) + " SecondParaEnd."
    p3 = " ".join(["ThirdParaWord"] * 199) + " ThirdParaEnd."
    full_text = f"{p1}\n\n{p2}\n\n{p3}"

    chunks = chunk_text_content(full_text, chunk_size=500, chunk_overlap=100)

    assert len(chunks) >= 2
    # Chunk 0 should hold P1 and P2 completely, ending at the boundary of P2
    assert "FirstParaEnd." in chunks[0]
    assert "SecondParaEnd." in chunks[0]
    # P3 should not be cut into pieces in Chunk 0; it should be kept for Chunk 1
    assert "ThirdParaWord" not in chunks[0]

    # Chunk 1 should contain P3 intact
    assert "ThirdParaEnd." in chunks[1]


# 3. Long paragraphs are split at sentence boundaries where possible.
def test_long_paragraphs_split_at_sentence_boundaries():
    # Single paragraph with 8 distinct sentences of 70 words each (total 560 words)
    sentences = [
        f"This is sentence number {i} containing information about distributed database architectures "
        + " ".join([f"detail{i}_{w}" for w in range(56)])
        + f" end of sentence {i}."
        for i in range(8)
    ]
    paragraph_text = " ".join(sentences)

    chunks = chunk_text_content(paragraph_text, chunk_size=400, chunk_overlap=80)

    assert len(chunks) >= 2
    # Verify that Chunk 0 ends at a clean sentence boundary (with a period)
    c0 = chunks[0].strip()
    assert c0.endswith(".")
    # Ensure it ends on one of the sentence endings
    assert any(c0.endswith(f"end of sentence {i}.") for i in range(8))

    # Verify that Chunk 1 begins with a clean sentence boundary
    c1 = chunks[1].strip()
    assert any(c1.startswith(f"This is sentence number {i}") for i in range(8))


# 4. Extremely long sentences are safely split at word boundaries.
def test_extremely_long_sentence_split_at_word_boundaries():
    # Single 650-word sentence with no sentence-ending punctuation
    words = [f"monolithicToken{i}" for i in range(650)]
    long_sentence = " ".join(words)

    chunks = chunk_text_content(long_sentence, chunk_size=500, chunk_overlap=100)

    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 500

    # Verify monotonic ordering across chunks
    all_reconstructed_words = []
    for c in chunks:
        c_words = c.split()
        indices = [int(w.replace("monolithicToken", "")) for w in c_words]
        assert indices == sorted(indices)
        all_reconstructed_words.extend(c_words)

    # First and last tokens must be preserved
    assert "monolithicToken0" in chunks[0]
    assert "monolithicToken649" in chunks[-1]


# 5. Headings and their content are kept together where reasonably possible.
def test_headings_kept_with_following_content():
    # Create text where Section 1 fills most of a chunk, followed by a heading for Section 2 and its body
    p1 = " ".join(["IntroWord"] * 460) + " End of introduction."
    heading2 = "## Section 2: Concurrency Control and ACID Transactions"
    p2 = " ".join(["DatabaseWord"] * 80) + " Concurrency control ensures serializability."

    full_text = f"{p1}\n\n{heading2}\n\n{p2}"
    chunks = chunk_text_content(full_text, chunk_size=500, chunk_overlap=100)

    assert len(chunks) >= 2
    # The heading should NOT be left stranded as the last item of Chunk 0
    assert not chunks[0].strip().endswith("## Section 2: Concurrency Control and ACID Transactions")
    # Instead, the heading must precede Section 2 content in Chunk 1
    assert heading2 in chunks[1]
    assert p2 in chunks[1]


# 6. Numbered steps and bullet lists are preserved with structured formatting.
def test_numbered_steps_and_bullet_lists_preserved():
    list_text = (
        "Prerequisites for distributed systems:\n\n"
        "1. Clock synchronization using NTP or vector clocks.\n"
        "2. Consensus algorithms such as Paxos or Raft.\n"
        "3. Reliable point-to-point and broadcast communication primitives.\n"
        "4. Failure detection mechanisms and heartbeat timeouts."
    )
    chunks = chunk_text_content(list_text, chunk_size=500, chunk_overlap=100)

    assert len(chunks) == 1
    chunk = chunks[0]
    # Check that numbered list items are separated by newlines, not flattened into single line
    assert "1. Clock synchronization" in chunk
    assert "2. Consensus algorithms" in chunk
    assert "3. Reliable point-to-point" in chunk
    assert "4. Failure detection" in chunk
    assert "\n" in chunk


# 7. Overlap is approximately correct and avoids excessive duplication.
def test_overlap_approximately_correct_and_no_excessive_duplication():
    # 6 paragraphs of 150 words each (900 words)
    paragraphs = [
        f"Paragraph {i} content. " + " ".join([f"term_{i}_{w}" for w in range(145)]) + f" End of para {i}."
        for i in range(6)
    ]
    full_text = "\n\n".join(paragraphs)

    chunks = chunk_text_content(full_text, chunk_size=500, chunk_overlap=100)

    assert len(chunks) >= 2
    # Ensure no two chunks are identical
    for i in range(len(chunks) - 1):
        assert chunks[i] != chunks[i + 1]
        c_i_words = set(chunks[i].split())
        c_next_words = set(chunks[i + 1].split())
        overlap_words = c_i_words.intersection(c_next_words)
        # There should be overlap, but it shouldn't be the entire chunk
        assert len(overlap_words) > 0
        assert len(overlap_words) < len(c_i_words)


# 8. Empty input, whitespace-only input, Unicode text handled safely.
def test_edge_cases_empty_whitespace_unicode():
    # Empty string
    assert chunk_text_content("", chunk_size=100, chunk_overlap=20) == []
    # Whitespace only
    assert chunk_text_content("   \n\t\r\n   ", chunk_size=100, chunk_overlap=20) == []

    # Unicode with math symbols, smart quotes, accented characters, emojis
    unicode_text = (
        "Calculus and Logic:\n\n"
        "Theorem 1: If f(x) = ∫₀ˣ g(t)dt, then f'(x) = g(x) ∀ x ∈ ℝ. "
        "“Smart quotes” and ‘apostrophes’ are common in academic texts. "
        "Also accented letters like café and résumé, plus emojis 🤖 📚."
    )
    unicode_chunks = chunk_text_content(unicode_text, chunk_size=500, chunk_overlap=100)

    assert len(unicode_chunks) == 1
    assert "∫₀ˣ" in unicode_chunks[0]
    assert "“Smart quotes”" in unicode_chunks[0]
    assert "café and résumé" in unicode_chunks[0]
    assert "🤖 📚" in unicode_chunks[0]


# 9. Invalid chunk-size and overlap settings raise clear errors.
def test_invalid_chunk_settings_raise_errors():
    with pytest.raises(ValueError) as exc1:
        validate_chunk_params(chunk_size=0, chunk_overlap=0)
    assert "positive" in str(exc1.value)

    with pytest.raises(ValueError) as exc2:
        validate_chunk_params(chunk_size=-10, chunk_overlap=5)
    assert "positive" in str(exc2.value)

    with pytest.raises(ValueError) as exc3:
        validate_chunk_params(chunk_size=100, chunk_overlap=-1)
    assert "non-negative" in str(exc3.value)

    with pytest.raises(ValueError) as exc4:
        validate_chunk_params(chunk_size=100, chunk_overlap=100)
    assert "strictly less" in str(exc4.value)

    with pytest.raises(ValueError) as exc5:
        validate_chunk_params(chunk_size=100, chunk_overlap=150)
    assert "strictly less" in str(exc5.value)


# 10. Chunk metadata preserves document identity, source, and page number.
def test_chunk_metadata_preservation():
    segment = ExtractedSegment(
        document_id="doc_xyz_123",
        source="report.pdf",
        page=4,
        text="Sample text for testing metadata retention across chunks.",
    )
    chunks = chunk_segment(segment, chunk_size=5, chunk_overlap=1)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_id == "doc_xyz_123"
        assert chunk.source == "report.pdf"
        assert chunk.page == 4
        assert chunk.chunk_id.startswith("doc_xyz_123_p4_")


# 11. PDF pages remain separate and their page attribution is correct.
def test_pdf_pages_remain_separate():
    seg1 = ExtractedSegment(
        document_id="doc_pages",
        source="manual.pdf",
        page=1,
        text="Content strictly belonging to Page 1 of the manual.",
    )
    seg2 = ExtractedSegment(
        document_id="doc_pages",
        source="manual.pdf",
        page=2,
        text="Content strictly belonging to Page 2 of the manual.",
    )

    chunks = chunk_document([seg1, seg2], chunk_size=10, chunk_overlap=2)
    assert len(chunks) == 2

    assert chunks[0].page == 1
    assert "Page 1" in chunks[0].text
    assert "Page 2" not in chunks[0].text

    assert chunks[1].page == 2
    assert "Page 2" in chunks[1].text
    assert "Page 1" not in chunks[1].text


# 12. Chunk IDs are deterministic for identical input.
def test_chunk_ids_are_deterministic():
    seg = ExtractedSegment(
        document_id="stable_doc_id",
        source="doc.txt",
        page=None,
        text="Repeated input text to verify deterministic ID generation.",
    )

    chunks_first_run = chunk_segment(seg, chunk_size=4, chunk_overlap=1)
    chunks_second_run = chunk_segment(seg, chunk_size=4, chunk_overlap=1)

    assert len(chunks_first_run) == len(chunks_second_run)
    for c1, c2 in zip(chunks_first_run, chunks_second_run):
        assert c1.chunk_id == c2.chunk_id
        assert c1.text == c2.text


# 13. Chunk text preserves the original content order.
def test_chunk_text_preserves_content_order():
    ordered_words = [f"token_{i:02d}" for i in range(30)]
    full_text = " ".join(ordered_words)

    chunks = chunk_text_content(full_text, chunk_size=10, chunk_overlap=3)

    for c in chunks:
        c_words = c.split()
        indices = [int(w.split("_")[1]) for w in c_words]
        assert indices == sorted(indices)
        assert indices == list(range(indices[0], indices[-1] + 1))


# 14. Long text without sentence punctuation is split into multiple chunks.
def test_long_text_is_split_into_multiple_chunks():
    words = [f"word{i}" for i in range(25)]
    text = " ".join(words)
    chunks = chunk_text_content(text, chunk_size=10, chunk_overlap=2)

    assert len(chunks) == 3


# 15. Chunk overlap behaves as configured for word-level slices.
def test_chunk_overlap_behaves_as_configured():
    words = [f"w{i}" for i in range(15)]
    text = " ".join(words)
    chunks = chunk_text_content(text, chunk_size=6, chunk_overlap=2)
    assert len(chunks) == 4

    c0_words = chunks[0].split()
    c1_words = chunks[1].split()

    assert c0_words[-2:] == c1_words[:2]
    assert c1_words[:2] == ["w4", "w5"]


# 16. Small text produces a single chunk.
def test_small_text_produces_single_chunk():
    text = "This is a short sentence of only eight words."
    chunks = chunk_text_content(text, chunk_size=50, chunk_overlap=10)

    assert len(chunks) == 1
    assert chunks[0] == text
