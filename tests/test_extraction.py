"""Unit tests for document validation and text extraction (Phase 2)."""

import io
import pytest
import pymupdf
from src.document_processor import (
    CorruptedFileError,
    EmptyFileError,
    ExtractedSegment,
    NoExtractableTextError,
    UnsupportedFileTypeError,
    extract_text,
    generate_document_id,
    process_document,
    validate_file,
)


def create_in_memory_pdf(pages: list[str]) -> bytes:
    """Helper to generate a multi-page PDF locally using PyMuPDF."""
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((50, 72), text)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


# 1. TXT extraction returns the expected content.
def test_txt_extraction():
    content = b"This is a sample plain text document for RAG ingestion."
    doc_id, segments, warnings = extract_text(content, "sample.txt")

    assert len(segments) == 1
    assert segments[0].text == "This is a sample plain text document for RAG ingestion."
    assert segments[0].source == "sample.txt"
    assert segments[0].page is None
    assert segments[0].document_id == doc_id
    assert len(warnings) == 0


# 2. Markdown extraction returns the expected content.
def test_markdown_extraction():
    content = b"# Heading 1\n\nThis is paragraph content in markdown format.\n\n- Bullet item"
    doc_id, segments, warnings = extract_text(content, "notes.md")

    assert len(segments) == 1
    assert "Heading 1" in segments[0].text
    assert "Bullet item" in segments[0].text
    assert segments[0].page is None
    assert segments[0].source == "notes.md"


# 3. UTF-8 BOM is handled correctly.
def test_utf8_bom_handling():
    # Byte Order Mark for UTF-8 is \xef\xbb\xbf
    bom_content = "\ufeffDocument text with UTF-8 BOM encoding.".encode("utf-8")
    doc_id, segments, warnings = extract_text(bom_content, "bom_test.txt")

    assert len(segments) == 1
    # Ensure the BOM character \ufeff was stripped by utf-8-sig
    assert not segments[0].text.startswith("\ufeff")
    assert segments[0].text == "Document text with UTF-8 BOM encoding."


# 4. PDF extraction preserves correct 1-based page numbers.
def test_pdf_extraction_page_numbers():
    pdf_bytes = create_in_memory_pdf(["Content on first page", "Content on second page"])
    doc_id, segments, warnings = extract_text(pdf_bytes, "multi_page.pdf")

    assert len(segments) == 2
    assert segments[0].page == 1
    assert "Content on first page" in segments[0].text
    assert segments[1].page == 2
    assert "Content on second page" in segments[1].text
    assert segments[0].document_id == doc_id
    assert segments[1].document_id == doc_id


# 5. Empty files are rejected.
def test_empty_files_rejected():
    with pytest.raises(EmptyFileError):
        extract_text(b"", "empty.txt")

    with pytest.raises(EmptyFileError):
        extract_text(b"", "empty.pdf")

    # Whitespace only txt
    with pytest.raises(NoExtractableTextError):
        extract_text(b"   \n  \t  ", "spaces.txt")


# 6. Invalid or corrupted PDFs fail gracefully.
def test_invalid_or_corrupted_pdf_handling():
    corrupted_bytes = b"This is definitely not a valid PDF file stream."
    with pytest.raises(CorruptedFileError):
        extract_text(corrupted_bytes, "corrupted.pdf")

    result = process_document(corrupted_bytes, "corrupted.pdf")
    assert result.status == "Failed"
    assert "Invalid or corrupted PDF" in result.error


# 7. Unsupported extensions are rejected.
def test_unsupported_extensions_rejected():
    content = b"Some data"
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(content, "spreadsheet.xlsx")

    with pytest.raises(UnsupportedFileTypeError):
        extract_text(content, "script.py")

    result = process_document(content, "image.png")
    assert result.status == "Failed"
    assert "Unsupported file type" in result.error


# 8. PDFs with blank pages and readable pages preserve the readable text.
def test_pdf_with_blank_pages_preserves_readable_text():
    pdf_bytes = create_in_memory_pdf(
        ["Page one valid text.", "", "Page three valid text."]
    )
    doc_id, segments, warnings = extract_text(pdf_bytes, "partially_blank.pdf")

    # Should retain exactly 2 segments for the 2 readable pages
    assert len(segments) == 2
    assert segments[0].page == 1
    assert "Page one valid text." in segments[0].text
    assert segments[1].page == 3
    assert "Page three valid text." in segments[1].text

    # Should issue a warning for the blank page
    assert len(warnings) == 1
    assert "Page 2" in warnings[0]


# 9. PDFs without an extractable text layer do not falsely report successful extraction.
def test_pdf_without_text_layer_fails():
    # Multi-page PDF where all pages are completely blank
    pdf_bytes = create_in_memory_pdf(["", ""])
    with pytest.raises(NoExtractableTextError) as exc_info:
        extract_text(pdf_bytes, "scanned_doc.pdf")

    assert "OCR" in str(exc_info.value)

    result = process_document(pdf_bytes, "scanned_doc.pdf")
    assert result.status == "Failed"
    assert "OCR" in result.error


def test_stable_document_id():
    content1 = b"Identical content"
    content2 = b"Identical content"
    content3 = b"Different content"

    id1 = generate_document_id("file.txt", content1)
    id2 = generate_document_id("file.txt", content2)
    id3 = generate_document_id("file.txt", content3)

    assert id1 == id2
    assert id1 != id3
