"""Document validation and text extraction module for AI Knowledge Based Chatbot.

Supports PDF, TXT, and Markdown files. Extracts page-by-page text for PDFs
(preserving 1-based page numbers) and full UTF-8 text for TXT/Markdown documents.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
from pathlib import Path
from typing import BinaryIO, Optional
import pymupdf


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown"}


class DocumentProcessingError(Exception):
    """Base exception for document processing failures."""
    pass


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file extension is not supported."""
    pass


class EmptyFileError(DocumentProcessingError):
    """Raised when an uploaded file is empty (0 bytes or 0 pages)."""
    pass


class CorruptedFileError(DocumentProcessingError):
    """Raised when a file cannot be parsed or opened."""
    pass


class NoExtractableTextError(DocumentProcessingError):
    """Raised when a document contains no extractable text layer."""
    pass


@dataclass(frozen=True)
class ExtractedSegment:
    """Represents an extracted text segment from a document.

    Attributes:
        document_id: Deterministic identifier for the source document.
        source: Filename of the source document.
        page: 1-based page number for PDFs, or None for TXT/Markdown.
        text: Non-empty extracted text content.
    """
    document_id: str
    source: str
    page: Optional[int]
    text: str

    def to_dict(self) -> dict:
        return {
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "text": self.text,
        }


@dataclass
class DocumentResult:
    """Overall result of processing a document."""
    document_id: str
    filename: str
    file_type: str
    status: str  # "Processed" | "Failed" | "Uploaded"
    segments: list[ExtractedSegment] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_pages_or_segments(self) -> int:
        return len(self.segments)


def generate_document_id(filename: str, content: bytes) -> str:
    """Generate a deterministic document ID from filename and content bytes."""
    hasher = hashlib.sha256()
    hasher.update(filename.encode("utf-8"))
    hasher.update(b":")
    hasher.update(content)
    return hasher.hexdigest()[:16]


def validate_file(filename: str, content: bytes) -> str:
    """Validate filename extension and non-empty content.

    Returns:
        The normalized file extension (e.g., '.pdf').

    Raises:
        UnsupportedFileTypeError: If the extension is not supported.
        EmptyFileError: If the content is empty.
    """
    if not filename or not filename.strip():
        raise UnsupportedFileTypeError("Filename cannot be empty.")

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(
            f"Unsupported file type '{ext}' for file '{filename}'. Supported extensions: {supported_str}"
        )

    if not content or len(content) == 0:
        raise EmptyFileError(f"File '{filename}' is empty (0 bytes).")

    return ext


def _read_content_bytes(file_content: bytes | BinaryIO) -> bytes:
    """Read content as bytes from bytes or file-like object."""
    if isinstance(file_content, bytes):
        return file_content
    elif hasattr(file_content, "read"):
        content = file_content.read()
        if isinstance(content, str):
            return content.encode("utf-8")
        return content
    else:
        raise TypeError(f"Expected bytes or file-like object, got {type(file_content).__name__}")


def extract_text(
    file_content: bytes | BinaryIO,
    filename: str,
) -> tuple[str, list[ExtractedSegment], list[str]]:
    """Extract text from a document into structured segments.

    Args:
        file_content: Raw bytes or file-like object of the document.
        filename: Name of the file, used for extension detection and metadata.

    Returns:
        tuple of (document_id, list of ExtractedSegment, list of warnings)

    Raises:
        UnsupportedFileTypeError: If the file type is not supported.
        EmptyFileError: If the file is empty or has 0 pages.
        CorruptedFileError: If the file cannot be opened/parsed or is encrypted.
        NoExtractableTextError: If no extractable text could be found.
    """
    raw_bytes = _read_content_bytes(file_content)
    ext = validate_file(filename, raw_bytes)
    doc_id = generate_document_id(filename, raw_bytes)

    segments: list[ExtractedSegment] = []
    warnings: list[str] = []

    if ext in {".txt", ".md", ".markdown"}:
        try:
            # utf-8-sig safely handles UTF-8 with or without BOM
            text = raw_bytes.decode("utf-8-sig")
        except UnicodeDecodeError as e:
            raise CorruptedFileError(
                f"Failed to decode text file '{filename}' as UTF-8: {e}"
            ) from e

        cleaned_text = text.strip()
        if not cleaned_text:
            raise NoExtractableTextError(
                f"File '{filename}' contains only whitespace or no text content."
            )

        segments.append(
            ExtractedSegment(
                document_id=doc_id,
                source=filename,
                page=None,
                text=cleaned_text,
            )
        )

    elif ext == ".pdf":
        try:
            doc = pymupdf.open(stream=raw_bytes, filetype="pdf")
        except Exception as e:
            raise CorruptedFileError(
                f"Invalid or corrupted PDF file '{filename}': {e}"
            ) from e

        try:
            if doc.is_encrypted:
                raise CorruptedFileError(
                    f"PDF '{filename}' is password-protected or encrypted."
                )

            page_count = len(doc)
            if page_count == 0:
                raise EmptyFileError(f"PDF file '{filename}' contains 0 pages.")

            for page_idx in range(page_count):
                page = doc[page_idx]
                page_num = page_idx + 1  # 1-based page number
                page_text = page.get_text() or ""
                stripped = page_text.strip()

                if stripped:
                    segments.append(
                        ExtractedSegment(
                            document_id=doc_id,
                            source=filename,
                            page=page_num,
                            text=stripped,
                        )
                    )
                else:
                    warnings.append(
                        f"Page {page_num} of '{filename}' contains no extractable text."
                    )
        finally:
            doc.close()

        if len(segments) == 0:
            raise NoExtractableTextError(
                f"No extractable text found in PDF '{filename}'. Scanned or image-only "
                "PDFs are not supported because OCR is not implemented."
            )

    return doc_id, segments, warnings


def process_document(
    file_content: bytes | BinaryIO,
    filename: str,
) -> DocumentResult:
    """Safely process an uploaded document and return a DocumentResult.

    Catches document processing errors and encapsulates them in DocumentResult
    with status 'Failed' or 'Processed'.
    """
    raw_bytes = _read_content_bytes(file_content)
    ext = Path(filename).suffix.lower() if filename else ""
    doc_id = generate_document_id(filename, raw_bytes) if raw_bytes else "unknown"

    try:
        doc_id, segments, warnings = extract_text(raw_bytes, filename)
        return DocumentResult(
            document_id=doc_id,
            filename=filename,
            file_type=ext,
            status="Processed",
            segments=segments,
            warnings=warnings,
            error=None,
        )
    except Exception as exc:
        return DocumentResult(
            document_id=doc_id,
            filename=filename,
            file_type=ext,
            status="Failed",
            segments=[],
            warnings=[],
            error=str(exc),
        )
