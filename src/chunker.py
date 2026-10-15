"""Document chunking module for AI Knowledge Based Chatbot.

Splits extracted document text into overlapping chunks while strictly preserving
source metadata (document ID, filename, and PDF page attribution).
Implements structure-aware chunking:
1. Prefers paragraph boundaries.
2. Splits long paragraphs at sentence boundaries.
3. Splits long sentences at word boundaries as a last resort.
4. Preserves headings and lists with their following content.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional, Sequence
from src.document_processor import ExtractedSegment


DEFAULT_CHUNK_SIZE = 500  # Target maximum chunk size in approximate tokens (words)
DEFAULT_CHUNK_OVERLAP = 100  # Target overlap in approximate tokens (words)


@dataclass(frozen=True)
class TextChunk:
    """Represents a text chunk created from an extracted document segment.

    Attributes:
        chunk_id: Deterministic identifier for the chunk.
        document_id: Identifier of the parent document.
        source: Filename of the parent document.
        page: 1-based page number for PDFs, or None for TXT/Markdown.
        chunk_index: 0-based sequential index of the chunk within the document.
        text: Non-empty chunk text content.
    """
    chunk_id: str
    document_id: str
    source: str
    page: Optional[int]
    chunk_index: int
    text: str

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "source": self.source,
            "page": self.page,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


def validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap parameters.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer greater than 0, got {chunk_size}"
        )
    if chunk_overlap < 0:
        raise ValueError(
            f"chunk_overlap must be a non-negative integer (>= 0), got {chunk_overlap}"
        )
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})"
        )


def _count_words(text: str) -> int:
    """Return the count of whitespace-delimited words."""
    if not text:
        return 0
    return len(text.split())


# Known abbreviations to protect from false sentence splits
_ABBREVIATIONS = (
    r"e\.g|i\.e|etc|vs|fig|figs|dr|mr|mrs|ms|prof|al|dept|approx|inc|ltd|co|no|vol|p|pp"
)


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences while respecting common abbreviations, numbers, and quotes."""
    if not text or not text.strip():
        return []

    # Mask abbreviations: replace dot with null byte
    def _mask_abbrev(m: re.Match) -> str:
        return m.group(0).replace(".", "\x00")

    masked = re.sub(rf"\b({_ABBREVIATIONS})\.", _mask_abbrev, text, flags=re.IGNORECASE)
    # Mask decimal points in numbers (e.g. 3.14)
    masked = re.sub(r"(?<=\d)\.(?=\d)", "\x00", masked)

    # Split on sentence terminators (. ! ? ...) followed by whitespace and a capital letter / quote / bracket
    sentence_delimiters = re.compile(
        r'(?<=[.!?…])\s+(?=[A-Z0-9"\'\(\[\u201C\u2018])'
    )
    raw_parts = sentence_delimiters.split(masked)

    sentences: list[str] = []
    for part in raw_parts:
        unmasked = part.replace("\x00", ".").strip()
        if unmasked:
            sentences.append(unmasked)

    return sentences if sentences else [text.strip()]


def _is_markdown_heading(line: str) -> bool:
    """Check if a line is a markdown heading (#, ##, ###, etc.)."""
    return bool(re.match(r"^#{1,6}\s+\S+", line.strip()))


def _is_short_heading(line: str) -> bool:
    """Check if a line is a section heading, title, or definition label."""
    s = line.strip()
    if not s or "\n" in s:
        return False
    words = s.split()
    if len(words) > 12:
        return False
    if _is_markdown_heading(s):
        return True
    if s.endswith(":") and len(words) <= 10:
        return True
    if re.match(r"^(?:Section\s+\d+|Chapter\s+\d+|\d+(?:\.\d+)*)\b", s, re.IGNORECASE):
        return True
    if s.isupper() and len(s) > 2 and len(words) <= 6:
        return True
    return False


def _is_list_item(line: str) -> bool:
    """Check if a line starts a list item (bullet or numbered)."""
    return bool(
        re.match(r"^([-\*+•]|\d+[\.\)]|\([0-9a-zA-Z]\)|[a-zA-Z][\.\)])\s+", line.strip())
    )


@dataclass
class _ChunkUnit:
    """Internal structural unit used for packing chunks."""
    text: str
    words: int
    paragraph_id: int
    is_heading: bool = False
    is_paragraph_start: bool = True
    is_list_item: bool = False


def _decompose_text(text: str, max_chunk_size: int) -> list[_ChunkUnit]:
    """Decompose text into atomic structural units respecting paragraphs, sentences, and words."""
    cleaned = text.strip()
    if not cleaned:
        return []

    raw_blocks = re.split(r"\n\s*\n+", cleaned)
    units: list[_ChunkUnit] = []
    current_para_id = 0

    for block in raw_blocks:
        block_clean = block.strip()
        if not block_clean:
            continue

        is_heading_block = _is_short_heading(block_clean)

        # Check if block contains multiple list items
        lines = [line.strip() for line in block_clean.split("\n") if line.strip()]
        if len(lines) > 1 and any(_is_list_item(line) for line in lines):
            is_first_in_block = True
            for line in lines:
                line_words = _count_words(line)
                if line_words <= max_chunk_size:
                    units.append(
                        _ChunkUnit(
                            text=line,
                            words=line_words,
                            paragraph_id=current_para_id,
                            is_heading=_is_short_heading(line),
                            is_paragraph_start=is_first_in_block,
                            is_list_item=_is_list_item(line),
                        )
                    )
                    is_first_in_block = False
                else:
                    # Item exceeds max_chunk_size -> split into sentences
                    sents = _split_into_sentences(line)
                    for s_idx, sent in enumerate(sents):
                        sent_words = _count_words(sent)
                        if sent_words <= max_chunk_size:
                            units.append(
                                _ChunkUnit(
                                    text=sent,
                                    words=sent_words,
                                    paragraph_id=current_para_id,
                                    is_heading=False,
                                    is_paragraph_start=is_first_in_block and (s_idx == 0),
                                    is_list_item=_is_list_item(sent),
                                )
                            )
                        else:
                            # Sentence exceeds max_chunk_size -> split into individual words
                            words = sent.split()
                            for w_idx, word in enumerate(words):
                                units.append(
                                    _ChunkUnit(
                                        text=word,
                                        words=1,
                                        paragraph_id=current_para_id,
                                        is_heading=False,
                                        is_paragraph_start=is_first_in_block and (s_idx == 0) and (w_idx == 0),
                                        is_list_item=False,
                                    )
                                )
                    is_first_in_block = False
            current_para_id += 1
            continue

        # Normal prose paragraph or heading
        sentences = _split_into_sentences(block_clean)
        if len(sentences) > 1:
            for s_idx, sent in enumerate(sentences):
                sent_words = _count_words(sent)
                if sent_words <= max_chunk_size:
                    units.append(
                        _ChunkUnit(
                            text=sent,
                            words=sent_words,
                            paragraph_id=current_para_id,
                            is_heading=(s_idx == 0 and is_heading_block),
                            is_paragraph_start=(s_idx == 0),
                            is_list_item=_is_list_item(sent),
                        )
                    )
                else:
                    # Giant sentence -> split into individual words
                    words = sent.split()
                    for w_idx, word in enumerate(words):
                        units.append(
                            _ChunkUnit(
                                text=word,
                                words=1,
                                paragraph_id=current_para_id,
                                is_heading=False,
                                is_paragraph_start=(s_idx == 0 and w_idx == 0),
                                is_list_item=False,
                            )
                        )
        else:
            # Single sentence or block
            sent = sentences[0] if sentences else block_clean
            sent_words = _count_words(sent)
            if sent_words <= max_chunk_size:
                units.append(
                    _ChunkUnit(
                        text=sent,
                        words=sent_words,
                        paragraph_id=current_para_id,
                        is_heading=is_heading_block,
                        is_paragraph_start=True,
                        is_list_item=_is_list_item(sent),
                    )
                )
            else:
                # Single long sentence or unpunctuated block -> split into individual words
                words = sent.split()
                for w_idx, word in enumerate(words):
                    units.append(
                        _ChunkUnit(
                            text=word,
                            words=1,
                            paragraph_id=current_para_id,
                            is_heading=False,
                            is_paragraph_start=(w_idx == 0),
                            is_list_item=False,
                        )
                    )

        current_para_id += 1

    return units


def _join_chunk_units(units: list[_ChunkUnit]) -> str:
    """Format and join units into a cohesive chunk string."""
    if not units:
        return ""
    result = units[0].text
    for i in range(1, len(units)):
        prev = units[i - 1]
        curr = units[i]
        if curr.is_paragraph_start:
            if curr.is_list_item and prev.is_list_item:
                result += "\n" + curr.text
            else:
                result += "\n\n" + curr.text
        elif curr.is_list_item:
            result += "\n" + curr.text
        else:
            # Continuation of same paragraph / sentence
            result += " " + curr.text
    return result


def chunk_text_content(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    """Split a string of text into overlapping structure-aware chunks.

    Prefers paragraph boundaries. If a paragraph is too large, splits at sentence
    boundaries. If a sentence is too large, splits at word boundaries as a last resort.
    Preserves headings with subsequent content and avoids breaking lists unnecessarily.

    Args:
        text: Text to split.
        chunk_size: Target maximum size in words/tokens (default 500).
        chunk_overlap: Target overlap in words/tokens (default 100).

    Returns:
        List of non-empty string chunks.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    if not text or not text.strip():
        return []

    total_words = _count_words(text)
    if total_words == 0:
        return []

    if total_words <= chunk_size:
        return [text.strip()]

    units = _decompose_text(text, max_chunk_size=chunk_size)
    if not units:
        return []

    total_units = len(units)
    if total_units == 1:
        return [units[0].text]

    # Pre-calculate paragraph word totals for paragraph-boundary preference
    para_totals: dict[int, int] = {}
    for u in units:
        para_totals[u.paragraph_id] = para_totals.get(u.paragraph_id, 0) + u.words

    chunks: list[str] = []
    current_start = 0

    while current_start < total_units:
        accumulated: list[_ChunkUnit] = []
        curr_words = 0
        idx = current_start

        while idx < total_units:
            u = units[idx]

            # Paragraph boundary preference:
            # If u starts a new paragraph, and we already accumulated content in this chunk:
            if accumulated and u.is_paragraph_start and u.paragraph_id != units[idx - 1].paragraph_id:
                p_total = para_totals.get(u.paragraph_id, u.words)
                # Does the entire paragraph fit in the remaining budget?
                if curr_words + p_total > chunk_size:
                    # If this paragraph CAN fit into a fresh chunk alone, prefer splitting here!
                    if p_total <= chunk_size:
                        break

            # Check if this single unit fits
            if curr_words + u.words <= chunk_size or not accumulated:
                accumulated.append(u)
                curr_words += u.words
                idx += 1
            else:
                break

        # Heading guard: If the last unit in accumulated is a heading, and more units remain,
        # don't leave the heading stranded at the end of the chunk without its section content.
        if len(accumulated) > 1 and idx < total_units:
            if accumulated[-1].is_heading:
                popped = accumulated.pop()
                curr_words -= popped.words
                idx -= 1

        chunk_str = _join_chunk_units(accumulated).strip()
        if chunk_str:
            if not chunks or chunk_str != chunks[-1]:
                chunks.append(chunk_str)

        # End of document reached
        if idx >= total_units:
            break

        # Calculate overlap for the next chunk
        if chunk_overlap <= 0:
            next_start = idx
        else:
            overlap_words = 0
            overlap_count = 0
            for j in range(idx - 1, current_start, -1):
                cand = units[j]
                if overlap_words + cand.words <= chunk_overlap:
                    overlap_words += cand.words
                    overlap_count += 1
                elif overlap_count == 0:
                    # Allow 1 unit if it is reasonably close to chunk_overlap (<= 1.3x)
                    if cand.words <= int(chunk_overlap * 1.3):
                        overlap_words += cand.words
                        overlap_count += 1
                    break
                else:
                    break

            next_start = idx - overlap_count

        # Ensure strictly positive progress to guarantee termination
        if next_start <= current_start:
            next_start = current_start + 1

        current_start = next_start

    return chunks


def chunk_segment(
    segment: ExtractedSegment,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    start_chunk_index: int = 0,
) -> list[TextChunk]:
    """Split an individual ExtractedSegment into TextChunks.

    Preserves source metadata and page attribution without merging across segments.

    Args:
        segment: The extracted segment to chunk.
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap in words.
        start_chunk_index: Initial chunk index to assign.

    Returns:
        List of TextChunk instances.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    raw_chunks = chunk_text_content(
        segment.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    if not raw_chunks:
        return []

    page_label = f"p{segment.page}" if segment.page is not None else "pNone"
    result: list[TextChunk] = []

    for local_idx, chunk_str in enumerate(raw_chunks):
        global_idx = start_chunk_index + local_idx
        chunk_id = f"{segment.document_id}_{page_label}_c{local_idx}"
        result.append(
            TextChunk(
                chunk_id=chunk_id,
                document_id=segment.document_id,
                source=segment.source,
                page=segment.page,
                chunk_index=global_idx,
                text=chunk_str,
            )
        )

    return result


def chunk_document(
    segments: Sequence[ExtractedSegment],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[TextChunk]:
    """Split all segments of a document into TextChunks.

    Chunks each segment independently, ensuring that text from different PDF pages
    is never merged into the same chunk.

    Args:
        segments: Sequence of extracted segments for a document.
        chunk_size: Target chunk size in words.
        chunk_overlap: Overlap in words.

    Returns:
        List of all TextChunk instances generated for the document in order.
    """
    validate_chunk_params(chunk_size, chunk_overlap)

    all_chunks: list[TextChunk] = []
    current_index = 0

    for seg in segments:
        seg_chunks = chunk_segment(
            seg,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            start_chunk_index=current_index,
        )
        all_chunks.extend(seg_chunks)
        current_index += len(seg_chunks)

    return all_chunks
