from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any


DOCUMENTS_DIR = Path("data/raw/documents")
MANIFEST_PATH = Path("data/processed/document_manifest.json")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")

CHUNKER_VERSION = "2.0"

TARGET_WORDS = 180
MAX_WORDS = 260
MIN_CHUNK_WORDS = 20
SMALL_SECTION_WORDS = 35
OVERLAP_WORDS = 30


# ---------------------------------------------------------
# TEXT HELPERS
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = str(text)

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize excessive whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def hash_text(text: str) -> str:
    normalized = clean_text(text)

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


def build_chunk_id(
    document_id: str,
    chunk_hash: str,
) -> str:
    return f"{document_id}__{chunk_hash[:16]}"


def split_words(text: str) -> list[str]:
    text = clean_text(text)

    if not text:
        return []

    return text.split()


def section_text(section: dict[str, Any]) -> str:
    """
    Convert a structured section into clean chunkable text.
    """

    parts: list[str] = []

    heading = clean_text(
        section.get("heading", "")
    )

    if heading:
        parts.append(heading)

    paragraphs = section.get("paragraphs", [])

    if isinstance(paragraphs, list):
        for paragraph in paragraphs:
            paragraph = clean_text(str(paragraph))

            if paragraph:
                parts.append(paragraph)

    bullets = section.get("bullets", [])

    if isinstance(bullets, list):
        for bullet in bullets:
            bullet = clean_text(str(bullet))

            if bullet:
                parts.append(bullet)

    # Some documents may already contain `content`
    content = clean_text(
        section.get("content", "")
    )

    if content and content not in parts:
        parts.append(content)

    return clean_text("\n".join(parts))


def is_noise_text(text: str) -> bool:
    """
    Remove obvious navigation/UI-only chunks.

    This is intentionally conservative.
    """

    text = clean_text(text)

    if not text:
        return True

    lowered = text.lower()

    noise_patterns = [
        "skip to content",
        "menu",
        "close menu",
        "open menu",
        "cookie settings",
        "privacy policy",
        "terms of service",
    ]

    if lowered in noise_patterns:
        return True

    return False


# ---------------------------------------------------------
# SECTION MERGING
# ---------------------------------------------------------

def merge_small_sections(
    sections: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Combine very small sections with neighboring sections.

    Important:
    - Never silently discard trailing sections.
    - Preserve all meaningful content.
    """

    merged: list[dict[str, Any]] = []

    buffer: dict[str, Any] | None = None

    for section in sections:

        text = section_text(section)

        if not text:
            continue

        if is_noise_text(text):
            continue

        words = split_words(text)

        current = {
            "heading": clean_text(
                section.get("heading", "")
            ),
            "level": section.get("level", 0),
            "text": text,
            "word_count": len(words),
        }

        # Normal-sized section
        if len(words) >= SMALL_SECTION_WORDS:

            if buffer is not None:
                combined_text = clean_text(
                    buffer["text"] + "\n" + current["text"]
                )

                merged.append(
                    {
                        "heading": buffer["heading"],
                        "level": buffer["level"],
                        "text": combined_text,
                        "word_count": len(
                            split_words(combined_text)
                        ),
                    }
                )

                buffer = None

            else:
                merged.append(current)

            continue

        # Small section
        if buffer is None:
            buffer = current

        else:
            combined_text = clean_text(
                buffer["text"] + "\n" + current["text"]
            )

            buffer = {
                "heading": buffer["heading"],
                "level": buffer["level"],
                "text": combined_text,
                "word_count": len(
                    split_words(combined_text)
                ),
            }

            # Flush once reasonably sized
            if buffer["word_count"] >= SMALL_SECTION_WORDS:
                merged.append(buffer)
                buffer = None

    # -----------------------------------------------------
    # CRITICAL:
    # Flush remaining trailing buffer.
    # This was the bug that caused service pages to vanish.
    # -----------------------------------------------------

    if buffer is not None:
        if buffer["word_count"] >= MIN_CHUNK_WORDS:
            merged.append(buffer)

    return merged


# ---------------------------------------------------------
# CHUNK SPLITTING
# ---------------------------------------------------------

def split_large_text(
    text: str,
    target_words: int = TARGET_WORDS,
    max_words: int = MAX_WORDS,
    overlap_words: int = OVERLAP_WORDS,
) -> list[str]:

    words = split_words(text)

    if not words:
        return []

    if len(words) <= max_words:
        return [" ".join(words)]

    chunks: list[str] = []

    start = 0
    total = len(words)

    while start < total:

        end = min(
            start + max_words,
            total,
        )

        chunk_words = words[start:end]

        # Try to end closer to target size
        if (
            end < total
            and len(chunk_words) > target_words
        ):
            preferred_end = start + target_words

            if preferred_end > start:
                end = min(
                    preferred_end,
                    total,
                )

                chunk_words = words[start:end]

        chunk = " ".join(chunk_words)

        if chunk:
            chunks.append(chunk)

        if end >= total:
            break

        next_start = max(
            end - overlap_words,
            start + 1,
        )

        start = next_start

    return chunks


# ---------------------------------------------------------
# DOCUMENT PROCESSING
# ---------------------------------------------------------

def process_document(
    document: dict[str, Any],
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:

    document_id = manifest.get("document_id", "")

    if not document_id:
        raise ValueError(
            "Manifest entry is missing document_id"
        )

    canonical_url = (
        manifest.get("canonical_url")
        or document.get("canonical_url")
        or document.get("url")
        or ""
    )

    content_hash = (
        manifest.get("content_hash")
        or hash_text(
            document.get("content", "")
        )
    )

    title = clean_text(
        document.get("title", "")
    )

    url = (
        document.get("url")
        or canonical_url
        or ""
    )

    page_type = (
        document.get("page_type")
        or "general"
    )

    sections = document.get(
        "sections",
        [],
    )

    if not isinstance(sections, list):
        sections = []

    # Merge small sections first
    merged_sections = merge_small_sections(
        sections
    )

    chunks: list[dict[str, Any]] = []

    chunk_index = 0

    for section_index, section in enumerate(
        merged_sections
    ):

        section_content = clean_text(
            section.get("text", "")
        )

        if not section_content:
            continue

        section_heading = clean_text(
            section.get("heading", "")
        )

        section_level = section.get(
            "level",
            0,
        )

        section_words = split_words(
            section_content
        )

        if not section_words:
            continue

        section_chunks = split_large_text(
            section_content
        )

        for section_chunk_index, chunk_text in enumerate(
            section_chunks
        ):

            chunk_text = clean_text(
                chunk_text
            )

            if not chunk_text:
                continue

            words = split_words(
                chunk_text
            )

            word_count = len(words)

            # Skip truly tiny chunks
            if word_count < MIN_CHUNK_WORDS:
                continue

            chunk_hash = hash_text(
                chunk_text
            )

            chunk_id = build_chunk_id(
                document_id,
                chunk_hash,
            )

            chunk = {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "chunk_hash": chunk_hash,
                "content_hash": content_hash,
                "chunker_version": CHUNKER_VERSION,

                "url": url,
                "canonical_url": canonical_url,
                "title": title,
                "page_type": page_type,

                "section": section_heading,
                "section_index": section_index,
                "section_level": section_level,

                "chunk_index": chunk_index,
                "section_chunk_index": section_chunk_index,

                "text": chunk_text,
                "word_count": word_count,
            }

            chunks.append(chunk)

            chunk_index += 1

    return chunks


# ---------------------------------------------------------
# DOCUMENT LOOKUP
# ---------------------------------------------------------

def build_document_lookup() -> dict[str, Path]:
    """
    Build URL -> document file mapping.

    We do NOT assume:

        document_id.json

    because the actual crawler filenames are slug-based.
    """

    lookup: dict[str, Path] = {}

    for path in DOCUMENTS_DIR.glob("*.json"):

        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as f:
                document = json.load(f)

        except Exception as exc:
            print(
                f"WARNING: Could not read {path}: {exc}"
            )
            continue

        url = clean_text(
            document.get("url", "")
        )

        canonical_url = clean_text(
            document.get("canonical_url", "")
        )

        if url:
            lookup[url] = path

        if canonical_url:
            lookup[canonical_url] = path

    return lookup


def find_document_path(
    manifest_entry: dict[str, Any],
    lookup: dict[str, Path],
) -> Path | None:

    url_candidates = [
        manifest_entry.get("canonical_url", ""),
        manifest_entry.get("url", ""),
    ]

    for url in url_candidates:

        url = clean_text(url)

        if not url:
            continue

        path = lookup.get(url)

        if path and path.exists():
            return path

    return None


# ---------------------------------------------------------
# ATOMIC JSONL WRITE
# ---------------------------------------------------------

def write_jsonl_atomic(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with temp_path.open(
        "w",
        encoding="utf-8",
    ) as f:

        for record in records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )

    temp_path.replace(path)


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main() -> None:

    print("=" * 60)
    print("SWARNASKY CHUNKING PIPELINE")
    print("=" * 60)

    if not DOCUMENTS_DIR.exists():
        raise FileNotFoundError(
            f"Documents directory not found: {DOCUMENTS_DIR}"
        )

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    # -----------------------------------------------------
    # Load manifest
    # -----------------------------------------------------

    with MANIFEST_PATH.open(
        "r",
        encoding="utf-8",
    ) as f:
        manifest_data = json.load(f)

    # Manifest can be either:
    # {
    #   "documents": [...]
    # }
    #
    # or directly [...]
    if isinstance(
        manifest_data,
        dict,
    ):
        manifest_entries = (
            manifest_data.get(
                "documents",
                []
            )
        )
    elif isinstance(
        manifest_data,
        list,
    ):
        manifest_entries = manifest_data
    else:
        raise ValueError(
            "Unsupported manifest format"
        )

    if not manifest_entries:
        raise RuntimeError(
            "No documents found in manifest."
        )

    # -----------------------------------------------------
    # Build actual file lookup
    # -----------------------------------------------------

    document_lookup = (
        build_document_lookup()
    )

    print(
        f"Manifest documents : {len(manifest_entries)}"
    )

    print(
        f"Document files     : {len(document_lookup)}"
    )

    # -----------------------------------------------------
    # Process documents
    # -----------------------------------------------------

    all_chunks: list[dict[str, Any]] = []

    documents_processed = 0
    failed_documents = 0
    missing_documents = 0

    for manifest_entry in manifest_entries:

        document_id = manifest_entry.get(
            "document_id",
            "",
        )

        url = (
            manifest_entry.get(
                "canonical_url"
            )
            or manifest_entry.get(
                "url"
            )
            or ""
        )

        document_path = find_document_path(
            manifest_entry,
            document_lookup,
        )

        if document_path is None:

            missing_documents += 1

            print(
                f"Missing document: "
                f"{document_id} | {url}"
            )

            continue

        try:

            with document_path.open(
                "r",
                encoding="utf-8",
            ) as f:
                document = json.load(f)

            chunks = process_document(
                document,
                manifest_entry,
            )

            all_chunks.extend(
                chunks
            )

            documents_processed += 1

            print(
                f"Processed: "
                f"{document_path.name} "
                f"-> {len(chunks)} chunks"
            )

        except Exception as exc:

            failed_documents += 1

            print(
                f"FAILED: "
                f"{document_path.name}: {exc}"
            )

    # -----------------------------------------------------
    # Global deduplication
    # -----------------------------------------------------

    unique_chunks: list[dict[str, Any]] = []

    seen_hashes: set[str] = set()

    duplicate_chunks = 0

    for chunk in all_chunks:

        chunk_hash = chunk["chunk_hash"]

        if chunk_hash in seen_hashes:

            duplicate_chunks += 1

            continue

        seen_hashes.add(
            chunk_hash
        )

        unique_chunks.append(
            chunk
        )

    # -----------------------------------------------------
    # Statistics
    # -----------------------------------------------------

    word_counts = [
        chunk["word_count"]
        for chunk in unique_chunks
    ]

    average_words = (
        sum(word_counts)
        / len(word_counts)
        if word_counts
        else 0.0
    )

    tiny_chunks = sum(
        1
        for count in word_counts
        if count < MIN_CHUNK_WORDS
    )

    oversized_chunks = sum(
        1
        for count in word_counts
        if count > MAX_WORDS
    )

    # -----------------------------------------------------
    # Write output
    # -----------------------------------------------------

    write_jsonl_atomic(
        OUTPUT_PATH,
        unique_chunks,
    )

    # -----------------------------------------------------
    # Final report
    # -----------------------------------------------------

    print()
    print("=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)

    print(
        f"Documents processed : {documents_processed}"
    )

    print(
        f"Missing documents   : {missing_documents}"
    )

    print(
        f"Failed documents    : {failed_documents}"
    )

    print(
        f"Raw chunks          : {len(all_chunks)}"
    )

    print(
        f"Unique chunks       : {len(unique_chunks)}"
    )

    print(
        f"Duplicate chunks    : {duplicate_chunks}"
    )

    print(
        f"Average words/chunk : {average_words:.1f}"
    )

    print(
        f"Tiny chunks         : {tiny_chunks}"
    )

    print(
        f"Oversized chunks    : {oversized_chunks}"
    )

    print(
        f"Output              : {OUTPUT_PATH}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()