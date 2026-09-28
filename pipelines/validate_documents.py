from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import ValidationError

from crawler.models import CrawledDocument


DOCUMENT_DIR = Path("data/raw/documents")
PROCESSED_DIR = Path("data/processed")

MANIFEST_FILE = PROCESSED_DIR / "document_manifest.json"
REPORT_FILE = PROCESSED_DIR / "validation_report.json"

SCHEMA_VERSION = "2.0"
HASH_VERSION = "v1"


def normalize_url(url: str) -> str:
    """
    Normalize a URL so equivalent URLs produce the same document identity.
    """
    parsed = urlsplit(url.strip())

    scheme = parsed.scheme.lower()
    hostname = (parsed.hostname or "").lower()

    if hostname == "swarnasky.com":
        hostname = "www.swarnasky.com"

    port = parsed.port

    if port and not (
        (scheme == "https" and port == 443)
        or (scheme == "http" and port == 80)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parsed.path or "/"

    if path != "/":
        path = path.rstrip("/")

    # Query parameters are deliberately retained.
    # This avoids accidentally treating meaningful URLs as identical.
    query = parsed.query

    return urlunsplit(
        (
            scheme,
            netloc,
            path,
            query,
            "",
        )
    )


def stable_document_id(canonical_url: str) -> str:
    """
    Stable identity derived only from canonical URL.

    The same URL always gets the same document_id.
    """
    digest = hashlib.sha256(
        canonical_url.encode("utf-8")
    ).hexdigest()

    return f"doc_{digest[:24]}"


def normalize_text(text: str) -> str:
    """
    Normalize whitespace for deterministic content hashing.

    We intentionally do not aggressively rewrite punctuation or wording.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        line = line.strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def build_content_fingerprint(document: CrawledDocument) -> str:
    """
    Hash the meaningful extracted content.

    Changes to navigation links alone should not force re-embedding.
    """
    sections = []

    for section in document.sections:
        sections.append(
            {
                "heading": normalize_text(section.heading),
                "level": section.level,
                "paragraphs": [
                    normalize_text(value)
                    for value in section.paragraphs
                ],
                "bullets": [
                    normalize_text(value)
                    for value in section.bullets
                ],
                "content": normalize_text(section.content),
            }
        )

    fingerprint_payload = {
        "title": normalize_text(document.title),
        "description": normalize_text(document.description),
        "sections": sections,
    }

    serialized = json.dumps(
        fingerprint_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        serialized.encode("utf-8")
    ).hexdigest()


def validate_document(document: CrawledDocument) -> list[str]:
    """
    Business-level validation after Pydantic schema validation.
    """
    errors: list[str] = []

    if not document.url:
        errors.append("missing url")

    if not document.title:
        errors.append("missing title")

    if not document.sections:
        errors.append("document has no sections")

    if document.word_count <= 0:
        errors.append("document has zero words")

    calculated_words = len(
        re.findall(
            r"\b[\w'-]+\b",
            document.content,
            flags=re.UNICODE,
        )
    )

    # Small tolerance because extractor word counting and this validator
    # may use slightly different tokenization.
    if document.word_count > 0:
        difference = abs(
            calculated_words - document.word_count
        )

        tolerance = max(
            10,
            int(document.word_count * 0.15),
        )

        if difference > tolerance:
            errors.append(
                f"word_count mismatch: "
                f"stored={document.word_count}, "
                f"calculated={calculated_words}"
            )

    return errors


def process_document(path: Path) -> dict:
    """
    Load, validate and fingerprint one document.
    """
    raw = json.loads(
        path.read_text(encoding="utf-8")
    )

    try:
        document = CrawledDocument.model_validate(raw)
    except ValidationError as exc:
        return {
            "file": path.name,
            "status": "invalid",
            "errors": [
                error["msg"]
                for error in exc.errors()
            ],
        }

    errors = validate_document(document)

    canonical_url = normalize_url(
        document.canonical_url or document.url
    )

    document_id = stable_document_id(
        canonical_url
    )

    content_hash = build_content_fingerprint(
        document
    )

    return {
        "file": path.name,
        "status": "valid" if not errors else "warning",
        "errors": errors,
        "document_id": document_id,
        "canonical_url": canonical_url,
        "url": document.url,
        "title": document.title,
        "page_type": document.page_type,
        "schema_version": document.schema_version,
        "hash_version": HASH_VERSION,
        "content_hash": content_hash,
        "word_count": document.word_count,
        "section_count": len(document.sections),
        "link_count": len(document.links),
    }


def main() -> None:
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    files = sorted(
        DOCUMENT_DIR.glob("*.json")
    )

    if not files:
        print(
            f"No documents found in {DOCUMENT_DIR}"
        )
        return

    results = []

    for path in files:
        print(f"Validating: {path.name}")

        try:
            result = process_document(path)
        except Exception as exc:
            result = {
                "file": path.name,
                "status": "invalid",
                "errors": [
                    f"{type(exc).__name__}: {exc}"
                ],
            }

        results.append(result)

    valid = [
        item
        for item in results
        if item["status"] in {"valid", "warning"}
    ]

    invalid = [
        item
        for item in results
        if item["status"] == "invalid"
    ]

    warnings = [
        item
        for item in results
        if item["status"] == "warning"
    ]

    # Detect duplicate canonical URLs.
    url_groups: dict[str, list[str]] = {}

    for item in valid:
        url = item["canonical_url"]

        url_groups.setdefault(
            url,
            [],
        ).append(item["file"])

    duplicate_urls = {
        url: files
        for url, files in url_groups.items()
        if len(files) > 1
    }

    # Detect identical extracted content.
    hash_groups: dict[str, list[str]] = {}

    for item in valid:
        content_hash = item["content_hash"]

        hash_groups.setdefault(
            content_hash,
            [],
        ).append(item["file"])

    duplicate_content = {
        content_hash: files
        for content_hash, files in hash_groups.items()
        if len(files) > 1
    }

    manifest = {
        "manifest_version": "1.0",
        "schema_version": SCHEMA_VERSION,
        "hash_version": HASH_VERSION,
        "document_count": len(valid),
        "invalid_count": len(invalid),
        "warning_count": len(warnings),
        "duplicate_url_count": len(duplicate_urls),
        "duplicate_content_count": len(
            duplicate_content
        ),
        "documents": valid,
    }

    report = {
        "manifest_version": "1.0",
        "documents_found": len(files),
        "valid_documents": len(valid),
        "invalid_documents": len(invalid),
        "warning_documents": len(warnings),
        "duplicate_urls": duplicate_urls,
        "duplicate_content": duplicate_content,
        "invalid": invalid,
        "warnings": warnings,
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    REPORT_FILE.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("DOCUMENT VALIDATION COMPLETED")
    print("=" * 70)
    print(f"Documents found     : {len(files)}")
    print(f"Valid documents     : {len(valid)}")
    print(f"Warnings            : {len(warnings)}")
    print(f"Invalid documents   : {len(invalid)}")
    print(f"Duplicate URLs      : {len(duplicate_urls)}")
    print(f"Duplicate content   : {len(duplicate_content)}")
    print(f"Manifest            : {MANIFEST_FILE}")
    print(f"Report              : {REPORT_FILE}")
    print("=" * 70)

    if invalid:
        print("\nINVALID DOCUMENTS:")
        for item in invalid:
            print(f"  {item['file']}")
            for error in item["errors"]:
                print(f"    - {error}")

    if warnings:
        print("\nWARNINGS:")
        for item in warnings:
            print(f"  {item['file']}")
            for warning in item["errors"]:
                print(f"    - {warning}")


if __name__ == "__main__":
    main()