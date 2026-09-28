import hashlib
import json
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag

import requests

from crawler.extractor import extract_page


# ============================================================
# CONFIGURATION
# ============================================================

START_URL = "https://www.swarnasky.com/"

CANONICAL_HOST = "www.swarnasky.com"

RAW_DIR = Path("data/raw")
HTML_DIR = RAW_DIR / "html"
DOCUMENT_DIR = RAW_DIR / "documents"

METADATA_FILE = RAW_DIR / "crawl_metadata.json"

REQUEST_TIMEOUT = 20
CRAWL_DELAY = 0.3
MAX_PAGES = 1000


# ============================================================
# DIRECTORIES
# ============================================================

RAW_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

HTML_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

DOCUMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# HTTP SESSION
# ============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 "
        "(X11; Linux x86_64) "
        "AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/154.0.0.0 Safari/537.36"
    ),
    "Accept": (
        "text/html,"
        "application/xhtml+xml,"
        "application/xml;q=0.9,"
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# ============================================================
# URL NORMALIZATION
# ============================================================

def normalize_url(url: str) -> str:
    """
    Normalize URLs so the crawler has one canonical representation.
    """

    if not url:
        return ""

    url = urldefrag(url)[0]

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        return ""

    hostname = parsed.hostname

    if not hostname:
        return ""

    hostname = hostname.lower()

    # Convert bare domain to canonical host.
    if hostname == "swarnasky.com":
        hostname = CANONICAL_HOST

    # Reject external domains.
    if hostname != CANONICAL_HOST:
        return ""

    path = parsed.path or "/"

    # Remove duplicate slashes.
    while "//" in path:
        path = path.replace("//", "/")

    if not path:
        path = "/"

    # Remove trailing slash except root.
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    # Preserve query string because some pages may use it.
    query = parsed.query

    normalized = (
        f"https://{CANONICAL_HOST}"
        f"{path}"
    )

    if query:
        normalized += f"?{query}"

    return normalized


# ============================================================
# URL VALIDATION
# ============================================================

def is_valid_url(url: str) -> bool:
    """
    Check whether a URL belongs to the website and represents
    a crawlable HTML-like resource.
    """

    if not url:
        return False

    parsed = urlparse(url)

    if parsed.scheme != "https":
        return False

    hostname = parsed.hostname

    if not hostname:
        return False

    hostname = hostname.lower()

    if hostname == "swarnasky.com":
        hostname = CANONICAL_HOST

    if hostname != CANONICAL_HOST:
        return False

    path = parsed.path.lower()

    blocked_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".webp",
        ".svg",
        ".ico",
        ".css",
        ".js",
        ".map",
        ".mp4",
        ".mp3",
        ".wav",
        ".avi",
        ".mov",
        ".zip",
        ".rar",
        ".7z",
        ".tar",
        ".gz",
        ".exe",
        ".dmg",
        ".woff",
        ".woff2",
        ".ttf",
        ".eot",
        ".pdf",
        ".doc",
        ".docx",
        ".xls",
        ".xlsx",
        ".ppt",
        ".pptx",
    )

    if path.endswith(blocked_extensions):
        return False

    return True


# ============================================================
# FILE IDENTIFIER
# ============================================================

def make_file_id(url: str) -> str:
    """
    Generate a stable filesystem-safe identifier from a URL.
    """

    parsed = urlparse(url)

    path = parsed.path.strip("/")

    if not path:
        path = "home"

    safe_path = path.replace("/", "__")

    # Remove characters that are awkward in filenames.
    safe_path = "".join(
        char if (
            char.isalnum()
            or char in {
                "_",
                "-",
                ".",
            }
        ) else "_"
        for char in safe_path
    )

    safe_path = safe_path[:100]

    url_hash = hashlib.sha256(
        url.encode("utf-8")
    ).hexdigest()[:16]

    return f"{safe_path}__{url_hash}"


# ============================================================
# SAVE RAW HTML
# ============================================================

def save_raw_html(
    url: str,
    html_bytes: bytes,
) -> Path:
    """
    Save the exact response bytes.
    """

    file_id = make_file_id(url)

    output_path = HTML_DIR / f"{file_id}.html"

    output_path.write_bytes(html_bytes)

    return output_path


# ============================================================
# SAVE STRUCTURED DOCUMENT
# ============================================================

def save_document(document) -> Path:
    """
    Save Pydantic CrawledDocument as JSON.
    """

    file_id = make_file_id(document.url)

    output_path = DOCUMENT_DIR / f"{file_id}.json"

    # Pydantic v2
    if hasattr(document, "model_dump"):
        data = document.model_dump()

    # Pydantic v1 fallback
    else:
        data = document.dict()

    output_path.write_text(
        json.dumps(
            data,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return output_path


# ============================================================
# FETCH PAGE
# ============================================================

def fetch_page(
    session: requests.Session,
    url: str,
):
    """
    Fetch a webpage and return the response.
    """

    response = session.get(
        url,
        timeout=REQUEST_TIMEOUT,
        allow_redirects=True,
    )

    response.raise_for_status()

    content_type = (
        response.headers
        .get("Content-Type", "")
        .lower()
    )

    # Accept normal HTML and XHTML.
    if (
        "text/html" not in content_type
        and "application/xhtml+xml" not in content_type
    ):
        raise ValueError(
            f"Unsupported content type: {content_type}"
        )

    return response


# ============================================================
# SAVE CRAWL METADATA
# ============================================================

def save_crawl_metadata(
    *,
    visited_urls,
    saved_urls,
    failed_urls,
    started_at,
    finished_at,
):
    """
    Save crawl statistics for reproducibility/debugging.
    """

    metadata = {
        "start_url": START_URL,
        "canonical_host": CANONICAL_HOST,
        "started_at": started_at,
        "finished_at": finished_at,
        "pages_visited": len(visited_urls),
        "pages_saved": len(saved_urls),
        "pages_failed": len(failed_urls),
        "visited_urls": sorted(visited_urls),
        "saved_urls": sorted(saved_urls),
        "failed_urls": failed_urls,
    }

    METADATA_FILE.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


# ============================================================
# CRAWLER
# ============================================================

def crawl():
    """
    Crawl the Swarnasky website and save:
        1. Raw HTML
        2. Structured CrawledDocument JSON
    """

    started_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )

    queue = deque()

    start_url = normalize_url(
        START_URL
    )

    queue.append(start_url)

    visited = set()
    saved_urls = set()
    failed_urls = []

    session = requests.Session()

    session.headers.update(
        HEADERS
    )

    print("=" * 70)
    print("SWARNSKY WEBSITE CRAWLER")
    print("=" * 70)
    print(f"Starting URL : {start_url}")
    print(f"Domain       : {CANONICAL_HOST}")
    print(f"HTML output  : {HTML_DIR}")
    print(f"Document     : {DOCUMENT_DIR}")
    print("=" * 70)
    print()

    while queue and len(visited) < MAX_PAGES:

        current_url = queue.popleft()

        current_url = normalize_url(
            current_url
        )

        if not current_url:
            continue

        if current_url in visited:
            continue

        visited.add(current_url)

        page_number = len(visited)

        print(
            f"[{page_number:04d}] Crawling: "
            f"{current_url}"
        )

        try:
            # ------------------------------------------------
            # FETCH
            # ------------------------------------------------

            response = fetch_page(
                session,
                current_url,
            )

            # ------------------------------------------------
            # FINAL URL AFTER REDIRECT
            # ------------------------------------------------

            final_url = normalize_url(
                response.url
            )

            if not final_url:
                raise ValueError(
                    "Redirected to unsupported URL"
                )

            # ------------------------------------------------
            # RAW HTML
            # ------------------------------------------------

            html_bytes = response.content

            raw_file = save_raw_html(
                final_url,
                html_bytes,
            )

            # ------------------------------------------------
            # EXTRACT STRUCTURED DOCUMENT
            # ------------------------------------------------
            #
            # IMPORTANT:
            # extract_page expects:
            #
            #   html
            #   url
            #   status_code
            #
            # It returns CrawledDocument.
            # ------------------------------------------------

            document = extract_page(
                html_bytes,
                final_url,
                response.status_code,
                response.encoding or "",
            )

            # ------------------------------------------------
            # MINIMUM CONTENT CHECK
            # ------------------------------------------------

            if document.word_count < 20:

                print(
                    "      SKIPPED: "
                    f"very little content "
                    f"({document.word_count} words)"
                )

                continue

            # ------------------------------------------------
            # SAVE STRUCTURED DOCUMENT
            # ------------------------------------------------

            document_file = save_document(
                document
            )

            saved_urls.add(
                final_url
            )

            print(
                "      OK"
                f" | {document.word_count} words"
                f" | {len(document.sections)} sections"
                f" | {len(document.links)} links"
            )

            print(
                f"      HTML     : {raw_file}"
            )

            print(
                f"      Document : {document_file}"
            )

            # ------------------------------------------------
            # DISCOVER LINKS
            # ------------------------------------------------

            for link in document.links:

                # Pydantic object.
                if hasattr(link, "url"):
                    link_url = link.url

                # Defensive fallback.
                elif isinstance(link, dict):
                    link_url = link.get(
                        "url",
                        "",
                    )

                else:
                    continue

                if not link_url:
                    continue

                absolute_url = urljoin(
                    final_url,
                    link_url,
                )

                normalized_link = normalize_url(
                    absolute_url
                )

                if not normalized_link:
                    continue

                if not is_valid_url(
                    normalized_link
                ):
                    continue

                if normalized_link in visited:
                    continue

                if normalized_link not in queue:
                    queue.append(
                        normalized_link
                    )

        except requests.RequestException as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            failed_urls.append(
                {
                    "url": current_url,
                    "error": error,
                }
            )

            print(
                f"      REQUEST ERROR: {error}"
            )

        except Exception as exc:

            error = (
                f"{type(exc).__name__}: {exc}"
            )

            failed_urls.append(
                {
                    "url": current_url,
                    "error": error,
                }
            )

            print(
                f"      EXTRACTION ERROR: {error}"
            )

        finally:

            if CRAWL_DELAY > 0:
                time.sleep(
                    CRAWL_DELAY
                )

    # ========================================================
    # FINISH
    # ========================================================

    finished_at = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )

    save_crawl_metadata(
        visited_urls=visited,
        saved_urls=saved_urls,
        failed_urls=failed_urls,
        started_at=started_at,
        finished_at=finished_at,
    )

    print()
    print("=" * 70)
    print("CRAWLING COMPLETED")
    print("=" * 70)
    print(
        f"URLs visited : {len(visited)}"
    )
    print(
        f"Pages saved  : {len(saved_urls)}"
    )
    print(
        f"Failed URLs  : {len(failed_urls)}"
    )
    print(
        f"HTML output  : {HTML_DIR}"
    )
    print(
        f"Documents    : {DOCUMENT_DIR}"
    )
    print(
        f"Metadata     : {METADATA_FILE}"
    )
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    crawl()