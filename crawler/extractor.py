from __future__ import annotations

import re
from typing import List
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field


# ============================================================
# MODELS
# ============================================================

class Link(BaseModel):
    url: str
    text: str = ""


class Heading(BaseModel):
    level: int
    text: str


class Section(BaseModel):
    heading: str = ""
    level: int = 0
    paragraphs: List[str] = Field(default_factory=list)
    bullets: List[str] = Field(default_factory=list)
    content: str = ""
    word_count: int = 0


class CrawledDocument(BaseModel):
    schema_version: str = "2.2"
    url: str
    canonical_url: str = ""
    title: str = ""
    description: str = ""
    page_type: str = "general"
    headings: List[Heading] = Field(default_factory=list)
    sections: List[Section] = Field(default_factory=list)
    links: List[Link] = Field(default_factory=list)
    content: str = ""
    word_count: int = 0
    status_code: int = 200
    encoding: str = ""


# ============================================================
# CONSTANTS
# ============================================================

REMOVE_TAGS = {
    "script",
    "style",
    "noscript",
    "svg",
    "canvas",
    "iframe",
    "form",
    "template",
}

REMOVE_STRUCTURAL_TAGS = {
    "header",
    "footer",
    "nav",
}

NOISE_CLASS_PATTERNS = [
    r"\bnavbar\b",
    r"\bnav-bar\b",
    r"\bnavigation\b",
    r"\bsite-nav\b",
    r"\bmain-nav\b",
    r"\bmobile-nav\b",
    r"\bmenu\b",
    r"\bfooter\b",
    r"\bheader\b",
    r"\bcookie\b",
    r"\bcookies\b",
    r"\bpopup\b",
    r"\bmodal\b",
    r"\bbreadcrumb\b",
]

NOISE_HEADING_PATTERNS = [
    r"^next step$",
    r"^get started$",
    r"^contact us$",
    r"^let'?s talk$",
    r"^talk to us$",
    r"^have a related business challenge\??$",
    r"^other services you may need\.?$",
    r"^related services$",
]

NOISE_TEXT_PATTERNS = [
    r"^tell us what you'?re dealing with",
    r"^we'?ll tell you honestly whether",
]


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\xa0", " ")
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_sentence_spacing(text: str) -> str:
    text = clean_text(text)

    if not text:
        return ""

    text = re.sub(r"([.!?])([A-Za-z])", r"\1 \2", text)
    text = re.sub(r"([,;:])([A-Za-z])", r"\1 \2", text)

    return clean_text(text)


def clean_and_normalize(text: str) -> str:
    return normalize_sentence_spacing(text)


def element_text(element: Tag | None) -> str:
    if element is None:
        return ""

    parts = []

    for item in element.stripped_strings:
        value = clean_and_normalize(item)

        if value:
            parts.append(value)

    return clean_and_normalize(" ".join(parts))


# ============================================================
# URL HELPERS
# ============================================================

def normalize_url(url: str) -> str:
    if not url:
        return ""

    url = urldefrag(url)[0].strip()

    parsed = urlparse(url)

    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()

    path = parsed.path or "/"

    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return f"{scheme}://{netloc}{path}"


def get_canonical_url(
    soup: BeautifulSoup,
    fallback_url: str,
) -> str:

    canonical = soup.find(
        "link",
        rel=lambda value: (
            value
            and (
                "canonical" in value
                if isinstance(value, list)
                else "canonical" in str(value).lower()
            )
        ),
    )

    if canonical and canonical.get("href"):
        return normalize_url(
            urljoin(
                fallback_url,
                canonical["href"],
            )
        )

    return normalize_url(fallback_url)


# ============================================================
# PAGE TYPE
# ============================================================

def detect_page_type(
    url: str,
    title: str = "",
) -> str:

    path = urlparse(url).path.lower().strip("/")

    if not path:
        return "home"

    if path == "about":
        return "about"

    if path in {"career", "careers"}:
        return "career"

    if path == "case-studies":
        return "case-studies"

    if path == "contact":
        return "contact"

    if path == "services":
        return "services"

    if path.startswith("services/"):
        return "service"

    return "general"


# ============================================================
# HTML CLEANUP
# ============================================================

def remove_unwanted_elements(
    soup: BeautifulSoup,
) -> None:

    for tag_name in REMOVE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag_name in REMOVE_STRUCTURAL_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    for tag in list(soup.find_all(True)):

        classes = tag.get("class", [])

        if not classes:
            continue

        class_text = " ".join(
            str(x).lower()
            for x in classes
        )

        if any(
            re.search(pattern, class_text)
            for pattern in NOISE_CLASS_PATTERNS
        ):
            if tag.name in {"main", "article"}:
                continue

            tag.decompose()


# ============================================================
# DUPLICATE CLEANUP
# ============================================================

def remove_repeated_lines(
    lines: List[str],
) -> List[str]:

    result = []
    previous = None

    for line in lines:

        value = clean_and_normalize(line)

        if not value:
            continue

        if value == previous:
            continue

        result.append(value)
        previous = value

    return result


# ============================================================
# HEADING EXTRACTION
# ============================================================

def extract_headings(
    soup: BeautifulSoup,
) -> List[Heading]:

    headings = []

    for tag in soup.find_all(
        ["h1", "h2", "h3", "h4", "h5", "h6"]
    ):

        text = element_text(tag)

        if not text:
            continue

        level = int(tag.name[1])

        headings.append(
            Heading(
                level=level,
                text=text,
            )
        )

    return headings


# ============================================================
# ELEMENT FILTERING
# ============================================================

def is_inside_noise_container(
    element: Tag,
    root: Tag,
) -> bool:

    parent = element.parent

    while isinstance(parent, Tag) and parent != root:

        classes = parent.get("class", [])

        if classes:

            class_text = " ".join(
                str(x).lower()
                for x in classes
            )

            if any(
                re.search(
                    pattern,
                    class_text,
                )
                for pattern in NOISE_CLASS_PATTERNS
            ):
                return True

        parent = parent.parent

    return False


def is_text_only_element(
    element: Tag,
) -> bool:

    """
    True when the element is essentially a text label.

    Example:

        <div class="...">Founder &amp; CEO</div>

    This lets us capture important metadata that is not
    represented by p/li/heading elements.
    """

    if element.find(
        [
            "div",
            "p",
            "ul",
            "ol",
            "li",
            "section",
            "article",
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        ]
    ):
        return False

    text = element_text(element)

    if not text:
        return False

    return True


def is_meaningful_inline_label(
    element: Tag,
) -> bool:

    """
    Capture short text labels such as:

        Founder & CEO
        Leadership
        Product
        Engineering

    while avoiding large layout containers.
    """

    text = element_text(element)

    if not text:
        return False

    word_count = len(text.split())

    if word_count > 12:
        return False

    if len(text) > 120:
        return False

    if element.name not in {"div", "span"}:
        return False

    return is_text_only_element(element)


# ============================================================
# SECTION EXTRACTION
# ============================================================

def is_noise_heading(
    text: str,
) -> bool:

    value = clean_and_normalize(text).lower()

    if not value:
        return True

    return any(
        re.search(
            pattern,
            value,
            re.IGNORECASE,
        )
        for pattern in NOISE_HEADING_PATTERNS
    )


def is_noise_text(
    text: str,
) -> bool:

    value = clean_and_normalize(text).lower()

    if not value:
        return False

    return any(
        re.search(
            pattern,
            value,
            re.IGNORECASE,
        )
        for pattern in NOISE_TEXT_PATTERNS
    )


def extract_sections(
    soup: BeautifulSoup,
) -> List[Section]:

    root = soup.find("main")

    if root is None:
        root = soup.find("article")

    if root is None:
        root = soup.body

    if root is None:
        return []

    sections: List[Section] = []

    current: Section | None = None

    elements = root.find_all(
        [
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
            "p",
            "li",
            "div",
            "span",
        ],
        recursive=True,
    )

    for element in elements:

        # ----------------------------------------------------
        # SKIP UNWANTED CONTENT
        # ----------------------------------------------------

        if element.find_parent(
            list(
                REMOVE_TAGS
                | REMOVE_STRUCTURAL_TAGS
            )
        ):
            continue

        if is_inside_noise_container(
            element,
            root,
        ):
            continue

        # ----------------------------------------------------
        # HEADING
        # ----------------------------------------------------

        if element.name in {
            "h1",
            "h2",
            "h3",
            "h4",
            "h5",
            "h6",
        }:

            heading = element_text(element)

            if not heading:
                continue

            if current is not None:

                current = finalize_section(
                    current
                )

                if current is not None:
                    sections.append(current)

            current = Section(
                heading=heading,
                level=int(element.name[1]),
            )

            continue

        # ----------------------------------------------------
        # PARAGRAPH
        # ----------------------------------------------------

        if element.name == "p":

            text = element_text(element)

            if not text:
                continue

            if is_noise_text(text):
                continue

            if current is None:
                current = Section(
                    heading="",
                    level=0,
                )

            if text not in current.paragraphs:
                current.paragraphs.append(text)

            continue

        # ----------------------------------------------------
        # LIST ITEM
        # ----------------------------------------------------

        if element.name == "li":

            text = element_text(element)

            if not text:
                continue

            if is_noise_text(text):
                continue

            if current is None:
                current = Section(
                    heading="",
                    level=0,
                )

            if text not in current.bullets:
                current.bullets.append(text)

            continue

        # ----------------------------------------------------
        # IMPORTANT INLINE LABELS
        # ----------------------------------------------------
        #
        # Example from Swarnasky:
        #
        # <h3>Anuradha</h3>
        # <div>Founder &amp; CEO</div>
        # <p>...</p>
        #
        # The old extractor ignored the div.
        # We now capture short text-only div/span labels.
        # ----------------------------------------------------

        if element.name in {"div", "span"}:

            if not is_meaningful_inline_label(
                element
            ):
                continue

            text = element_text(element)

            if not text:
                continue

            if is_noise_text(text):
                continue

            if current is None:
                current = Section(
                    heading="",
                    level=0,
                )

            if text not in current.paragraphs:
                current.paragraphs.append(text)

    # --------------------------------------------------------
    # FLUSH FINAL SECTION
    # --------------------------------------------------------

    if current is not None:

        current = finalize_section(
            current
        )

        if current is not None:
            sections.append(current)

    return sections


# ============================================================
# SECTION FINALIZATION
# ============================================================

def finalize_section(
    section: Section,
) -> Section | None:

    heading = clean_and_normalize(
        section.heading
    )

    paragraphs = remove_repeated_lines(
        section.paragraphs
    )

    bullets = remove_repeated_lines(
        section.bullets
    )

    if heading and is_noise_heading(
        heading
    ):
        return None

    parts = []

    if heading:
        parts.append(heading)

    parts.extend(paragraphs)

    for bullet in bullets:
        parts.append(bullet)

    parts = remove_repeated_lines(parts)

    content = "\n".join(parts)

    content = clean_and_normalize(
        content
    )

    if not content:
        return None

    section.heading = heading
    section.paragraphs = paragraphs
    section.bullets = bullets
    section.content = content
    section.word_count = len(
        content.split()
    )

    return section


# ============================================================
# LINKS
# ============================================================

def extract_links(
    soup: BeautifulSoup,
    base_url: str,
) -> List[Link]:

    links = []
    seen = set()

    for anchor in soup.find_all(
        "a",
        href=True,
    ):

        href = anchor.get(
            "href",
            "",
        ).strip()

        if not href:
            continue

        if href.startswith(
            (
                "#",
                "javascript:",
                "mailto:",
                "tel:",
            )
        ):
            continue

        absolute = urljoin(
            base_url,
            href,
        )

        absolute = normalize_url(
            absolute
        )

        if not absolute:
            continue

        if absolute in seen:
            continue

        seen.add(absolute)

        text = element_text(anchor)

        links.append(
            Link(
                url=absolute,
                text=text,
            )
        )

    return links


# ============================================================
# DESCRIPTION
# ============================================================

def extract_description(
    soup: BeautifulSoup,
) -> str:

    meta = soup.find(
        "meta",
        attrs={
            "name": "description"
        },
    )

    if meta and meta.get("content"):

        return clean_and_normalize(
            meta["content"]
        )

    return ""


# ============================================================
# MAIN EXTRACTOR
# ============================================================

def extract_page(
    html: bytes,
    url: str,
    status_code: int = 200,
    encoding: str = "",
) -> CrawledDocument:

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # --------------------------------------------------------
    # BASIC METADATA
    # --------------------------------------------------------

    title = ""

    if soup.title:
        title = element_text(
            soup.title
        )

    description = extract_description(
        soup
    )

    canonical_url = get_canonical_url(
        soup,
        url,
    )

    page_type = detect_page_type(
        canonical_url,
        title,
    )

    # --------------------------------------------------------
    # LINKS
    # --------------------------------------------------------

    links = extract_links(
        soup,
        url,
    )

    # --------------------------------------------------------
    # CLEAN HTML
    # --------------------------------------------------------

    remove_unwanted_elements(
        soup
    )

    # --------------------------------------------------------
    # HEADINGS
    # --------------------------------------------------------

    headings = extract_headings(
        soup
    )

    # --------------------------------------------------------
    # SECTIONS
    # --------------------------------------------------------

    sections = extract_sections(
        soup
    )

    # --------------------------------------------------------
    # FULL CONTENT
    # --------------------------------------------------------

    content_parts = []

    for section in sections:

        if section.content:

            content_parts.append(
                section.content
            )

    content = "\n\n".join(
        content_parts
    )

    content = clean_and_normalize(
        content
    )

    word_count = (
        len(content.split())
        if content
        else 0
    )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    if word_count == 0:

        root = (
            soup.find("main")
            or soup.find("article")
            or soup.body
        )

        if root:

            fallback_text = element_text(
                root
            )

            fallback_text = (
                clean_and_normalize(
                    fallback_text
                )
            )

            if fallback_text:

                content = fallback_text

                word_count = len(
                    content.split()
                )

                if not sections:

                    sections = [
                        Section(
                            heading="",
                            level=0,
                            paragraphs=[
                                content
                            ],
                            content=content,
                            word_count=word_count,
                        )
                    ]

    # --------------------------------------------------------
    # RETURN
    # --------------------------------------------------------

    return CrawledDocument(
        schema_version="2.2",
        url=normalize_url(url),
        canonical_url=canonical_url,
        title=title,
        description=description,
        page_type=page_type,
        headings=headings,
        sections=sections,
        links=links,
        content=content,
        word_count=word_count,
        status_code=status_code,
        encoding=encoding or "",
    )