from typing import List

from pydantic import BaseModel, Field


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
    schema_version: str = "2.0"

    url: str
    canonical_url: str = ""

    title: str = ""
    description: str = ""

    page_type: str = "general"

    headings: List[Heading] = Field(
        default_factory=list
    )

    sections: List[Section] = Field(
        default_factory=list
    )

    links: List[Link] = Field(
        default_factory=list
    )

    content: str = ""

    word_count: int = 0

    status_code: int = 200

    encoding: str = ""