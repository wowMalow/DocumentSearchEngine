from typing import List, NamedTuple

SCROLL_LIMIT = 100
SIMILARITY_THRESHOLD = 0.95


class InfoObject(NamedTuple):
    id: int
    question: str
    answer: str


class InfoDocumentObject(NamedTuple):
    id: int
    content: str


class LemmaInfoObject(NamedTuple):
    id: int
    content: str
    lemmas: List[str]


class VectorInfoObject(NamedTuple):
    id: int
    content: str
    vector: List[float]
