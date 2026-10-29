"""Data types shared across ingest, retrieval, the agent and the API."""

from typing import Any, Literal

from pydantic import BaseModel, Field

DISCLAIMER = "Demo project. Not medical advice. Not affiliated with or endorsed by AstraZeneca."


class Chunk(BaseModel):
    chunk_id: str
    drug: str
    brand: str
    generic: str
    section: str
    section_title: str
    subsection: str | None = None
    position: int
    text: str
    set_id: str
    label_version: str
    effective_time: str
    source_url: str

    @property
    def header(self) -> str:
        parts = [f"{self.brand} ({self.generic})", self.section_title]
        if self.subsection:
            parts.append(self.subsection)
        return " | ".join(parts)

    @property
    def index_text(self) -> str:
        """Text used for search: a short context header plus the chunk body."""
        return f"{self.header}\n{self.text}"


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    bm25_rank: int | None = None
    dense_rank: int | None = None


class Citation(BaseModel):
    chunk_id: str
    drug: str
    section_title: str
    subsection: str | None
    label_version: str
    effective_time: str
    source_url: str
    quote: str


class AnswerSentence(BaseModel):
    text: str
    citations: list[str] = Field(default_factory=list)


JudgeLabel = Literal["supported", "partial", "unsupported", "not_checked"]


class SentenceCheck(BaseModel):
    text: str
    citations: list[str]
    unknown_citations: list[str]
    token_support: float
    unsupported_numbers: list[str]
    deterministic_passed: bool
    judge_label: JudgeLabel = "not_checked"
    judge_reason: str | None = None


class GroundingReport(BaseModel):
    """`passed` covers the deterministic checks only. Judge labels are reported beside them
    because they come from a model and are a heuristic, not proof."""

    passed: bool
    sentences: list[SentenceCheck]
    mean_token_support: float
    problems: list[str]
    judge_model: str | None = None


class TraceStep(BaseModel):
    step: int
    kind: Literal["tool_call", "final_answer", "validation", "error", "limit"]
    name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    summary: str
    duration_ms: float = 0.0
    flags: list[str] = Field(default_factory=list)


class AgentAnswer(BaseModel):
    question: str
    answer: str
    sentences: list[AnswerSentence]
    citations: list[Citation]
    abstained: bool
    abstain_reason: str | None = None
    grounding: GroundingReport | None = None
    trace: list[TraceStep]
    provider: str
    latency_ms: float
    disclaimer: str = DISCLAIMER
