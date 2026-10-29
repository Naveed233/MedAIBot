"""Deterministic rules in place of an LLM. Used by tests, CI and when no API key is set.

It follows the same tool protocol as a real model, so the agent loop, tools and checks
are exercised end to end. Answers are extractive: the best-matching label sentence,
quoted with its chunk id. It does not understand the question; it matches words.
"""

import json
from itertools import count
from typing import Any

from medaibot.drugs import DrugCatalog
from medaibot.llm.base import LLMResponse, Message, ToolCall, ToolDef
from medaibot.query import normalise_query
from medaibot.text import split_sentences, tokenize

MIN_MATCH = 0.5
FRESHNESS_WORDS = ("latest", "up to date", "up-to-date", "current version", "newest", "最新")


class OfflineProvider:
    name = "offline-rules"

    def __init__(self, catalog: DrugCatalog) -> None:
        self.catalog = catalog
        self._ids = count(1)

    def _call(self, name: str, **arguments: Any) -> LLMResponse:
        return LLMResponse(
            text="",
            tool_calls=[ToolCall(id=f"offline-{next(self._ids)}", name=name, arguments=arguments)],
            model=self.name,
        )

    def _abstain(self, reason: str) -> LLMResponse:
        return self._call("final_answer", abstain=True, abstain_reason=reason, sentences=[])

    def complete(
        self, system: str, messages: list[Message], tools: list[ToolDef], require_tool: bool = True
    ) -> LLMResponse:
        if any(t.name == "record_judgements" for t in tools):
            return self._call("record_judgements", judgements=[])

        question = messages[0].content
        nq = normalise_query(question, self.catalog)
        calls = [tc for m in messages if m.role == "assistant" for tc in m.tool_calls]
        results = [json.loads(m.content) for m in messages if m.role == "tool"]
        if results and results[-1].get("accepted") is False:
            return self._abstain("the extracted sentence did not pass the grounding checks")

        if not nq.drugs:
            return self._abstain("no indexed drug is named; please say which drug you mean")
        if len(nq.drugs) > 1:
            return self._abstain("more than one drug is named; please ask about one at a time")
        drug = self.catalog.get(nq.drugs[0])
        assert drug is not None

        asks_freshness = any(w in question.lower() for w in FRESHNESS_WORDS)
        if asks_freshness and any(t.name == "check_live_label" for t in tools):
            if not calls:
                return self._call(
                    "check_live_label",
                    reason=f"the user asks whether the {drug.brand} label is current",
                    drug=drug.brand,
                )
            last = results[-1]
            if last.get("chunk_id"):
                return self._call(
                    "final_answer", abstain=False, sentences=[{"text": last["text"], "citations": [last["chunk_id"]]}]
                )
            return self._abstain(f"the live label check did not succeed: {last.get('error') or last.get('note')}")

        if not calls:
            return self._call(
                "search_label",
                reason=f"find label text about the question for {drug.brand}",
                query=nq.text,
                drug=drug.key,
            )

        best = self._best_sentence(nq.text, drug.names, results)
        if best is not None:
            text, chunk_id = best
            return self._call("final_answer", abstain=False, sentences=[{"text": text, "citations": [chunk_id]}])

        if nq.section and not any(c.name == "get_section" for c in calls):
            return self._call(
                "get_section",
                reason=f"search did not contain a matching sentence; read the whole {nq.section} section",
                drug=drug.key,
                section=nq.section,
            )
        return self._abstain("the retrieved label text does not address the question")

    @staticmethod
    def _best_sentence(
        query: str, drug_names: tuple[str, ...], results: list[dict[str, Any]]
    ) -> tuple[str, str] | None:
        name_tokens = {t for n in drug_names for t in tokenize(n)}
        terms = set(tokenize(query)) - name_tokens
        terms = {t for t in terms if t.isascii()}
        if not terms:
            return None
        best: tuple[float, str, str] | None = None
        for result in results:
            for hit in result.get("results", []):
                for sentence in split_sentences(hit["text"]):
                    overlap = len(terms & set(tokenize(sentence))) / len(terms)
                    if best is None or overlap > best[0]:
                        best = (overlap, sentence, hit["chunk_id"])
        if best is None or best[0] < MIN_MATCH:
            return None
        return best[1], best[2]
