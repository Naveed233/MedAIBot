from medaibot.followup import REWRITE_SYSTEM, REWRITE_TOOL, Turn, resolve_follow_up
from medaibot.llm.base import LLMError, LLMResponse, ToolCall
from medaibot.query import normalise_query


class FakeRewriter:
    """Returns a fixed rewrite (or raises) and records what it was sent."""

    name = "fake"

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def complete(self, system, messages, tools, require_tool=True):
        self.calls.append((system, messages))
        if self.error is not None:
            raise self.error
        call = ToolCall("r1", REWRITE_TOOL.name, {"question": self.reply})
        return LLMResponse(text="", tool_calls=[call], model="fake")


SIDE_EFFECTS = Turn(
    question="What are the most common adverse reactions with TAGRISSO monotherapy?",
    answer="The most common adverse reactions were diarrhea, rash and dry skin.",
)


def test_a_first_question_is_left_alone(catalog):
    rewriter = FakeRewriter("unused")
    resolved = resolve_follow_up("Which of those are the most serious?", [], catalog, rewriter)
    assert resolved.question == "Which of those are the most serious?" and not resolved.rewritten
    assert rewriter.calls == []


def test_a_standalone_question_skips_the_model(catalog):
    rewriter = FakeRewriter("unused")
    question = "What is the IMFINZI dose for metastatic NSCLC in patients weighing 30 kg or more?"
    resolved = resolve_follow_up(question, [SIDE_EFFECTS], catalog, rewriter)
    assert resolved.question == question and not resolved.rewritten
    assert rewriter.calls == []


def test_the_model_rewrite_is_used_and_turns_stay_out_of_the_system_prompt(catalog):
    rewriter = FakeRewriter("Which serious adverse reactions are described in the TAGRISSO label?")
    resolved = resolve_follow_up("Which of those are the most serious?", [SIDE_EFFECTS], catalog, rewriter)
    assert resolved.method == "model" and "TAGRISSO" in resolved.question
    system, messages = rewriter.calls[0]
    assert system == REWRITE_SYSTEM
    assert SIDE_EFFECTS.question in messages[0].content and SIDE_EFFECTS.question not in system


def test_a_rewrite_that_brings_in_another_drug_is_rejected(catalog):
    rewriter = FakeRewriter("Which serious adverse reactions does LYNPARZA cause?")
    resolved = resolve_follow_up("Which of those are the most serious?", [SIDE_EFFECTS], catalog, rewriter)
    assert resolved.method == "drug-carried"
    assert resolved.question == "About TAGRISSO: Which of those are the most serious?"


def test_a_failed_rewrite_falls_back_to_carrying_the_drug(catalog):
    rewriter = FakeRewriter(error=LLMError("upstream 503"))
    resolved = resolve_follow_up("And the recommended dose?", [SIDE_EFFECTS], catalog, rewriter)
    assert resolved.method == "drug-carried"
    assert normalise_query(resolved.question, catalog).drug == "tagrisso"


def test_without_a_model_the_drug_is_carried_over_in_japanese(catalog):
    resolved = resolve_follow_up("副作用は？", [Turn(question="タグリッソの用量は？")], catalog)
    assert resolved.method == "drug-carried"
    assert normalise_query(resolved.question, catalog).drug == "tagrisso"


def test_no_earlier_drug_means_no_guess(catalog):
    resolved = resolve_follow_up("Which of those are the most serious?", [Turn(question="Hello")], catalog)
    assert resolved.question == "Which of those are the most serious?" and not resolved.rewritten


def test_a_bare_drug_name_answers_the_earlier_question(catalog):
    earlier = Turn(question="What is the recommended starting dose?", answer="Which drug do you mean?")
    rewriter = FakeRewriter("What is the recommended starting dose of TAGRISSO?")
    resolved = resolve_follow_up("TAGRISSO", [earlier], catalog, rewriter)
    assert resolved.question == "What is the recommended starting dose of TAGRISSO?"
    assert resolved.method == "model"


def test_only_the_last_three_turns_are_sent(catalog):
    turns = [Turn(question="What is the dose of LYNPARZA?")] + [
        Turn(question=f"Question {i} about TAGRISSO") for i in range(3)
    ]
    rewriter = FakeRewriter("How should TAGRISSO tablets be stored?")
    resolved = resolve_follow_up("How should it be stored?", turns, catalog, rewriter)
    assert resolved.method == "model"
    assert "LYNPARZA" not in rewriter.calls[0][1][0].content
