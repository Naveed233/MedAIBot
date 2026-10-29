from medaibot.followup import CLASSIFY_SYSTEM, CLASSIFY_TOOL, Turn, resolve_follow_up
from medaibot.llm.base import LLMError, LLMResponse, ToolCall
from medaibot.query import normalise_query


class FakeClassifier:
    """Returns a fixed classification (or raises) and records what it was sent."""

    name = "fake"

    def __init__(self, kind="continuation", question=None, clarify=None, error=None):
        self.args = {"kind": kind, "question": question or ""}
        if clarify:
            self.args["clarify"] = clarify
        self.error, self.calls = error, []

    def complete(self, system, messages, tools, require_tool=True):
        self.calls.append((system, messages))
        if self.error is not None:
            raise self.error
        return LLMResponse(text="", tool_calls=[ToolCall("c1", CLASSIFY_TOOL.name, self.args)], model="fake")


SIDE_EFFECTS = Turn(
    question="What are the most common adverse reactions with TAGRISSO monotherapy?",
    answer="The most common adverse reactions were diarrhea, rash and dry skin.",
)
IMFINZI_DOSE = Turn(
    question="What is the IMFINZI dose for metastatic NSCLC in patients weighing 30 kg or more?",
    answer="1,500 mg every 3 weeks with tremelimumab-actl and platinum-based chemotherapy.",
)


def test_a_first_question_is_left_alone(catalog):
    classifier = FakeClassifier(question="unused")
    resolved = resolve_follow_up("Which of those are the most serious?", [], catalog, classifier)
    assert resolved.question == "Which of those are the most serious?" and not resolved.rewritten
    assert classifier.calls == []


def test_a_standalone_question_skips_the_model(catalog):
    classifier = FakeClassifier(question="unused")
    question = "What is the IMFINZI dose for metastatic NSCLC in patients weighing 30 kg or more?"
    resolved = resolve_follow_up(question, [SIDE_EFFECTS], catalog, classifier)
    assert resolved.question == question and not resolved.rewritten and resolved.kind is None
    assert classifier.calls == []


def test_a_continuation_is_rewritten_and_turns_stay_out_of_the_system_prompt(catalog):
    classifier = FakeClassifier("continuation", "Which serious adverse reactions are described in the TAGRISSO label?")
    resolved = resolve_follow_up("Which of those are the most serious?", [SIDE_EFFECTS], catalog, classifier)
    assert resolved.method == "model" and resolved.kind == "continuation" and "TAGRISSO" in resolved.question
    system, messages = classifier.calls[0]
    assert system == CLASSIFY_SYSTEM
    assert SIDE_EFFECTS.question in messages[0].content and SIDE_EFFECTS.question not in system


def test_a_new_topic_keeps_the_drug_but_not_the_old_question(catalog):
    classifier = FakeClassifier("new_topic", "What does the IMFINZI label say about age (pediatric or geriatric use)?")
    resolved = resolve_follow_up("anything related to age?", [IMFINZI_DOSE], catalog, classifier)
    assert resolved.kind == "new_topic"
    assert normalise_query(resolved.question, catalog).drug == "imfinzi"
    assert "30 kg" not in resolved.question


def test_an_unclear_message_gets_a_clarifying_question_not_a_guess(catalog):
    turns = [Turn(question="What is the TAGRISSO dose?"), Turn(question="What is the LYNPARZA dose?")]
    clarify = "Do you mean the side effects of TAGRISSO or of LYNPARZA?"
    classifier = FakeClassifier("unclear", "What about side effects?", clarify=clarify)
    resolved = resolve_follow_up("What about side effects?", turns, catalog, classifier)
    assert resolved.kind == "unclear" and resolved.clarify == clarify
    assert resolved.question == "What about side effects?"


def test_a_message_the_model_calls_new_gets_no_drug_carried_over(catalog):
    classifier = FakeClassifier("new", "What is the recommended starting dose?")
    resolved = resolve_follow_up("What is the recommended starting dose?", [SIDE_EFFECTS], catalog, classifier)
    assert resolved.question == "What is the recommended starting dose?" and resolved.kind == "new"


def test_a_rewrite_that_brings_in_another_drug_is_rejected(catalog):
    classifier = FakeClassifier("continuation", "Which serious adverse reactions does LYNPARZA cause?")
    resolved = resolve_follow_up("Which of those are the most serious?", [SIDE_EFFECTS], catalog, classifier)
    assert resolved.method == "drug-carried"
    assert resolved.question == "About TAGRISSO: Which of those are the most serious?"


def test_a_failed_classification_falls_back_to_carrying_the_drug(catalog):
    classifier = FakeClassifier(error=LLMError("upstream 503"))
    resolved = resolve_follow_up("And the recommended dose?", [SIDE_EFFECTS], catalog, classifier)
    assert resolved.method == "drug-carried" and resolved.kind is None
    assert normalise_query(resolved.question, catalog).drug == "tagrisso"


def test_an_unknown_kind_is_treated_as_a_failed_classification(catalog):
    classifier = FakeClassifier("guess", "Which serious adverse reactions does TAGRISSO cause?")
    resolved = resolve_follow_up("Which of those are the most serious?", [SIDE_EFFECTS], catalog, classifier)
    assert resolved.method == "drug-carried" and resolved.kind is None


def test_without_a_model_the_drug_is_carried_over_in_japanese(catalog):
    resolved = resolve_follow_up("副作用は？", [Turn(question="タグリッソの用量は？")], catalog)
    assert resolved.method == "drug-carried"
    assert normalise_query(resolved.question, catalog).drug == "tagrisso"


def test_no_earlier_drug_means_no_guess(catalog):
    resolved = resolve_follow_up("Which of those are the most serious?", [Turn(question="Hello")], catalog)
    assert resolved.question == "Which of those are the most serious?" and not resolved.rewritten


def test_a_bare_drug_name_answers_the_earlier_question(catalog):
    earlier = Turn(question="What is the recommended starting dose?", answer="Which drug do you mean?")
    classifier = FakeClassifier("drug_reply", "What is the recommended starting dose of TAGRISSO?")
    resolved = resolve_follow_up("TAGRISSO", [earlier], catalog, classifier)
    assert resolved.question == "What is the recommended starting dose of TAGRISSO?"
    assert resolved.kind == "drug_reply"


def test_only_the_last_three_turns_are_sent(catalog):
    turns = [Turn(question="What is the dose of LYNPARZA?")] + [
        Turn(question=f"Question {i} about TAGRISSO") for i in range(3)
    ]
    classifier = FakeClassifier("continuation", "How should TAGRISSO tablets be stored?")
    resolved = resolve_follow_up("How should it be stored?", turns, catalog, classifier)
    assert resolved.method == "model"
    assert "LYNPARZA" not in classifier.calls[0][1][0].content
