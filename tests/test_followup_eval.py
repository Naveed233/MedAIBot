from medaibot.followup import Resolved
from medaibot.followup_eval import FollowUpCase, load_cases, score_case, summarise

AGE = FollowUpCase(
    id="t1",
    turns=[{"question": "What is the IMFINZI dose for patients weighing 30 kg or more?", "answer": ""}],
    message="anything related to age?",
    kinds=["new_topic"],
    drug="imfinzi",
    any_terms=["age", "geriatric"],
    no_terms=["30 kg"],
)


def test_a_good_new_topic_rewrite_scores_on_every_check(catalog):
    resolved = Resolved("What does the IMFINZI label say about age?", "model", "new_topic")
    row = score_case(AGE, resolved, catalog)
    assert row["kind_ok"] and row["drug_ok"] and row["question_ok"] and row["all_ok"]


def test_copying_the_old_question_fails_the_question_check(catalog):
    resolved = Resolved(
        "What is the IMFINZI dose for patients weighing 30 kg or more related to age?", "model", "continuation"
    )
    row = score_case(AGE, resolved, catalog)
    assert not row["kind_ok"] and row["drug_ok"] and not row["question_ok"] and not row["all_ok"]


def test_an_unclear_case_needs_a_clarifying_question(catalog):
    case = FollowUpCase(id="t2", turns=[], message="What about side effects?", kinds=["unclear"])
    row = score_case(case, Resolved("What about side effects?", "model", "unclear", "TAGRISSO or LYNPARZA?"), catalog)
    assert row["all_ok"] and row["drug_ok"] is None
    assert summarise([row])["drug_ok"] is None


def test_the_case_file_loads_and_covers_every_kind():
    kinds = {k for case in load_cases() for k in case.kinds}
    assert {"new", "continuation", "new_topic", "redo", "drug_reply", "unclear"} <= kinds
