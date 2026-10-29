"""Deterministic query normalisation: find the drug, guess the section, translate Japanese terms.

The index is English, so a question like タグリッソの用量は？ finds nothing by keyword and
very little by an English embedding model. Mapping the katakana brand to TAGRISSO and 用量
to "recommended dosage" is cheap and testable. An LLM can do the same rewrite, but this
layer works without one and gives the eval a fixed baseline.
"""

from dataclasses import dataclass, field

from medaibot.drugs import DrugCatalog
from medaibot.text import is_cjk, normalize

# Section field -> keywords (English and Japanese) that suggest it.
SECTION_HINTS: dict[str, tuple[str, ...]] = {
    "dosage_and_administration": (
        "dose",
        "dosage",
        "dosing",
        "how much",
        "how to take",
        "how should",
        "administer",
        "用量",
        "用法",
        "投与量",
        "服用",
        "飲み方",
    ),
    "contraindications": ("contraindicat", "must not take", "should not take", "禁忌"),
    "boxed_warning": ("boxed warning", "black box", "警告"),
    "warnings_and_cautions": ("warning", "precaution", "注意"),
    "adverse_reactions": ("side effect", "adverse", "副作用", "有害事象"),
    "drug_interactions": ("interact", "together with", "combine", "相互作用", "併用"),
    "use_in_specific_populations": (
        "pregnan",
        "lactation",
        "breastfeed",
        "pediatric",
        "children",
        "elderly",
        "geriatric",
        "older adult",
        "older patient",
        " age",  # the leading space keeps "dosage" and "usage" out
        "年齢",
        "renal",
        "kidney",
        "hepatic",
        "liver",
        "妊娠",
        "授乳",
        "小児",
        "高齢",
        "腎",
        "肝",
    ),
    "indications_and_usage": ("indicat", "used for", "what is it for", "treat", "効能", "効果", "適応"),
    "clinical_pharmacology": ("mechanism", "how does it work", "pharmacokinetic", "half-life", "作用機序", "半減期"),
    "how_supplied": ("storage", "store", "supplied", "保管", "保存"),
    "overdosage": ("overdose", "過量"),
}

# Japanese medical terms -> English words found in US labels.
JA_TO_EN: dict[str, str] = {
    "用量": "recommended dosage dose",
    "用法": "dosage administration",
    "投与量": "recommended dosage dose",
    "服用": "take dose",
    "禁忌": "contraindications",
    "警告": "boxed warning",
    "注意": "warnings precautions",
    "副作用": "adverse reactions side effects",
    "有害事象": "adverse reactions",
    "相互作用": "drug interactions",
    "併用": "drug interactions combination",
    "妊娠": "pregnancy",
    "授乳": "lactation breastfeeding",
    "小児": "pediatric use",
    "高齢": "geriatric use",
    "腎": "renal impairment",
    "肝": "hepatic impairment",
    "効能": "indications and usage",
    "効果": "indications and usage",
    "適応": "indications",
    "作用機序": "mechanism of action",
    "半減期": "half-life",
    "保管": "storage",
    "保存": "storage",
    "過量": "overdosage",
}


@dataclass(frozen=True)
class NormalisedQuery:
    original: str
    text: str
    drugs: list[str] = field(default_factory=list)
    section: str | None = None
    translated_terms: list[str] = field(default_factory=list)

    @property
    def drug(self) -> str | None:
        """The drug filter to use, only when exactly one drug is named."""
        return self.drugs[0] if len(self.drugs) == 1 else None


def guess_section(question: str) -> str | None:
    norm = normalize(question)
    for section, hints in SECTION_HINTS.items():
        if any(h in norm for h in hints):
            return section
    return None


def normalise_query(question: str, catalog: DrugCatalog) -> NormalisedQuery:
    drugs = catalog.find_in_text(question)
    parts = [question]
    translated: list[str] = []
    if is_cjk(question):
        for drug in drugs:
            parts.append(f"{drug.brand} {drug.generic}")
        for term, english in JA_TO_EN.items():
            if term in question:
                parts.append(english)
                translated.append(term)
    return NormalisedQuery(
        original=question,
        text=" ".join(parts),
        drugs=[d.key for d in drugs],
        section=guess_section(question),
        translated_terms=translated,
    )
