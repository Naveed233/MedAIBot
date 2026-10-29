"""Turn raw openFDA label records into section-aware chunks with provenance."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from medaibot.drugs import Drug, DrugCatalog
from medaibot.models import Chunk
from medaibot.openfda import dailymed_url
from medaibot.text import split_sentences

# Metadata, packaging text and lay-language duplicates of professional sections.
SKIP_FIELDS = frozenset(
    {
        "set_id",
        "id",
        "effective_time",
        "version",
        "openfda",
        "spl_product_data_elements",
        "package_label_principal_display_panel",
        "spl_unclassified_section",
        "spl_medguide",
        "spl_patient_package_insert",
        "patient_medication_information",
        "instructions_for_use",
    }
)

SECTION_TITLES = {
    "boxed_warning": "Boxed Warning",
    "recent_major_changes": "Recent Major Changes",
    "indications_and_usage": "Indications and Usage",
    "dosage_and_administration": "Dosage and Administration",
    "dosage_forms_and_strengths": "Dosage Forms and Strengths",
    "contraindications": "Contraindications",
    "warnings_and_cautions": "Warnings and Precautions",
    "warnings": "Warnings",
    "precautions": "Precautions",
    "adverse_reactions": "Adverse Reactions",
    "drug_interactions": "Drug Interactions",
    "use_in_specific_populations": "Use in Specific Populations",
    "pregnancy": "Pregnancy",
    "lactation": "Lactation",
    "pediatric_use": "Pediatric Use",
    "geriatric_use": "Geriatric Use",
    "overdosage": "Overdosage",
    "description": "Description",
    "clinical_pharmacology": "Clinical Pharmacology",
    "mechanism_of_action": "Mechanism of Action",
    "pharmacodynamics": "Pharmacodynamics",
    "pharmacokinetics": "Pharmacokinetics",
    "nonclinical_toxicology": "Nonclinical Toxicology",
    "clinical_studies": "Clinical Studies",
    "how_supplied": "How Supplied/Storage and Handling",
    "storage_and_handling": "Storage and Handling",
    "information_for_patients": "Patient Counseling Information",
}

# A subsection heading such as " 2.1 Recommended Dosage". Cross-references look like
# "( 2.3 )" or "(5.3)", so a heading must not follow an opening bracket.
_SUBSECTION_RE = re.compile(r"(?<![(\[])\s(\d{1,2}\.\d{1,2})\s+(?=[A-Z])")
_SPACES_RE = re.compile(r"\s+")
_SMALL_WORDS = frozenset({"and", "of", "to", "in", "for", "with", "the", "or", "on", "a"})


@dataclass(frozen=True)
class RawLabel:
    drug: Drug
    provenance: dict[str, Any]
    label: dict[str, Any]


def section_title(field: str) -> str:
    return SECTION_TITLES.get(field, field.replace("_", " ").title())


def load_raw(path: Path, catalog: DrugCatalog) -> RawLabel:
    record = json.loads(path.read_text(encoding="utf-8"))
    set_id = record["label"]["set_id"]
    drug = catalog.by_set_id(set_id)
    if drug is None:
        raise ValueError(f"{path} has set_id {set_id}, which is not in the catalogue")
    return RawLabel(drug=drug, provenance=record["provenance"], label=record["label"])


def load_raw_dir(raw_dir: Path, catalog: DrugCatalog) -> list[RawLabel]:
    return [load_raw(p, catalog) for p in sorted(raw_dir.glob("*.json"))]


def label_sections(label: dict[str, Any]) -> list[tuple[str, str]]:
    """(field, text) for each prose section, without tables or duplicated child sections.

    openFDA repeats subsections as their own fields, e.g. `pregnancy` is also inside
    `use_in_specific_populations`. Indexing both would return the same text twice.
    """
    candidates: list[tuple[str, str]] = []
    for field, value in label.items():
        if field in SKIP_FIELDS or field.endswith("_table") or not isinstance(value, list):
            continue
        text = _SPACES_RE.sub(" ", " ".join(str(v) for v in value)).strip()
        if text:
            candidates.append((field, text))

    kept: list[tuple[str, str]] = []
    for field, text in sorted(candidates, key=lambda c: len(c[1]), reverse=True):
        probe = text[:300]
        if any(probe in parent for _, parent in kept):
            continue
        kept.append((field, text))
    order = {field: i for i, (field, _) in enumerate(candidates)}
    return sorted(kept, key=lambda c: order[c[0]])


def _subsection_label(number: str, following: str) -> str:
    words: list[str] = []
    for word in following.split()[:10]:
        if word == "•" or not (word[0].isupper() or word in _SMALL_WORDS):
            break
        words.append(word)
    return " ".join([number, *words[:8]]).strip()


def split_subsections(text: str) -> list[tuple[str | None, str]]:
    """Split a section at its numbered subsection headings. The text before the
    first heading is the section summary and gets no subsection label."""
    matches = list(_SUBSECTION_RE.finditer(text))
    if not matches:
        return [(None, text)]
    parts: list[tuple[str | None, str]] = []
    head = text[: matches[0].start()].strip()
    if head:
        parts.append((None, head))
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[match.start() : end].strip()
        parts.append((_subsection_label(match.group(1), text[match.end() : end]), body))
    return parts


def pack_sentences(text: str, max_chars: int = 900, overlap_chars: int = 250) -> list[str]:
    """Group whole sentences into chunks of at most `max_chars`.

    The last sentence of a chunk is repeated at the start of the next one when it is
    short, so a fact split across a boundary is still found with its context.
    """
    sentences: list[str] = []
    for sentence in split_sentences(text):
        while len(sentence) > max_chars:
            cut = sentence.rfind(" ", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            sentences.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            sentences.append(sentence)

    chunks: list[str] = []
    current: list[str] = []
    for sentence in sentences:
        if current and len(" ".join([*current, sentence])) > max_chars:
            chunks.append(" ".join(current))
            last = current[-1]
            current = [last] if len(last) <= overlap_chars else []
        current.append(sentence)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_label(raw: RawLabel, max_chars: int = 900) -> list[Chunk]:
    drug, label = raw.drug, raw.label
    chunks: list[Chunk] = []
    for field, text in label_sections(label):
        position = 0
        for subsection, body in split_subsections(text):
            for piece in pack_sentences(body, max_chars=max_chars):
                chunks.append(
                    Chunk(
                        chunk_id=f"{drug.key}:{field}:{position:03d}",
                        drug=drug.key,
                        brand=drug.brand,
                        generic=drug.generic,
                        section=field,
                        section_title=section_title(field),
                        subsection=subsection,
                        position=position,
                        text=piece,
                        set_id=label["set_id"],
                        label_version=str(label.get("version", "")),
                        effective_time=str(label.get("effective_time", "")),
                        source_url=dailymed_url(label["set_id"]),
                    )
                )
                position += 1
    return chunks


def build_chunks(raw_labels: list[RawLabel], max_chars: int = 900) -> list[Chunk]:
    return [chunk for raw in raw_labels for chunk in chunk_label(raw, max_chars=max_chars)]
