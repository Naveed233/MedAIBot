from medaibot.corpus import label_sections, pack_sentences, split_subsections


def test_child_sections_already_inside_a_parent_are_dropped(raw_labels):
    tagrisso = next(r for r in raw_labels if r.drug.key == "tagrisso")
    fields = [f for f, _ in label_sections(tagrisso.label)]
    assert "use_in_specific_populations" in fields
    assert "pregnancy" not in fields


def test_html_table_fields_are_skipped(raw_labels):
    tagrisso = next(r for r in raw_labels if r.drug.key == "tagrisso")
    assert "dosage_and_administration_table" in tagrisso.label
    assert all(not f.endswith("_table") for f, _ in label_sections(tagrisso.label))


def test_subsections_split_at_headings_not_cross_references():
    text = "2 DOSAGE Summary 80 mg daily. ( 2.3 ) more (5.3) text. 2.1 Recommended Testing Do tests. 2.3 Recommended Dosage Take 80 mg."
    parts = split_subsections(text)
    assert [label for label, _ in parts] == [None, "2.1 Recommended Testing Do", "2.3 Recommended Dosage Take"]
    assert parts[0][1].endswith("(5.3) text.")


def test_pack_sentences_respects_limit_and_keeps_whole_sentences():
    text = " ".join(f"Sentence number {i} is here." for i in range(60))
    chunks = pack_sentences(text, max_chars=200, overlap_chars=50)
    assert len(chunks) > 1
    assert all(c.endswith(".") for c in chunks)
    # overlap: each chunk after the first starts with the previous chunk's last sentence
    assert chunks[1].startswith(chunks[0].split(". ")[-1].rstrip("."))


def test_chunks_carry_provenance_and_unique_stable_ids(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    c = next(c for c in chunks if c.chunk_id == "tagrisso:contraindications:000")
    assert c.set_id == "5e81b4a7-b971-45e1-9c31-29cea8c87ce7"
    assert c.label_version == "36" and c.effective_time == "20240925"
    assert c.source_url.endswith(c.set_id)
    assert "None" in c.text
