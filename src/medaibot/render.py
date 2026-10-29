"""Plain-text rendering of an answer for the terminal demo."""

from medaibot.models import AgentAnswer


def render_answer(answer: AgentAnswer) -> str:
    lines = [f"Q: {answer.question}", ""]
    number = {c.chunk_id: i for i, c in enumerate(answer.citations, start=1)}
    if answer.abstained:
        lines.append(f"ABSTAINED: {answer.abstain_reason}")
    else:
        for s in answer.sentences:
            marks = "".join(f"[{number[c]}]" for c in s.citations if c in number)
            lines.append(f"{s.text} {marks}".strip())
    if answer.citations:
        lines += ["", "Sources:"]
        for i, c in enumerate(answer.citations, start=1):
            sub = f" > {c.subsection}" if c.subsection else ""
            lines.append(f"  [{i}] {c.drug} | {c.section_title}{sub} | label v{c.label_version} ({c.effective_time})")
            lines.append(f"      {c.chunk_id}  {c.source_url}")
    if answer.grounding is not None:
        g = answer.grounding
        lines += [
            "",
            f"Checks: deterministic {'PASSED' if g.passed else 'FAILED'}, mean token support {g.mean_token_support:.2f}",
        ]
        for i, s in enumerate(g.sentences, start=1):
            judge = f", judge={s.judge_label}" if s.judge_label != "not_checked" else ""
            nums = f", numbers not in source={s.unsupported_numbers}" if s.unsupported_numbers else ""
            lines.append(f"  sentence {i}: support {s.token_support:.2f}{nums}{judge}")
        for p in g.problems:
            lines.append(f"  - {p}")
    lines += ["", f"Trace ({answer.provider}, {answer.latency_ms:.0f} ms):"]
    for t in answer.trace:
        args = ", ".join(f"{k}={v!r}" for k, v in t.arguments.items() if k != "reason")
        reason = t.arguments.get("reason")
        head = f"  {t.step}. {t.kind}" + (f" {t.name}({args})" if t.name else "")
        lines.append(head + f" [{t.duration_ms:.0f} ms]" if t.duration_ms else head)
        if reason:
            lines.append(f"       why: {reason}")
        lines.append(f"       -> {t.summary}")
        if t.flags:
            lines.append(f"       flags: {t.flags}")
    lines += ["", answer.disclaimer]
    return "\n".join(lines)
