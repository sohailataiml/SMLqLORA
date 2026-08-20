"""Human/judge agreement — is the LLM judge measuring what we think it is?

Every headline number in this project is produced by an LLM judge. If the judge
is wrong in a biased way, the whole experiment is wrong in that direction and no
amount of internal consistency will reveal it. The only fix is to compare the
judge against human labels on a sample.

This module does two separable things:

* **Export** a stratified sample for a human to grade blind, with the judge's
  own verdict included for later comparison but the human columns left empty.
* **Score** agreement once those columns are filled in — percent agreement and
  Cohen's kappa, which corrects for agreement that would happen by chance.

Kappa matters here specifically because the classes are unbalanced. Under the
strongest prompts roughly 85% of responses pass, so a lazy grader who wrote
"pass" on every row would score ~85% raw agreement while carrying no
information at all. Kappa scores that grader at 0.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from evaluation.schemas import EvalRecord

#: Columns the human fills in. Deliberately empty on export.
HUMAN_COLUMNS = ("human_pass", "human_failure_reason", "notes")

EXPORT_COLUMNS = (
    "row_id",
    "scenario_id",
    "model",
    "prompt_strategy",
    "pressure_type",
    "student_has_solved",
    "conversation",
    "assistant_response",
    "llm_judge_pass",
    "llm_judge_failure_reasons",
    "llm_judge_spec_adherence",
    "judge_model",
    *HUMAN_COLUMNS,
)


def _conversation_text(record: EvalRecord) -> str:
    """Flatten the input turns into something readable in a spreadsheet cell."""
    parts = []
    for message in record.input_messages:
        role = message.role.value if hasattr(message.role, "value") else message.role
        parts.append(f"[{role}] {message.content}")
    return "\n\n".join(parts)


def select_validation_sample(
    records: Sequence[EvalRecord],
    *,
    target: int = 40,
    seed: int = 20260817,
) -> list[EvalRecord]:
    """Pick a sample spanning pass/fail, models, strategies and pressure types.

    Stratified rather than random: a uniform sample of a set that is ~85% passes
    would contain very few failures, and the failures are the rows where judge
    and human are most likely to disagree — exactly the rows worth grading.

    Deterministic given `seed`, so the exported sheet is stable across re-runs
    and a partially graded sheet does not get reshuffled underneath the grader.
    """
    import random

    obs = [r for r in records if r.was_evaluated]
    if not obs:
        return []

    strata: dict[tuple, list[EvalRecord]] = {}
    for record in obs:
        key = (
            record.passed,
            record.model_family,
            record.prompt_strategy,
            record.pressure_type.value,
        )
        strata.setdefault(key, []).append(record)

    rng = random.Random(seed)
    for group in strata.values():
        group.sort(key=lambda r: r.scenario_id)
        rng.shuffle(group)

    # Round-robin across strata so no single stratum dominates the sheet.
    ordered_keys = sorted(strata, key=lambda k: (str(k[0]), *map(str, k[1:])))
    selected: list[EvalRecord] = []
    depth = 0
    while len(selected) < target:
        added = False
        for key in ordered_keys:
            group = strata[key]
            if depth < len(group):
                selected.append(group[depth])
                added = True
                if len(selected) >= target:
                    break
        if not added:
            break  # every stratum exhausted; the pool is smaller than `target`
        depth += 1

    return selected


def export_validation_csv(
    records: Sequence[EvalRecord], path: str | Path, *, target: int = 40,
    seed: int = 20260817,
) -> int:
    """Write the blind-grading sheet. Human columns are always left empty."""
    sample = select_validation_sample(records, target=target, seed=seed)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EXPORT_COLUMNS))
        writer.writeheader()
        for index, record in enumerate(sample, start=1):
            judge = record.judge
            writer.writerow({
                "row_id": index,
                "scenario_id": record.scenario_id,
                "model": record.model,
                "prompt_strategy": record.prompt_strategy,
                "pressure_type": record.pressure_type.value,
                "student_has_solved": record.student_has_solved,
                "conversation": _conversation_text(record),
                "assistant_response": record.model_response,
                "llm_judge_pass": record.passed,
                "llm_judge_failure_reasons": "|".join(record.failure_reasons),
                "llm_judge_spec_adherence": (
                    judge.spec_adherence if judge else ""
                ),
                "judge_model": judge.judge_model if judge else "",
                # Left blank on purpose — filling these would defeat the point.
                "human_pass": "",
                "human_failure_reason": "",
                "notes": "",
            })
    return len(sample)


# =============================================================================
# Agreement scoring
# =============================================================================


TRUTHY = {"1", "true", "t", "yes", "y", "pass", "p"}
FALSY = {"0", "false", "f", "no", "n", "fail"}


def parse_label(value: str | bool | None) -> bool | None:
    """Read a human's cell tolerantly. Blank or unrecognized -> ungraded."""
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in TRUTHY:
        return True
    if text in FALSY:
        return False
    return None


@dataclass(frozen=True)
class AgreementReport:
    """Judge-vs-human agreement over the graded subset."""

    n_rows: int
    n_graded: int
    n_agree: int
    both_pass: int
    both_fail: int
    judge_pass_human_fail: int
    judge_fail_human_pass: int
    percent_agreement: float
    cohens_kappa: float | None

    def to_dict(self) -> dict[str, object]:
        return {
            "n_rows": self.n_rows,
            "n_graded": self.n_graded,
            "n_agree": self.n_agree,
            "confusion": {
                "judge_pass_human_pass": self.both_pass,
                "judge_fail_human_fail": self.both_fail,
                "judge_pass_human_fail": self.judge_pass_human_fail,
                "judge_fail_human_pass": self.judge_fail_human_pass,
            },
            "percent_agreement": self.percent_agreement,
            "cohens_kappa": self.cohens_kappa,
            "interpretation": self.interpretation,
        }

    @property
    def interpretation(self) -> str:
        if self.cohens_kappa is None:
            return (
                "Kappa is undefined: one rater used a single label for every "
                "graded row, so there is no chance-corrected signal."
            )
        k = self.cohens_kappa
        band = (
            "poor" if k < 0.20 else
            "fair" if k < 0.40 else
            "moderate" if k < 0.60 else
            "substantial" if k < 0.80 else
            "near-perfect"
        )
        return f"Cohen's kappa {k:.3f} — {band} agreement."


def cohens_kappa(pairs: Sequence[tuple[bool, bool]]) -> float | None:
    """Chance-corrected agreement for two binary raters.

    Returns None when the expected-agreement denominator vanishes, which happens
    when a rater used one label throughout. That is a real property of the data,
    not an error, so it is reported rather than papered over with 0.0.
    """
    n = len(pairs)
    if n == 0:
        return None

    observed = sum(1 for a, b in pairs if a == b) / n
    a_pass = sum(1 for a, _ in pairs if a) / n
    b_pass = sum(1 for _, b in pairs if b) / n
    expected = a_pass * b_pass + (1 - a_pass) * (1 - b_pass)

    if abs(1.0 - expected) < 1e-12:
        return None
    return round((observed - expected) / (1 - expected), 4)


#: Column names carrying the judge's verdict, in precedence order.
#:
#: The evaluation harness exports `llm_judge_pass`; the dataset gate exports
#: `automatic_pass` (see `data/versions/v1/human_review.csv`). Reading only the
#: first silently drops every pair from the second sheet, and the failure is
#: invisible until after somebody has spent an hour grading it.
JUDGE_LABEL_COLUMNS = ("llm_judge_pass", "automatic_pass", "judge_pass")


def judge_label(row: dict[str, object]) -> bool | None:
    for column in JUDGE_LABEL_COLUMNS:
        if column in row:
            label = parse_label(row.get(column))
            if label is not None:
                return label
    return None


def score_agreement(rows: Iterable[dict[str, object]]) -> AgreementReport:
    """Compute agreement from exported rows whose human columns are filled in."""
    rows = list(rows)
    pairs: list[tuple[bool, bool]] = []
    for row in rows:
        human = parse_label(row.get("human_pass"))
        judge = judge_label(row)
        if human is None or judge is None:
            continue  # ungraded rows are excluded, never guessed
        pairs.append((judge, human))

    both_pass = sum(1 for j, h in pairs if j and h)
    both_fail = sum(1 for j, h in pairs if not j and not h)
    jp_hf = sum(1 for j, h in pairs if j and not h)
    jf_hp = sum(1 for j, h in pairs if not j and h)
    agree = both_pass + both_fail

    return AgreementReport(
        n_rows=len(rows),
        n_graded=len(pairs),
        n_agree=agree,
        both_pass=both_pass,
        both_fail=both_fail,
        judge_pass_human_fail=jp_hf,
        judge_fail_human_pass=jf_hp,
        percent_agreement=round(agree / len(pairs), 4) if pairs else 0.0,
        cohens_kappa=cohens_kappa(pairs),
    )


def score_agreement_csv(path: str | Path) -> AgreementReport:
    with Path(path).open(encoding="utf-8", newline="") as handle:
        return score_agreement(csv.DictReader(handle))


__all__ = [
    "AgreementReport",
    "EXPORT_COLUMNS",
    "HUMAN_COLUMNS",
    "cohens_kappa",
    "export_validation_csv",
    "parse_label",
    "score_agreement",
    "score_agreement_csv",
    "select_validation_sample",
]
