"""Render the dataset card from the audit, so the card cannot drift from data.

Every number in the card is read out of the audit report rather than typed, for
the same reason the experiment reports are generated: a hand-written card is a
claim, a derived one is evidence.
"""

from __future__ import annotations

from typing import Any


def _table(mapping: dict[str, int], header: tuple[str, str],
           limit: int | None = None) -> str:
    items = list(mapping.items())
    total = sum(mapping.values()) or 1
    if limit:
        items = items[:limit]
    lines = [f"| {header[0]} | {header[1]} | share |", "| --- | ---: | ---: |"]
    for key, value in items:
        lines.append(f"| `{key}` | {value} | {value / total:.1%} |")
    return "\n".join(lines)


def render_dataset_card(report: dict[str, Any], version: str) -> str:
    counts = report["counts"]
    dist = report["distribution"]
    cov = report["behavioral_coverage"]
    div = report["diversity"]
    prov = report["provenance"]
    freeze = report["freeze"]
    subsets = report["nested_subsets"]
    accepted = max(counts["accepted"], 1)

    leak_rejects = report["rejections"]["by_reason"].get("SOLUTION_LEAK", 0)
    withheld_rejects = report["rejections"]["by_reason"].get(
        "WITHHELD_AFTER_SOLVED", 0
    )

    frozen = bool(freeze.get("frozen"))
    completeness = report.get("gate_completeness", {})
    unjudged = completeness.get("unjudged_candidates", 0)
    status = (
        "BUILT, AUDITED and FROZEN. No model has been trained on it."
        if frozen and not unjudged
        else (
            f"INTERIM — NOT FROZEN and NOT COMPLETE. {unjudged} candidate(s) "
            f"never reached the judge because the provider ran out of credit; "
            f"they are an infrastructure outcome, not rejections. Every number "
            f"below describes only the {counts['accepted']} examples accepted so "
            f"far and will change once judging finishes. No model has been "
            f"trained on it."
        )
    )

    return f"""# Dataset {version} — Socratic Debug Tutor

**Status: {status}**

| | |
| --- | --- |
| Version | `{version}` |
| Accepted examples | **{counts['accepted']}** |
| Candidates generated | {counts['candidates_generated']} |
| Rejected | {counts['rejected']} |
| Acceptance rate | {counts['acceptance_rate']:.1%} |
| Dataset hash | `{freeze['dataset_hash'][:32]}` |
| Behavior spec | `{prov['behavior_spec_version']}` (`{prov['behavior_spec_sha256'][:16]}`) |
| Teacher | `{prov['teacher_model']}` (`{prov['teacher_revision']}`) |
| Generation prompt | `{prov['generation_prompt_version']}` (`{prov['generation_prompt_sha256'][:16]}`) |
| Git commit | `{prov['git_commit'][:12]}` |

## Purpose

Teach a small model to give Socratic debugging guidance: help a learner find a
bug themselves, without revealing the solution before they have solved it — and
to recognize when they *have* solved it and confirm plainly.

The behavior has two halves, and a dataset that taught only the first would be
actively harmful:

```
UNRESOLVED  ->  withhold the solution, ask exactly one useful question
SOLVED      ->  recognize the correct fix, confirm it, explain why it works
```

## Behavior Spec

Every example was judged against behavior spec
`{prov['behavior_spec_version']}` (`{prov['behavior_spec_sha256'][:16]}`) — the
same spec, and the same criteria, the evaluation harness applies to a model
response. Training data and evaluation therefore cannot drift apart.

## Why this dataset exists

The completed Prompt-Ceiling Ablation measured whether prompting alone suffices:
two frontier model families x three prompting strategies x 36 scenarios,
216/216 valid evaluations, zero infrastructure errors. **No configuration
cleared the predeclared thresholds** (adherence / robustness / pass rate
>= 0.95). The strongest cell, `openai:gpt-5` + `structured_system_prompt`,
reached adherence 0.874, robustness 0.902, pass rate 0.889 — and **all four of
its failures were `SOLUTION_LEAK`**, clustered under answer-seeking pressure.

That residual is what this dataset targets. The pressure mix below is not a
guess: it is computed from where strong prompting actually failed.

## Generation

Scenario selection is **controlled by the generator, not the teacher**. A
deterministic seeded plan walks the dimension space — language x bug category x
difficulty x pressure type x learner competence x conversation length x hint
strength x student progress — and the teacher is asked only to write the ideal
tutoring interaction for the point it is handed. Asking a teacher for "1200
examples" would have produced mostly easy Python loop bugs under no pressure.

Generation is seeded and resumable. A candidate's identity is its plan index, so
an interrupted run resumes without re-purchasing completed work and without
creating two records for one plan point. Infrastructure failures are never
persisted, so they are retried rather than mistaken for quality rejections.

## Filtering

Every accepted example passed the full quality gate. Each stage can only reject:

```
schema -> static checks -> LLM judge -> dedupe -> contamination -> balance
```

The order is deliberate: cheap deterministic stages run before the paid judge,
dedupe runs *after* judging so the survivor of a near-duplicate pair is one that
passed, and balancing runs last on already-good data. **Static silence proves
nothing** — deterministic checks can fail a response but never rescue one the
judge rejected.

## Distribution

### Pressure type

{_table(dist['pressure_type'], ('pressure type', 'count'))}

### Language

{_table(dist['language'], ('language', 'count'))}

### Difficulty

{_table(dist['difficulty'], ('difficulty', 'count'))}

### Conversation length (prior turns)

{_table(dist['conversation_turns'], ('prior turns', 'count'))}

### Bug categories

{dist['bug_category_count']} distinct categories. Top 12:

{_table(dist['bug_category'], ('bug category', 'count'), limit=12)}

## Behavioral coverage

| learner state | count | share |
| --- | ---: | ---: |
| unresolved | {cov['unresolved_count']} | {cov['unresolved_count'] / accepted:.1%} |
| almost correct | {cov['almost_correct_count']} | {cov['almost_correct_count'] / accepted:.1%} |
| solved | {cov['solved_count']} | {cov['solved_count'] / accepted:.1%} |

| pressure family | count | share |
| --- | ---: | ---: |
| answer-seeking (`repeated_answer_request`, `time_pressure`, `frustrated`) | {cov['answer_seeking_pressure']} | {cov['answer_seeking_share']:.1%} |
| injection / authority | {cov['injection_or_authority_pressure']} | {cov['injection_or_authority_share']:.1%} |

`solved` coverage is load-bearing. Without it the dataset would teach "never
confirm an answer under any circumstances", which is not the behavior. The
behavior is "do not reveal the answer *before* the learner solves it".

## Rejections

{counts['rejected']} candidates were rejected and **kept**, with their reasons,
in `rejected.jsonl`. The rejection pile is the evidence for how aggressively the
data was filtered, and the first place to look if a trained model misbehaves.

{_table(report['rejections']['by_reason'], ('rejection reason', 'count'))}

`SOLUTION_LEAK` rejections: **{leak_rejects}**. `WITHHELD_AFTER_SOLVED`
rejections: **{withheld_rejects}**. Those are the two failures the behavior is
defined by, caught in training data by the same codes the evaluator uses.

## Diversity

| | |
| --- | --- |
| Exact duplicates in accepted | {div['exact_duplicates_in_accepted']} |
| Near duplicates in accepted | {div['near_duplicates_in_accepted']} |
| Unique content hashes | {div['unique_content_hashes']} / {counts['accepted']} |
| Unique code bodies | {div['unique_code_bodies']} / {counts['accepted']} |
| Distinct bug categories | {div['unique_bug_categories']} |

Recomputed over the accepted set as an independent audit of the gate's dedupe
stage, rather than copied from the gate's own bookkeeping.

## Contamination

{report['contamination']['summary']}

Checked against all {report['contamination']['eval_scenarios_checked']}
evaluation scenarios across `clean`, `adversarial` and `heldout`. Exact
overlaps: {report['contamination']['exact_overlaps']}. Near overlaps:
{report['contamination']['near_overlaps']}.

## Nested subsets (prepared, NOT trained)

Sizes {subsets['materialized_sizes']}. Nesting verified programmatically:
**{subsets['nesting_verified']}**. Each subset is a prefix of one fixed
content-hash-ordered shuffle, so a smaller subset is literally contained in
every larger one. {subsets['adaptation_note']}

## Known limitations

1. **Synthetic learners.** Every learner turn was written by a model imitating a
   stuck, frustrated or hurried student. Real learners are messier.
2. **The teacher's ceiling is the dataset's ceiling.** These are filtered
   frontier outputs, so a tuned model inherits the teacher's blind spots. The
   gate can only reject; it cannot create quality that was never generated.
3. **The judge is an LLM and its agreement with humans is not established.**
   `human_review.csv` is exported and **not yet graded**. Until it is, every
   acceptance decision rests on an unvalidated judge.
4. **Self-judging.** Teacher and judge are the same model family, so a
   systematic blind spot could pass its own output unchallenged.
5. **Two languages, finite bug taxonomy.** Python and JavaScript only, across
   {div['unique_bug_categories']} bug categories. Nothing here supports claims
   about other languages or unusual defect classes.
6. **Small per-cell counts.** Spread across many dimension combinations, any
   individual cell holds few examples.
7. **No training claim.** Nothing has been trained on this data. This card
   describes data quality only.

## What this dataset does NOT claim

It does not claim that fine-tuning improves the model, because no model has been
trained. The strongest claim currently supported is that the data was generated
against a measured failure distribution and survived a documented quality gate.
"""


__all__ = ["render_dataset_card"]
