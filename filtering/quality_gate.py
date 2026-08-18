"""The data quality gate.

    candidate -> schema -> static checks -> LLM judge -> dedupe
              -> contamination -> balance -> ACCEPT / REJECT

Every stage can only reject. Rejected examples are written to `data/rejected/`
with their reason, never deleted — the rejection pile is evidence about how
aggressively the dataset was filtered, and it is the first place to look when a
trained model behaves oddly.

Stage order matters and is deliberate:

* cheap deterministic stages run before the expensive judge, so an obvious leak
  never costs an API call;
* dedupe runs after judging so that when two near-duplicates differ in quality,
  the one that survives is one that passed;
* balancing runs last, on already-good data, so the caps trade away quality-
  equivalent examples rather than rescuing bad ones.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from behavior.spec import BehaviorSpec, load_spec
from evaluation.judge import Judge
from evaluation.schemas import Scenario, write_jsonl
from filtering.balance import (
    DEFAULT_CAPS,
    balance,
    conversation_length_distribution,
    distribution,
)
from filtering.dedupe import check_contamination, deduplicate, drop_contaminated
from filtering.judge import UNJUDGED, judge_candidates
from filtering.static_checks import static_screen
from generation.schemas import GeneratedExample

STAGES = (
    "schema",
    "static_checks",
    "llm_judge",
    "dedupe",
    "contamination",
    "balance",
)


@dataclass
class GateReport:
    """Everything a dataset version needs to be understood and audited."""

    dataset_version: str
    candidate_count: int = 0
    accepted_count: int = 0
    rejected_count: int = 0
    acceptance_rate: float = 0.0
    #: Candidates the judge could not be reached for. Excluded from both
    #: `accepted` and `rejected`, and from the acceptance-rate denominator,
    #: because an outage describes the account and not the data.
    unjudged_count: int = 0
    judged_count: int = 0
    complete: bool = True

    rejections_by_reason: dict[str, int] = field(default_factory=dict)
    rejections_by_stage: dict[str, int] = field(default_factory=dict)

    language_distribution: dict[str, int] = field(default_factory=dict)
    bug_category_distribution: dict[str, int] = field(default_factory=dict)
    pressure_distribution: dict[str, int] = field(default_factory=dict)
    difficulty_distribution: dict[str, int] = field(default_factory=dict)
    conversation_length_distribution: dict[str, int] = field(default_factory=dict)

    balance_caps: dict[str, int] = field(default_factory=dict)
    contamination_summary: str = ""
    dataset_hash: str = ""
    teacher: dict[str, str] = field(default_factory=dict)
    judge: dict[str, str] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)


@dataclass
class GateOutcome:
    accepted: list[GeneratedExample] = field(default_factory=list)
    rejected: list[GeneratedExample] = field(default_factory=list)
    #: Never judged because the judge was unreachable. Retry these; do not
    #: count them as rejections and do not freeze a dataset while any remain.
    unjudged: list[GeneratedExample] = field(default_factory=list)
    report: GateReport | None = None


def dataset_hash(examples: Sequence[GeneratedExample]) -> str:
    """Order-independent hash of an accepted set."""
    digest = hashlib.sha256()
    for content in sorted(e.content_hash() for e in examples):
        digest.update(content.encode("ascii"))
    return digest.hexdigest()


def _reject(
    example: GeneratedExample, codes: Sequence[str], stage: str, notes: str = ""
) -> GeneratedExample:
    return example.model_copy(
        update={
            "accepted": False,
            "rejection_codes": tuple(dict.fromkeys(codes)) or ("LOW_QUALITY",),
            "gate_notes": " | ".join(x for x in (example.gate_notes, f"stage={stage}", notes) if x),
        }
    )


def run_quality_gate(
    candidates: Sequence[GeneratedExample],
    judge: Judge,
    *,
    spec: BehaviorSpec | None = None,
    eval_scenarios: Sequence[Scenario] = (),
    dataset_version: str = "v1",
    balance_caps: dict[str, float] | None = None,
    max_workers: int = 4,
    teacher_description: dict[str, str] | None = None,
    on_progress: Callable[[str, int, int], None] | None = None,
    notes: str = "",
) -> GateOutcome:
    """Run every stage and return accepted, rejected and the dataset report."""
    spec = spec or load_spec()
    rejected: list[GeneratedExample] = []
    stage_counts: Counter[str] = Counter()

    # ---- Stage 1: schema ---------------------------------------------------
    # Candidates arriving here are already `GeneratedExample` instances, so the
    # schema held. Anything that failed parsing was recorded at generation time.
    survivors = list(candidates)
    if on_progress:
        on_progress("schema", len(survivors), len(candidates))

    # ---- Stage 2: static checks -------------------------------------------
    after_static: list[GeneratedExample] = []
    for example in survivors:
        ok, codes, behavior, integrity = static_screen(example, spec)
        example = example.model_copy(update={"deterministic": behavior})
        if ok:
            after_static.append(example)
        else:
            rejected.append(
                _reject(example, codes, "static_checks", "; ".join(integrity.notes))
            )
            stage_counts["static_checks"] += 1
    if on_progress:
        on_progress("static_checks", len(after_static), len(survivors))

    # ---- Stage 3: LLM judge ------------------------------------------------
    after_judge: list[GeneratedExample] = []
    unjudged: list[GeneratedExample] = []
    judged = judge_candidates(after_static, judge, spec, max_workers=max_workers)
    for example, codes in judged:
        if codes == (UNJUDGED,):
            unjudged.append(example)
            stage_counts["judge_unavailable"] += 1
        elif codes:
            rejected.append(_reject(example, codes, "llm_judge"))
            stage_counts["llm_judge"] += 1
        else:
            after_judge.append(example)
    if on_progress:
        on_progress("llm_judge", len(after_judge), len(after_static))

    # ---- Stage 4: dedupe ---------------------------------------------------
    dedupe_result = deduplicate(after_judge)
    rejected.extend(dedupe_result.duplicates)
    stage_counts["dedupe"] += dedupe_result.removed
    if on_progress:
        on_progress("dedupe", len(dedupe_result.kept), len(after_judge))

    # ---- Stage 5: contamination -------------------------------------------
    contamination = check_contamination(dedupe_result.kept, eval_scenarios)
    clean, dirty = drop_contaminated(dedupe_result.kept, contamination)
    rejected.extend(dirty)
    stage_counts["contamination"] += len(dirty)
    if on_progress:
        on_progress("contamination", len(clean), len(dedupe_result.kept))

    # ---- Stage 6: balance --------------------------------------------------
    balance_result = balance(clean, caps=balance_caps or DEFAULT_CAPS)
    rejected.extend(balance_result.dropped)
    stage_counts["balance"] += balance_result.removed
    accepted = [
        e.model_copy(update={"accepted": True}) for e in balance_result.kept
    ]
    if on_progress:
        on_progress("balance", len(accepted), len(clean))

    # ---- Report ------------------------------------------------------------
    reasons: Counter[str] = Counter()
    for example in rejected:
        for code in example.rejection_codes:
            reasons[code] += 1

    dists = distribution(accepted)
    report = GateReport(
        dataset_version=dataset_version,
        candidate_count=len(candidates),
        accepted_count=len(accepted),
        rejected_count=len(rejected),
        unjudged_count=len(unjudged),
        judged_count=len(candidates) - len(unjudged),
        complete=not unjudged,
        # Denominator excludes candidates the judge never saw, so an outage
        # cannot depress the reported acceptance rate.
        acceptance_rate=round(len(accepted) / (len(candidates) - len(unjudged)), 4)
        if (len(candidates) - len(unjudged)) else 0.0,
        rejections_by_reason=dict(reasons.most_common()),
        rejections_by_stage=dict(stage_counts.most_common()),
        language_distribution=dists.get("language", {}),
        bug_category_distribution=dists.get("bug_category", {}),
        pressure_distribution=dists.get("pressure_type", {}),
        difficulty_distribution=dists.get("difficulty", {}),
        conversation_length_distribution=conversation_length_distribution(accepted),
        balance_caps=balance_result.caps,
        contamination_summary=contamination.summary(),
        dataset_hash=dataset_hash(accepted),
        teacher=teacher_description or {},
        judge=judge.describe(),
        thresholds={
            "min_judge_spec_adherence": spec.gates.quality_gate.min_judge_spec_adherence,
            "min_judge_hint_relevance": spec.gates.quality_gate.min_judge_hint_relevance,
            "min_judge_robustness": spec.gates.quality_gate.min_judge_robustness,
            "balance_caps_share": balance_caps or DEFAULT_CAPS,
        },
        notes=notes,
    )

    return GateOutcome(accepted=accepted, rejected=rejected,
                       unjudged=unjudged, report=report)


# =============================================================================
# Persistence
# =============================================================================


def write_dataset_version(
    outcome: GateOutcome,
    *,
    repo_root: Path,
    dataset_version: str,
) -> dict[str, Path]:
    """Write accepted, rejected and the report for one dataset version."""
    version_dir = repo_root / "data" / "versions" / dataset_version
    version_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "accepted": repo_root / "data" / "accepted" / f"{dataset_version}.jsonl",
        "rejected": repo_root / "data" / "rejected" / f"{dataset_version}.jsonl",
        "version_accepted": version_dir / "accepted.jsonl",
        "version_rejected": version_dir / "rejected.jsonl",
        "version_unjudged": version_dir / "unjudged.jsonl",
        "report_json": version_dir / "report.json",
        "report_md": version_dir / "report.md",
    }

    write_jsonl(paths["accepted"], outcome.accepted)
    write_jsonl(paths["rejected"], outcome.rejected)
    write_jsonl(paths["version_accepted"], outcome.accepted)
    write_jsonl(paths["version_rejected"], outcome.rejected)
    # Written even when empty, so "no unjudged candidates" is an assertion in
    # the artifacts rather than an absent file open to interpretation.
    write_jsonl(paths["version_unjudged"], outcome.unjudged)

    assert outcome.report is not None
    paths["report_json"].write_text(
        json.dumps(outcome.report.to_dict(), indent=2) + "\n", encoding="utf-8"
    )
    paths["report_md"].write_text(render_report(outcome.report), encoding="utf-8")
    return paths


def render_report(report: GateReport) -> str:
    from ablations.reporting import markdown_table

    def table(title: str, mapping: dict[str, Any], key: str = "value") -> list[str]:
        return [
            f"### {title}",
            "",
            markdown_table(
                [{"bucket": k, "count": v} for k, v in mapping.items()],
                ["bucket", "count"],
            ),
            "",
        ]

    lines = [
        f"# Dataset `{report.dataset_version}`",
        "",
        f"Teacher: `{report.teacher.get('model', 'unknown')}`  ",
        f"Judge: `{report.judge.get('judge_model', 'unknown')}`  ",
        f"Dataset hash: `{report.dataset_hash[:16]}`",
        "",
        "## Funnel",
        "",
        markdown_table(
            [
                {"metric": "candidates", "count": report.candidate_count},
                {"metric": "accepted", "count": report.accepted_count},
                {"metric": "rejected", "count": report.rejected_count},
                {
                    "metric": "acceptance rate",
                    "count": f"{report.acceptance_rate:.1%}",
                },
            ],
            ["metric", "count"],
        ),
        "",
        "## Rejections by stage",
        "",
        markdown_table(
            [{"stage": k, "rejected": v} for k, v in report.rejections_by_stage.items()],
            ["stage", "rejected"],
        ),
        "",
        "## Rejections by reason",
        "",
        markdown_table(
            [{"reason": k, "count": v} for k, v in report.rejections_by_reason.items()],
            ["reason", "count"],
        ),
        "",
        "## Accepted distribution",
        "",
    ]
    lines += table("Language", report.language_distribution)
    lines += table("Bug category", report.bug_category_distribution)
    lines += table("Pressure type", report.pressure_distribution)
    lines += table("Difficulty", report.difficulty_distribution)
    lines += table("Conversation length (learner turns)",
                   report.conversation_length_distribution)

    lines += [
        "## Contamination",
        "",
        report.contamination_summary,
        "",
        "## Thresholds applied",
        "",
        "```json",
        json.dumps(report.thresholds, indent=2),
        "```",
    ]
    if report.notes:
        lines += ["", "## Notes", "", report.notes]
    return "\n".join(lines) + "\n"


__all__ = [
    "GateOutcome",
    "GateReport",
    "STAGES",
    "dataset_hash",
    "render_report",
    "run_quality_gate",
    "write_dataset_version",
]
