"""How much of the optimization signal teaches release, and how much questioning.

The example counts are 85 solved against 515 not, but example counts are not
what a language model optimises. TRL 1.10.0 defaults `assistant_only_loss` to
False and `completion_only_loss` to None, and the corrected recipe sets neither,
so for a conversational `messages` dataset the loss covers **every token in the
sequence** -- system prompt, learner turns and tutor turn alike. The proportion
of loss-bearing tokens that carry release behaviour is therefore a different
number from 85/600, and it is the one gradient descent actually sees.

Everything here is measured with the run's own tokenizer at the pinned base
revision, over the exact train split the corrected adapter consumed.

    python -m analysis.training_signal
    python -m analysis.training_signal --write
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.solved_state import tutor_profile  # noqa: E402

ANALYSIS_VERSION = "1.0.0"

BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/socratic-v1-n600-bestckpt"
DEFAULT_OUTPUT = REPO_ROOT / "results/solved_state_analysis/training_signal.json"

#: What the recipe leaves unset, and what TRL 1.10.0 therefore does. Recorded
#: here so the token accounting below states its own premise.
LOSS_MASKING = {
    "assistant_only_loss": {"requested": None, "trl_default": False},
    "completion_only_loss": {"requested": None, "trl_default": None},
    "effective": "full_sequence",
    "evidence": (
        "trl/trainer/sft_config.py: assistant_only_loss defaults to False "
        "('loss is computed on the entire sequence'); completion_only_loss "
        "defaults to None (completion-only only for prompt-completion datasets). "
        "trl/trainer/sft_trainer.py:1510-1519 takes the language-modeling branch "
        "for a `messages` dataset and emits only input_ids when "
        "assistant_only_loss is False; the collator at line 465 then defaults "
        "labels to input_ids, so nothing is masked to -100."
    ),
}

QUESTION_MARK = re.compile(r"\?")


def load_tokenizer():
    """The run's own tokenizer, pinned to the base revision it trained against."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    path = hf_hub_download(BASE_MODEL, "tokenizer.json", revision=BASE_REVISION)
    return Tokenizer.from_file(path)


def load_chat_template() -> str:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        BASE_MODEL, "tokenizer_config.json", revision=BASE_REVISION
    )
    return json.loads(Path(path).read_text(encoding="utf-8"))["chat_template"]


def render(messages: Sequence[dict], template: str) -> str:
    """Apply the model's own chat template, as the trainer does."""
    from jinja2 import Environment

    env = Environment(trim_blocks=False, lstrip_blocks=False)
    env.policies["json.dumps_kwargs"] = {"ensure_ascii": False}
    return env.from_string(template).render(
        messages=list(messages), add_generation_prompt=False, tools=None
    )


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def token_count(tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def classify_target(response: str) -> str:
    """What behaviour a tutor target teaches, by the shared detector."""
    profile = tutor_profile(response)
    if profile["confirms"] and not profile["asks_question"]:
        return "confirm_no_question"
    if profile["confirms"]:
        return "confirm_with_question"
    if profile["asks_question"]:
        return "question_only"
    return "neither"


def measure(run_dir: Path | None = None) -> dict[str, Any]:
    run_dir = run_dir or DEFAULT_RUN_DIR
    train = read_jsonl(run_dir / "data/train.jsonl")
    tokenizer = load_tokenizer()
    template = load_chat_template()

    rows: list[dict[str, Any]] = []
    for row in train:
        messages = row["messages"]
        system = messages[0]["content"]
        target = messages[-1]["content"]
        learner = [m for m in messages[1:-1]]

        full_text = render(messages, template)
        system_text = render([messages[0]], template)

        total = token_count(tokenizer, full_text)
        system_tokens = token_count(tokenizer, system_text)
        target_tokens = token_count(tokenizer, target)
        rows.append({
            "id": row["meta"]["id"],
            "solved": bool(row["meta"]["student_has_solved"]),
            "behaviour": classify_target(target),
            "total_tokens": total,
            "system_tokens": system_tokens,
            "target_tokens": target_tokens,
            "context_tokens": max(0, total - system_tokens - target_tokens),
            "learner_turns": len(learner),
        })

    solved = [r for r in rows if r["solved"]]
    unsolved = [r for r in rows if not r["solved"]]

    def total_of(subset, key):
        return sum(r[key] for r in subset)

    all_tokens = total_of(rows, "total_tokens")
    all_target = total_of(rows, "target_tokens")

    behaviour_counts: dict[str, int] = {}
    behaviour_target_tokens: dict[str, int] = {}
    for r in rows:
        behaviour_counts[r["behaviour"]] = behaviour_counts.get(r["behaviour"], 0) + 1
        behaviour_target_tokens[r["behaviour"]] = (
            behaviour_target_tokens.get(r["behaviour"], 0) + r["target_tokens"]
        )

    return {
        "analysis_version": ANALYSIS_VERSION,
        "run_dir": str(run_dir),
        "tokenizer": {"model": BASE_MODEL, "revision": BASE_REVISION},
        "loss_masking": LOSS_MASKING,
        "examples": {
            "train_total": len(rows),
            "solved": len(solved),
            "unsolved": len(unsolved),
            "unsolved_to_solved_ratio": round(len(unsolved) / max(1, len(solved)), 3),
            "solved_share": round(len(solved) / max(1, len(rows)), 4),
        },
        "target_behaviour_counts": behaviour_counts,
        "tokens": {
            "all_loss_bearing": all_tokens,
            "system_prompt": total_of(rows, "system_tokens"),
            "learner_and_context": total_of(rows, "context_tokens"),
            "tutor_target": all_target,
            "system_share_of_loss": round(
                total_of(rows, "system_tokens") / max(1, all_tokens), 4
            ),
            "tutor_target_share_of_loss": round(all_target / max(1, all_tokens), 4),
        },
        "release_signal": {
            "solved_target_tokens": total_of(solved, "target_tokens"),
            "unsolved_target_tokens": total_of(unsolved, "target_tokens"),
            "solved_share_of_target_tokens": round(
                total_of(solved, "target_tokens") / max(1, all_target), 4
            ),
            "solved_share_of_all_loss_tokens": round(
                total_of(solved, "total_tokens") / max(1, all_tokens), 4
            ),
            "solved_target_share_of_all_loss_tokens": round(
                total_of(solved, "target_tokens") / max(1, all_tokens), 4
            ),
        },
        "behaviour_target_tokens": behaviour_target_tokens,
        "per_example": rows,
    }


def render_report(report: dict[str, Any]) -> str:
    ex, tok, rel = report["examples"], report["tokens"], report["release_signal"]
    lines = [
        "",
        f"loss masking            : {report['loss_masking']['effective'].upper()}",
        f"  assistant_only_loss   : requested=None, TRL default=False",
        f"  completion_only_loss  : requested=None, TRL default=None",
        "",
        f"train examples          : {ex['train_total']}",
        f"  solved                : {ex['solved']}",
        f"  unsolved              : {ex['unsolved']}",
        f"  ratio unsolved:solved : {ex['unsolved_to_solved_ratio']}:1",
        f"  solved share          : {ex['solved_share']:.2%}",
        "",
        "target behaviour (example counts):",
    ]
    for name, count in sorted(report["target_behaviour_counts"].items(),
                              key=lambda kv: -kv[1]):
        lines.append(f"  {name:24} {count}")
    lines += [
        "",
        f"loss-bearing tokens     : {tok['all_loss_bearing']:,}",
        f"  system prompt         : {tok['system_prompt']:,} ({tok['system_share_of_loss']:.2%})",
        f"  learner + context     : {tok['learner_and_context']:,}",
        f"  tutor target          : {tok['tutor_target']:,} ({tok['tutor_target_share_of_loss']:.2%})",
        "",
        "how much signal teaches RELEASE:",
        f"  solved target tokens          : {rel['solved_target_tokens']:,}",
        f"  as share of tutor targets     : {rel['solved_share_of_target_tokens']:.2%}",
        f"  as share of ALL loss tokens   : {rel['solved_target_share_of_all_loss_tokens']:.2%}",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    report = measure(run_dir)
    print(render_report(report))

    if args.write:
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        slim = {k: v for k, v in report.items() if k != "per_example"}
        slim["per_example_count"] = len(report["per_example"])
        out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"\nWrote {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
