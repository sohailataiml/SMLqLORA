#!/usr/bin/env python
"""Prove assistant-only masking really happens, before spending GPU time on it.

Arm A changes one flag. A flag that is accepted and then quietly ignored would
produce a run identical to the baseline, and the only symptom would be a null
result three hours later -- the same failure shape as the MVP's pruned
checkpoint, which also looked like a configured setting doing nothing.

So this does not trust the flag. It walks TRL 1.10.0's own code path for a
conversational `messages` dataset, tokenizes real training examples with the
run's own tokenizer at the pinned base revision, and inspects the resulting
labels directly: system and learner tokens must be -100, tutor tokens must not.

Exit codes: 0 masking verified, 1 masking does NOT occur, 2 cannot tell.

    python scripts/verify_assistant_only_loss.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

BASE_MODEL = "Qwen/Qwen3-1.7B"
BASE_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
DEFAULT_RUN_DIR = REPO_ROOT / "outputs/socratic-v1-n600-bestckpt"
DEFAULT_OUTPUT = (
    REPO_ROOT / "results/solved_state_analysis/assistant_loss_verification.json"
)

IGNORE_INDEX = -100


def load_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)


def training_chat_template(tokenizer) -> str | None:
    """The template TRL would actually use when assistant_only_loss is on.

    Qwen's shipped template has no `{% generation %}` markers, so TRL swaps in a
    training variant that does. Replicating that choice here rather than
    assuming it is what makes this a proof about the real path.
    """
    from trl.chat_template_utils import (
        get_training_chat_template,
        has_generation_markers,
    )

    if has_generation_markers(tokenizer.chat_template):
        return None
    return get_training_chat_template(tokenizer)


def tokenize_with_mask(tokenizer, messages: list[dict], chat_template: str | None):
    """Exactly what sft_trainer.py's language-modeling branch does."""
    return tokenizer.apply_chat_template(
        messages,
        chat_template=chat_template,
        return_dict=True,
        return_assistant_tokens_mask=True,
        tokenize=True,
    )


def labels_from_mask(input_ids: list[int], assistant_mask: list[int]) -> list[int]:
    """The collator's effect: everything outside the assistant span is ignored."""
    return [
        token if flag else IGNORE_INDEX
        for token, flag in zip(input_ids, assistant_mask)
    ]


def inspect(tokenizer, row: dict, chat_template: str | None) -> dict[str, Any]:
    messages = row["messages"]
    processed = tokenize_with_mask(tokenizer, messages, chat_template)
    input_ids = list(processed["input_ids"])
    mask = list(processed["assistant_masks"])
    labels = labels_from_mask(input_ids, mask)

    target = messages[-1]["content"]
    supervised_text = tokenizer.decode(
        [t for t, m in zip(input_ids, mask) if m], skip_special_tokens=True
    )
    # The system prompt is the first message; its tokens must all be ignored.
    system_ids = tokenizer.apply_chat_template(
        [messages[0]], chat_template=chat_template, tokenize=True
    )
    system_prefix_masked = all(m == 0 for m in mask[: len(system_ids)])

    return {
        "id": row["meta"]["id"],
        "student_has_solved": bool(row["meta"]["student_has_solved"]),
        "total_tokens": len(input_ids),
        "supervised_tokens": sum(mask),
        "ignored_tokens": len(mask) - sum(mask),
        "system_prefix_fully_masked": system_prefix_masked,
        "any_token_supervised": sum(mask) > 0,
        "labels_outside_assistant_are_ignore_index": all(
            label == IGNORE_INDEX for label, flag in zip(labels, mask) if not flag
        ),
        "labels_inside_assistant_are_real": all(
            label != IGNORE_INDEX for label, flag in zip(labels, mask) if flag
        ),
        "supervised_text_head": supervised_text[:160],
        "target_head": target[:160],
        # Qwen's training template opens the assistant turn with an empty
        # `<think></think>` block, so the supervised span legitimately starts
        # with scaffold rather than with the target's first words. What must
        # hold is that the target text is inside the supervised span and the
        # learner's words are not.
        "target_inside_supervised_span": target.strip()[:60] in supervised_text,
        "learner_text_excluded_from_supervision": (
            messages[-2]["content"].strip()[:60] not in supervised_text
        ),
    }


def measure_arm_a_distribution(tokenizer, train: list[dict],
                               chat_template: str | None) -> dict[str, Any]:
    """Recomputed, not asserted: what the loss looks like under Arm A."""
    solved_supervised = unsolved_supervised = total_all = 0
    for row in train:
        processed = tokenize_with_mask(tokenizer, row["messages"], chat_template)
        mask = list(processed["assistant_masks"])
        total_all += len(mask)
        if row["meta"]["student_has_solved"]:
            solved_supervised += sum(mask)
        else:
            unsolved_supervised += sum(mask)

    supervised = solved_supervised + unsolved_supervised
    return {
        "sequence_tokens_total": total_all,
        "assistant_target_tokens": supervised,
        "solved_assistant_target_tokens": solved_supervised,
        "unsolved_assistant_target_tokens": unsolved_supervised,
        "release_share_of_loss_under_arm_a": round(
            solved_supervised / max(1, supervised), 4
        ),
        "supervised_share_of_sequence": round(supervised / max(1, total_all), 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--full", action="store_true",
                        help="measure the whole split, not just two examples")
    parser.add_argument("--output", default=None)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_DIR
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir

    train_path = run_dir / "data/train.jsonl"
    if not train_path.exists():
        print(f"\nERROR: {train_path} not found.\n", file=sys.stderr)
        return 2

    train = [
        json.loads(line)
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    tokenizer = load_tokenizer()
    chat_template = training_chat_template(tokenizer)

    solved = next(r for r in train if r["meta"]["student_has_solved"])
    unsolved = next(r for r in train if not r["meta"]["student_has_solved"])
    samples = [inspect(tokenizer, solved, chat_template),
               inspect(tokenizer, unsolved, chat_template)]

    print("\nassistant-only loss - label inspection\n")
    print(f"  template swapped in by TRL : {chat_template is not None}")
    for s in samples:
        print(f"\n  {s['id']}  (solved={s['student_has_solved']})")
        print(f"    total tokens               : {s['total_tokens']}")
        print(f"    supervised (assistant)     : {s['supervised_tokens']}")
        print(f"    ignored (-100)             : {s['ignored_tokens']}")
        print(f"    system prefix fully masked : {s['system_prefix_fully_masked']}")
        print(f"    outside-assistant == -100  : {s['labels_outside_assistant_are_ignore_index']}")
        print(f"    inside-assistant real      : {s['labels_inside_assistant_are_real']}")
        print(f"    target inside supervised span    : {s['target_inside_supervised_span']}")
        print(f"    learner text excluded            : {s['learner_text_excluded_from_supervision']}")
        print(f"    supervised head            : {s['supervised_text_head'][:110]!r}")

    verified = all(
        s["system_prefix_fully_masked"]
        and s["any_token_supervised"]
        and s["labels_outside_assistant_are_ignore_index"]
        and s["labels_inside_assistant_are_real"]
        and s["target_inside_supervised_span"]
        and s["learner_text_excluded_from_supervision"]
        for s in samples
    )

    report: dict[str, Any] = {
        "artifact_status": "PRE_TRAINING_VERIFICATION",
        "question": "Does assistant_only_loss actually mask non-assistant tokens?",
        "tokenizer": {"model": BASE_MODEL, "revision": BASE_REVISION},
        "trl_swapped_in_training_template": chat_template is not None,
        "samples": samples,
        "masking_verified": verified,
    }

    if args.full:
        report["arm_a_distribution"] = measure_arm_a_distribution(
            tokenizer, train, chat_template
        )
        d = report["arm_a_distribution"]
        print("\n  Arm A loss distribution over the 540-row split:")
        print(f"    assistant-target tokens       : {d['assistant_target_tokens']:,}")
        print(f"    solved (release)              : {d['solved_assistant_target_tokens']:,}")
        print(f"    unsolved                      : {d['unsolved_assistant_target_tokens']:,}")
        print(f"    release share of loss         : {d['release_share_of_loss_under_arm_a']:.2%}")
        print(f"    supervised share of sequence  : {d['supervised_share_of_sequence']:.2%}")

    print(f"\n  VERDICT: {'MASKING_VERIFIED' if verified else 'MASKING_NOT_OCCURRING'}")
    if not verified:
        print("  Do NOT train. The flag is not producing masked labels.")

    if args.write:
        out = Path(args.output) if args.output else DEFAULT_OUTPUT
        if not out.is_absolute():
            out = REPO_ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        try:
            shown = out.relative_to(REPO_ROOT)
        except ValueError:
            shown = out
        print(f"\n  Wrote {shown}")

    return 0 if verified else 1


if __name__ == "__main__":
    raise SystemExit(main())
