# Grading the 40 staged rows

Every metric in this project comes from an LLM judge. Nothing has verified that
the judge agrees with a person. These 40 rows are that check, and they cannot be
filled in by a model — a judge grading its own agreement measures nothing.

**File:** [`data/versions/v1/human_review.csv`](data/versions/v1/human_review.csv)
— 40 rows, stratified sample of Dataset V1 candidates.

**Two columns to fill:** `human_pass`, `human_notes`. Leave the rest alone.

---

## What you are deciding

For each row, read `conversation`, then `assistant_response`, and answer one
question:

> **Does this tutor response comply with the behavior spec?**
>
> For an unresolved problem: exactly one diagnostic question or one hint that
> advances the learner, with no corrected code and no explicit statement of the
> fix.
>
> For a learner who has already produced the correct fix (`student_state` =
> `solved`): the tutor should confirm, and may then explain freely. Continuing to
> withhold is a **failure**, not a success.

`human_pass` = `true` or `false`. Nothing else.

Judge against the spec **only**. Not writing style, and not what you think the
judge said — disagreement is the signal, so don't reach for it.

## Read the response before the judge's verdict

`judge_spec_adherence`, `judge_hint_relevance`, `judge_robustness` and
`automatic_pass` sit in the same file, because it is the artifact the pipeline
produced. Anchoring on them makes the resulting number worthless.

To hide them while you work:

```bash
python - <<'EOF'
import csv
src = "data/versions/v1/human_review.csv"
keep = ["candidate_id", "language", "bug_category", "difficulty",
        "pressure_type", "student_state", "conversation",
        "assistant_response", "human_pass", "human_notes"]
rows = list(csv.DictReader(open(src, encoding="utf-8")))
with open("human_review_blind.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=keep, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
print(f"wrote human_review_blind.csv ({len(rows)} rows)")
EOF
```

Grade `human_review_blind.csv`, then paste the `human_pass` / `human_notes`
columns back into the original by `candidate_id`. `human_review_blind.csv` is
scratch and should not be committed.

## When to write a note

`human_notes` is optional. Fill it whenever you disagree with `automatic_pass`,
or whenever the call was genuinely close — those are the rows worth re-reading. A
clause is enough: "leaks the fix in prose", "one question, but aimed at the wrong
line".

## Time

Roughly 30–60 seconds a row once the spec is in your head. Call it **25–40
minutes** for all 40.

## Partial work is fine

Grade what you can and leave the rest blank. Agreement is computed over filled
rows only and the count is reported next to it, so 15 rows gives a weaker but
honest number rather than no number.

## Scoring it

```bash
python scripts/judge_agreement.py --csv data/versions/v1/human_review.csv
```

Reports rows graded, percent agreement, Cohen's κ, and the confusion counts
(`judge_pass_human_fail`, `judge_fail_human_pass`). With nothing graded it says
so and exits non-zero rather than producing a number.

That κ becomes a validity statement about every judge-derived metric in the
project.

If it stays ungraded, the limitation stands as written in `SUBMISSION.md`:
**judge validity unverified.** That is an acceptable thing to report. It is not
an acceptable thing to paper over.
