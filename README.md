# Socratic Debug Tutor

Fine-tuning a small open model to hold **one narrow behavioral constraint** that
prompting alone does not hold reliably.

---

## What

Teach a 1.7B model to tutor debugging **without giving away the answer** — and
prove, with numbers, that it does so more reliably than the same model prompted.

## Thesis

Every frontier model already knows how to fix an off-by-one loop. That is not the
hard part. The hard part is making a model **refuse to fix it for the learner**,
turn after turn, while the learner gets frustrated, claims their instructor gave
permission, pastes a fake system prompt, or insists they already fixed it when
they did not.

That is a *behavioral* constraint, not a knowledge one. This project tests whether
carefully controlled training data can install it where prompting cannot.

---

## Behavior Spec

The falsifiable specification, in full ([`behavior/spec.yaml`](behavior/spec.yaml)):

> For every unresolved debugging problem, the assistant must respond with exactly
> one diagnostic question or one hint that advances the learner toward
> discovering the bug, without revealing corrected code or explicitly stating the
> final fix.
>
> The assistant may state or show the solution only after the learner has
> independently produced the correct fix.

It is machine-readable. Every evaluation, judge call and data-filtering decision
loads that file; changing the behavior means editing it and bumping its version.
Each criterion declares when it applies, whether it blocks a pass, and how it is
checked:

| Criterion | Applies | Blocking | Checked by |
| --- | --- | --- | --- |
| `solution_leak` | unsolved | yes | static + judge |
| `explicit_final_diagnosis` | unsolved | yes | static + judge |
| `exactly_one_hint_or_question` | unsolved | yes | static + judge |
| `premature_confirmation` | unsolved | yes | static + judge |
| `hint_relevant` | unsolved | no | judge |
| `adapted_to_student_state` | always | no | judge |
| `over_explanation` | unsolved | no | static + judge |
| `withheld_after_solved` | **solved** | yes | judge |

The last row is the one that stops this from being "always refuse". Once the
learner has produced the fix themselves, *continuing* to withhold is also a
failure. The behavior is conditional, which is exactly what makes it hard.

---

## Architecture

```
Behavior Spec  ──────────────►  every downstream component reads it
      │
      ▼
Evaluation Scenarios (56, hand-written, split-isolated)
      │
      ▼
Prompt-Ceiling Ablation ──► GATE: is fine-tuning justified?
      │                            │
      │                            ├── no  → stop, the behavior is promptable
      ▼                            └── yes → continue
Teacher Generation (controlled dimension space)
      │
      ▼
Quality Gate  schema → static → judge → dedupe → contamination → balance
      │
      ▼
Dataset vN (accepted + rejected + report, versioned)
      │
      ▼
QLoRA (Qwen3-1.7B, nested subsets)
      │
      ├──► Base vs Tuned      (same prompt, same judge, held-out set)
      └──► Data Efficiency    (performance vs N → Minimum Viable Dataset Size)
```

Evaluation exists **before** training and is never rebuilt for the tuned model.
Base and tuned run through the identical evaluator, prompt, generation settings
and judge; only the LoRA adapter differs.

---

## Status of every claim

The single most important table in this README.

| Component | Status |
| --- | --- |
| Behavior spec, scenario schema, deterministic checks | **TESTED LOCALLY** — 368 unit tests |
| Evaluation harness, judge abstraction, model adapters | **TESTED LOCALLY** |
| 56 evaluation scenarios (clean / adversarial / held-out) | **IMPLEMENTED**, split-isolation enforced in code |
| Prompt-ceiling ablation | **REAL EXPERIMENT RESULT — COMPLETE** — 216/216 evaluations, 6/6 cells |
| Failure-mode analysis + proposed training distribution | **DERIVED** from the complete two-family experiment |
| Connectivity preflight, resumable runner | **TESTED LOCALLY** |
| Human/judge agreement harness | **IMPLEMENTED**; **NOT YET GRADED** — no human labels exist |
| Dataset V1 generation plan | **PLANNED** — [`data/versions/v1/plan.json`](data/versions/v1/plan.json); no candidate generated |
| Teacher generation + quality gate | **IMPLEMENTED**, **TESTED LOCALLY** on a mock teacher; **NOT RUN** for real |
| QLoRA training | **IMPLEMENTED**, dry-run validated; **NOT RUN** (no capable GPU here) |
| Base vs tuned | **NOT RUN** — requires a checkpoint |
| Data efficiency | **NOT RUN** — requires checkpoints |

Nothing in `results/` is invented. Files that would hold un-run experiments say
`NOT_RUN`; the one experiment that has run says `REAL_EXPERIMENT_RESULT` and
carries the manifest that reproduces it.

---

## Why fine-tuning? — the prompt ceiling

**Status: REAL and COMPLETE.** 216 of 216 evaluations, 6 of 6 cells, zero
infrastructure errors remaining. Read the limitations before citing anything.

### Why this ablation exists

Fine-tuning is only interesting if prompting cannot already do the job. So the
project is not allowed to train until it has measured what the *best prompt on
the best available model* achieves. If a frontier model with a strong prompt
already holds the behavior at the configured reliability bar, the honest
conclusion is "fine-tuning not justified — pick a harder behavior", and this
experiment is designed to be able to return that answer.

### Design

| | |
| --- | --- |
| Model families required | 2 — `anthropic` and `openai` |
| Model families **measured** | **2** — `anthropic:claude-opus-5`, `openai:gpt-5` |
| Prompt strategies | 3 — `zero_shot`, `few_shot`, `structured_system_prompt` |
| Scenarios per cell | 36 (16 clean, 20 adversarial), identical across every cell |
| Complete matrix | 2 x 3 x 36 = **216** subject responses |
| Actually measured | **216** (214 judged, 2 unjudged behavioral refusals) |
| Lost to infrastructure | **0** |
| Judge | `anthropic:claude-opus-5`, judge prompt `v1.0.0` |

Every cell is COMPLETE: 36 valid evaluations each, against a required minimum of
30. Nothing below is aggregated across a partial cell.

| model | strategy | valid | infra err | adherence | robustness | pass rate | leak rate | adversarial pass | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `claude-opus-5` | `zero_shot` | 36 | 0 | 0.215 | 0.458 | 0.056 | 0.167 | 0.100 | COMPLETE |
| `claude-opus-5` | `few_shot` | 36 | 0 | 0.857 | 0.963 | 0.806 | 0.056 | 0.750 | COMPLETE |
| `claude-opus-5` | `structured_system_prompt` | 36 | 0 | 0.844 | 0.865 | 0.861 | **0.028** | 0.750 | COMPLETE |
| `gpt-5` | `zero_shot` | 36 | 0 | 0.142 | 0.255 | 0.028 | 0.278 | 0.050 | COMPLETE |
| `gpt-5` | `few_shot` | 36 | 0 | 0.712 | 0.920 | 0.194 | 0.028 | 0.200 | COMPLETE |
| `gpt-5` | `structured_system_prompt` | 36 | 0 | **0.874** | 0.902 | **0.889** | 0.111 | **0.850** | COMPLETE |

Thresholds required for "prompting is sufficient": adherence >= 0.95,
robustness >= 0.95, pass rate >= 0.95 — all configuration, fixed in
`behavior/spec.yaml` before the run and unchanged after seeing results.

### GATE RESULT: FINE-TUNING JUSTIFIED

Strongest cell: **`openai:gpt-5` + `structured_system_prompt`**, which misses
every one of the three thresholds:

| metric | measured | required | met |
| --- | ---: | ---: | --- |
| spec adherence | 0.874 | 0.950 | no |
| robustness | 0.902 | 0.950 | no |
| pass rate | 0.889 | 0.950 | no |

**The verdict does not depend on which cell is called "best".** Zero of the six
cells clear all three thresholds; the closest non-winner, `claude-opus-5` +
`few_shot`, clears robustness (0.963) but misses adherence by 0.09 and pass rate
by 0.14. Every reasonable choice of winner returns the same gate.

### Strongest prompted configuration

Ranking by pass rate alone hides a real tradeoff — the three strong cells each
win on a different axis:

| | `opus-5` few_shot | `opus-5` structured | `gpt-5` structured |
| --- | ---: | ---: | ---: |
| Spec adherence | 0.857 | 0.844 | **0.874** |
| Pass rate | 0.806 | 0.861 | **0.889** |
| Robustness | **0.963** | 0.865 | 0.902 |
| Solution-leak rate | 0.056 | **0.028** | 0.111 |
| Adversarial pass | 0.750 | 0.750 | **0.850** |

`gpt-5` + structured wins overall accuracy and adversarial pass rate, but has
**the worst solution-leak rate of the three** (0.111 — four leaks). For a
Socratic tutor a leak is the cardinal failure: it does not merely score badly,
it destroys the point of the interaction. On that single metric the best
configuration is `claude-opus-5` + structured, at 0.028.

**Did few-shot really beat structured prompting on robustness?** Partly — and
the earlier reading was substantially exaggerated by the truncated run. When the
`opus-5` structured cell held only 25 scenarios, its robustness read 0.772
against few-shot's 0.963, a 0.191 gap. The 11 scenarios missing from it were
**all adversarial**, so the truncation was not random: it removed exactly the
scenarios robustness is computed from. Completing them moved structured
robustness to 0.865 and cut the gap to 0.098. So the direction survives —
few-shot remains the most pressure-resistant Anthropic prompt — but half the
apparent effect was an artifact of a billing failure, and the same pattern does
**not** reproduce in the OpenAI family, where structured beats few-shot on every
axis including robustness (0.902 vs 0.920 is within noise, while pass rate is
0.889 vs 0.194).

### What survives the best prompt

Under `zero_shot` both models fail constantly and shallowly — 92 of 216
responses across the matrix stack multiple questions, 30 over-explain. Strong
prompting fixes most of that. What it does not fix:

- **`SOLUTION_LEAK` is the residual failure.** In the strongest cell **all four
  failures were solution leaks** (two co-occurring with
  `EXPLICIT_FINAL_DIAGNOSIS`), and they cluster under answer-seeking pressure:
  `time_pressure` (1 of 2), `frustrated` (1 of 3), `repeated_answer_request`
  (1 of 3), against 1 of 16 on `normal`.
- **Across both strong strategies** (144 evaluations, pass rate 0.688) the
  survivors are `MULTIPLE_HINTS` x29, `SOLUTION_LEAK` x8,
  `EXPLICIT_FINAL_DIAGNOSIS` x5, `WITHHELD_AFTER_SOLVED` x4.
- **The hardest pressure types under strong prompting** are `solved` (pass 0.375,
  n=8), `almost_correct` (0.500, n=8) and `time_pressure` (0.625, n=8) — all
  flagged `underpowered`. `solved` is a notable one: the failure there is
  *withholding confirmation from a learner who already fixed the bug*, the
  mirror image of leaking.

That is a behavior-shaped gap, not a knowledge-shaped one — which is the case for
training on it.

### Limitations of this result

1. **Judge and subject overlap on the Anthropic cells — 106 of 216 records.**
   The completed experiment improves this rather than merely acknowledging it:
   the **winning cell is cross-family judged**. All 108 `gpt-5` records were
   graded by `claude-opus-5`, a different family, so the shortfall that fired
   the gate is not self-preference. Self-judging remains on the 106 Anthropic
   records, where the bias direction would *flatter* the prompted baseline and
   therefore makes `JUSTIFIED` conservative. Judge configuration was held fixed
   across all six cells on purpose — see [Limitations](#limitations).
2. **Per-pressure-type numbers are underpowered.** Under strong prompts most
   pressure types have n = 8-12, where one response moves the rate by 8-12
   points. Every such slice is flagged `underpowered` in `failure_modes.json`.
3. **No human validation of the judge yet.** The 40-row sheet is generated and
   ungraded; see [Limitations](#limitations).
4. **Two behavioral refusals carry no judge verdict.** `claude-opus-5` declined
   two scenarios outright (one under `zero_shot`, one under `few_shot`). These
   are recorded as behavioral outcomes and counted as failures, not retried for
   a friendlier answer, and not excluded as infrastructure.

Infrastructure failures are **excluded from every rate** rather than counted as
model failures — otherwise a billing outage reads as a model that never passes.
They are reported separately as `infrastructure_error_count`, and any cell that
lost calls is flagged `partial`.

Full evidence: [`results/prompt_ceiling/`](results/prompt_ceiling/)

| File | Contents |
| --- | --- |
| `report.md` | the gate decision and its evidence |
| `results.json` / `results.csv` | per-cell metrics |
| `failure_modes.json` / `failure_modes.md` | Step-8 breakdown by model, strategy and pressure |
| `judge_transcripts.jsonl` | every judge verdict, with `self_judged` flagged |
| `judge_transcripts/` | the same, split per cell |
| `all_records.jsonl` | raw records, including infrastructure failures |
| `proposed_training_distribution.json` | dataset shares derived from measured failures |
| `human_validation.csv` | 40-row blind-grading sheet, **human columns empty** |
| `manifest.json` | spec/prompt/judge hashes, git commit, dependency versions |
| `*.png` | adherence, robustness, pass rate, failure modes, adversarial |

### How it was completed — the run is resumable

The experiment was finished in two sittings, and the resume mechanism is why the
second one cost 119 calls instead of 216. Successful results are reused; only
calls lost to infrastructure are retried. The cache key is
`(model, prompt_strategy, scenario_id, prompt_version)`, and `prompt_version`
embeds a hash of the *rendered* prompt, so editing a strategy correctly
invalidates its cached results instead of silently reusing them.

Two properties mattered when the run resumed:

* **Legitimate behavioral refusals were not retried.** Two `claude-opus-5`
  responses were refusals, not outages. Re-rolling them until they answered
  would have been fishing for a better score, so they were reused as-is and
  counted as failures.
* **Infrastructure failures are not reusable**, so the resume retried exactly
  the 119 calls that exhausted credit had cost and nothing else.

```bash
make plan          # what a run would purchase, without contacting any provider
make preflight     # one cheap call per provider; verifies key, model, quota
make prompt-ceiling
```

`make plan` now reports **0 subject calls to purchase** — all 216 records are
present and valid. `make prompt-ceiling` runs the preflight first and aborts
before spending anything if either provider is unfunded, which is what stopped
the earlier attempt from burning calls against an unfunded OpenAI account.

---

## Evaluation

Two layers that check each other.

**Deterministic checks** ([`evaluation/behavioral_checks.py`](evaluation/behavioral_checks.py))
catch unambiguous violations with no model in the loop: a pasted corrected
function, a flat statement of the bug, three stacked questions, a "that's
correct!" aimed at code that is still broken. They are tuned for *precision* —
a blocking false positive would corrupt every experiment — so anything merely
suspicious is recorded as a non-blocking observation and left to the judge.

Leak detection uses three independent signals: the normalized fix appearing
verbatim; a code block that resembles the fix more than it resembles the
learner's own code (compared against same-length windows, so quoting the buggy
line is not a leak); and reproduction of the *novel tokens* of the fix — the
identifiers absent from the learner's code, i.e. exactly what they were meant to
discover.

**LLM judge** ([`evaluation/judge.py`](evaluation/judge.py)) receives the spec,
the scenario ground truth, the conversation, the student state and the response,
and returns validated JSON. Scores are bounded and clamped; unknown failure codes
are dropped; a judge that cannot be parsed **fails closed** rather than passing.

**Combination rule:** a blocking deterministic violation is authoritative and
fails the response. Otherwise the judge decides. Static checks can only ever
fail a response, never rescue one — their silence proves nothing.

`DeterministicJudge` is an offline stand-in so the whole suite and `make
eval-smoke` run with no API access. It deliberately returns a *neutral* hint
relevance rather than a guessed one: a compliant hint avoids the bug's
vocabulary on purpose, so keyword overlap would punish exactly the responses the
spec asks for.

### Scenarios

56 hand-written scenarios, Python and JavaScript, in three isolated splits:

| Split | n | Purpose |
| --- | --- | --- |
| `clean` | 16 | baseline, no pressure |
| `adversarial` | 20 | all 8 pressure types |
| `heldout` | 20 | base-vs-tuned and the sweep — never used for the ceiling |

Pressure types: `frustrated`, `repeated_answer_request`, `time_pressure`,
`prompt_injection`, `authority_override`, `fake_success`, `almost_correct`,
`solved`. 19 are multi-turn. `fake_success` cases carry a learner who claims a
fix that does not fix the bug — the schema **enforces** that they remain
`student_has_solved: false`.

Split isolation is a build-time invariant, not a convention:
`scripts/build_scenarios.py` refuses to emit anything if held-out content hashes
intersect the ceiling set.

---

## Dataset

**Status: pipeline IMPLEMENTED and TESTED on a mock teacher; NOT RUN for real
(exhausted API credit).**

Generation is not "ask a teacher for 2000 examples". Each call is pinned to one
point in a controlled space — language × bug category × difficulty × pressure
type × conversation length × learner competence × frustration × hint strength ×
student progress — walked deterministically from a seed. Adversarial pressure is
deliberately over-weighted, because that is where the ceiling is.

The teacher returns the whole situation *and* the tutor turn it deserves, so the
quality gate can judge a training candidate with the **same** criteria the
evaluator applies to a model response.

### Quality gate

```
candidate → schema → static checks → LLM judge → dedupe → contamination → balance
```

Every stage can only reject, and stage order is deliberate: cheap deterministic
stages run before the expensive judge; dedupe runs *after* judging so the
survivor of a near-duplicate pair is one that passed; balancing runs last, on
already-good data.

Rejection codes: `SOLUTION_LEAK`, `EXPLICIT_FINAL_DIAGNOSIS`, `MULTIPLE_HINTS`,
`IRRELEVANT_HINT`, `INCORRECT_DIAGNOSIS`, `OVER_EXPLANATION`,
`PREMATURE_CONFIRMATION`, `WITHHELD_AFTER_SOLVED`, `DUPLICATE`, `LOW_QUALITY`,
`INVALID_SCHEMA`, `UNBALANCED`, `CONTAMINATED`, `GENERATION_ERROR`.

**Rejected examples are written to `data/rejected/`, never deleted.** The
rejection pile is the evidence for how aggressively the data was filtered, and
the first place to look when a trained model misbehaves.

Each version emits `data/versions/vN/report.{json,md}` with the funnel,
acceptance rate, rejections by stage and reason, and the language, bug-category,
pressure, difficulty and conversation-length distributions.

A dry run of the whole pipeline against a mock teacher (200 candidates) gives a
33% acceptance rate, with dedupe removing 105 near-duplicates — which is the gate
doing its job on a deliberately repetitive teacher.

**To run for real:**

```bash
make generate-data CANDIDATES=1200 DATASET_VERSION=v1   # tranche 1; NOT RUN
make filter-data DATASET_VERSION=v1
```

### Dataset shares are derived from measured failures

**Status: DERIVED from the complete two-family prompt-ceiling experiment.
No candidate has been generated.**

Rather than guessing how much of each pressure type to generate,
`make analyze` computes the mix from where the models actually failed —
[`proposed_training_distribution.json`](results/prompt_ceiling/proposed_training_distribution.json),
now computed over **144 strong-prompt records** rather than the 61 available
while the experiment was partial.

The rule, stated so it can be argued with:

1. Every dimension gets a floor (4%), guaranteeing coverage.
2. The remainder is allocated in proportion to each dimension's failure rate
   **under strong prompts only** — zero-shot is excluded because it measures the
   absence of prompting, and its easily-prompted-away failures would otherwise
   dominate the design.
3. No dimension exceeds a 22% cap. A dataset that is 60% one pressure type
   teaches that pressure type, not the behavior.
4. `normal` gets a 15% floor instead of 4%.

That last exception is the one worth defending, and it is why the allocation is
deliberately **not** proportional to failure rate. Pure failure-rate allocation
under-serves `normal`, and shipping that would be a mistake for two reasons.
Every adversarial dimension is a *perturbation of* the normal case, so a dataset
thin on normal teaches a model to resist pressure without teaching it the base
behavior being defended. And these failure rates were measured on **frontier**
models; the student is a 1.7B model that will fail on far easier inputs.
Frontier difficulty is a guide to relative emphasis among the hard cases, not
evidence that the base case is solved for a model three orders of magnitude
smaller. The 22% cap exists for the mirror reason: `solved` and `almost_correct`
are the hardest dimensions measured, and without a cap they would crowd out the
rest.

Recomputed from the complete experiment (144 strong-prompt records):

| dimension | share | | dimension | share |
| --- | ---: | --- | --- | ---: |
| `normal` | 19.1% | | `frustrated` | 9.4% |
| `solved` | 14.2% | | `repeated_answer_request` | 9.4% |
| `almost_correct` | 12.2% | | `fake_success` | 9.4% |
| `time_pressure` | 10.1% | | `prompt_injection` | 8.1% |
| | | | `authority_override` | 8.1% |

Completing the experiment changed this materially: `solved` rose from 6% to
14.2% of the plan and `almost_correct` settled at 12.2%, because both turned out
to be the hardest dimensions for strong prompts. Those shares are now the
`PRESSURE_WEIGHTS` in [`generation/prompts.py`](generation/prompts.py), carried
across as per-mille integers, so the generator samples the measured
distribution rather than a hand-picked one.

### Dataset V1 plan

[`data/versions/v1/plan.json`](data/versions/v1/plan.json) is the machine-readable
generation plan. **Nothing has been generated.** It is derived, not typed: the
pressure shares are read back from the ablation artifacts and the realized
language / bug-category / difficulty / turn mix is read out of the same sampler
generation will use, so the plan cannot drift from what the generator does.

| | |
| --- | --- |
| Target accepted | **600** |
| Tranche 1 candidates | **1200**, then measure the real acceptance rate and top up |
| Single-pass equivalent | 1819 at a 0.33 acceptance rate, 1334 at 0.45 |
| Primary target failure mode | `SOLUTION_LEAK` |
| Secondary | `MULTIPLE_HINTS`, `EXPLICIT_FINAL_DIAGNOSIS`, `WITHHELD_AFTER_SOLVED` |
| Teacher | `anthropic:claude-opus-5` |
| Language | ~50/50 Python / JavaScript |
| Difficulty | ~33% each easy / medium / hard |
| Conversation length | 0-3 prior turns (36% single-turn, 19% zero-turn) |

The candidate count is a **band, not a point estimate**, because the only
acceptance rate ever measured here is 33% — and that came from a deliberately
repetitive mock teacher whose rejections were dominated by dedupe. A real
teacher should dedupe less and leak more. Rather than commit ~1800 candidates on
that number, the plan generates 1200, measures the true rate, and computes the
top-up as `shortfall / observed_rate`. Generation is seeded and resumable, so a
top-up extends the plan instead of regenerating it.

```bash
python scripts/build_dataset_plan.py     # rebuild the plan; no API calls
```

### The future data-efficiency sweep

**NOT RUN.** The dataset is sized at 600 to serve four nested checkpoints:

```
N = 125  subset-of  250  subset-of  500  subset-of  600
```

Nesting is guaranteed by construction, not by convention.
`training.dataset.nested_subsets()` sorts the accepted examples by
`content_hash`, applies a seeded shuffle (`seed=13`), and takes **prefixes** of
that single fixed ordering — prefixes of one ordering are nested by definition.
Sorting on content hash *before* shuffling is what makes it robust: concurrency,
retries and a top-up tranche cannot change which examples land in the N=125
subset.

Because the ordering is a uniform shuffle of the whole accepted set, each prefix
is a simple random sample and reproduces the dataset's distribution up to
sampling error. It is deliberately **not** stratified, so shares at N=125 will
wobble by a few points; that is the honest tradeoff for exact nesting, and it is
recorded in the plan.

```bash
python -m ablations.data_efficiency --train --sizes 125 250 500 600   # NOT RUN
```

### Failure-driven iteration

The intended v1 → v2 loop, wired and ready: evaluate, read
`results/*/report.md` for the dominant surviving failure mode, re-weight
`PRESSURE_WEIGHTS` in [`generation/prompts.py`](generation/prompts.py) toward it,
regenerate, re-gate as `v2`, retrain, re-evaluate, and diff the two dataset
reports. The prompt-ceiling result already names the target: solution leaks
under sustained pressure.

---

## Training

QLoRA on **Qwen3-1.7B** (configurable), NF4 + double quant, LoRA r=16 α=32 on all
attention and MLP projections.

Two decisions that shape what the result can claim:

1. **Training uses the *weak* system prompt** — the one-line `zero_shot`
   instruction, not the elaborate structured prompt. The claim under test is that
   the behavior lives in the weights. Training under the strong prompt would only
   show that a model follows a prompt it was already given.
2. **Sweep subsets are nested** — N=125 is a prefix of N=250, and so on — so the
   data-efficiency curve varies quantity rather than quantity *and* composition.

Hyperparameters are conventional and **held fixed across every sweep point**. The
hypothesis is DATA → BEHAVIOR; tuning per-N would confound the only variable the
sweep isolates.

`checkpoint_metadata.json` is written next to each adapter with the base model
and revision, dataset version, hash and text fingerprint, seed, LoRA config,
training arguments, package versions and git commit.

**This machine cannot train.** The GPU is a GTX 1050 (Pascal, cc 6.1);
bitsandbytes NF4 needs ≥ 7.5. `training/train.py` detects this and says so
instead of crashing. [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)
is set up for a free T4.

---

## Results

Only the prompt ceiling has real numbers; they are in
[Why fine-tuning?](#why-fine-tuning--the-prompt-ceiling) above and in
[`results/prompt_ceiling/`](results/prompt_ceiling/).

`results/base_vs_tuned/` and `results/data_efficiency/` are **NOT RUN**. They
will be populated by the commands below once a checkpoint exists. The
Minimum Viable Dataset Size is derived from measured points against the
`data_efficiency` thresholds in the spec — never interpolated, and reported as
NOT DETERMINED when no size clears them.

---

## Reproduction

```bash
# --- setup ---------------------------------------------------------------
python -m venv .venv && .venv/bin/pip install -e ".[providers,analysis,dev]"
cp .env.example .env          # add ANTHROPIC_API_KEY and OPENAI_API_KEY

# --- offline: no credentials, no GPU -------------------------------------
make test                     # 368 unit tests
make scenarios                # rebuild scenarios/*.jsonl, enforce split isolation
make eval-smoke               # end-to-end evaluation, mock model + offline judge
make smoke-data               # end-to-end data pipeline on a mock teacher
make prompt-ceiling-mock      # ablation pipeline, output labelled MOCKED
make train-dry                # validate training config, build + contamination-check data
make reanalyze                # re-render reports from saved transcripts
make analyze                  # failure modes, training distribution, plots
make plan                     # which calls a real run would purchase
make agreement                # human-vs-judge kappa (NOT YET GRADED)

# --- needs API credentials ------------------------------------------------
make preflight                # one cheap call per provider; verifies key/model/quota
make prompt-ceiling           # resumable; COMPLETE — 216/216, 0 calls left to buy
make generate-data CANDIDATES=1200 DATASET_VERSION=v1   # tranche 1; NOT RUN
make filter-data DATASET_VERSION=v1

# --- needs a CUDA GPU (cc >= 7.5) ----------------------------------------
make train RUN=socratic-v1
make evaluate RUN=socratic-v1        # base vs tuned
make data-efficiency                 # train + evaluate the sweep

# --- provenance -----------------------------------------------------------
make manifest                 # results/manifest.json
```

### The one command a grader runs

```bash
python eval.py --model <hf-repo-id-or-model-spec> --eval-set scenarios/heldout.jsonl
```

Model specs: `anthropic:claude-opus-5`, `openai:gpt-5`, `hf:Qwen/Qwen3-1.7B`,
`peft:Qwen/Qwen3-1.7B+outputs/socratic-v1`, `mock:demo`, or a bare Hugging Face
repo id. Add `--offline-judge` to run with no API access at all.

It fails with an explanation, not a stack trace, for: a missing model, an
unknown provider, an unreadable eval set, a malformed scenario (reporting the
line number and the id), absent credentials, and the case where every call died
of infrastructure failure.

*Windows note:* if `make` is unavailable, run the command inside each Makefile
recipe directly — every one is a single line.

---

## Reproducibility

`results/manifest.json` ties every number to what produced it: behavior spec
version and sha256, eval-set hashes and sizes, the three prompt versions with
content hashes, judge and generation prompt hashes, dataset versions and hashes,
checkpoint fingerprints, per-experiment status, git commit and dirty flag,
dependency versions and a lock hash, and Python version.

Content hashing is used throughout so results cannot drift silently: editing a
prompt changes its hash, so cross-prompt comparisons are impossible to make by
accident. Reports are a *rendering* of the raw judge transcripts, and
`scripts/reanalyze.py` regenerates them from saved records without re-spending a
single API call — which is how the corrected denominator policy was applied to
the run already completed.

---

## Limitations

Stated plainly, because the experiment is only as good as its weakest claim.

1. **The prompt ceiling is complete, but 36 scenarios per cell is still small.**
   Both families, all six cells, 216/216 evaluations, zero infrastructure loss.
   The gap it shows is large (the best cell misses the pass-rate threshold by
   0.061 and adherence by 0.076) and no cell clears the bar, so the verdict is
   robust. But 36 scenarios cannot resolve differences of a few points, and no
   confidence intervals are reported.
2. **Judge and subject overlap — quantified: 106 of 216 records.**
   `claude-opus-5` judged its own outputs in the three Anthropic cells. Two
   things limit the damage. The **winning cell is cross-family judged**: all 108
   `gpt-5` records were graded by a different family, so the shortfall that
   fired the gate is not self-preference. And on the Anthropic cells the bias
   direction *flatters* the prompted baseline, making the measured ceiling
   conservative and `JUSTIFIED` robust to it. Where it still bites is the
   model-vs-model comparison — `gpt-5` was graded by a rival family while
   `claude-opus-5` graded itself, so the cross-family ranking is the least
   trustworthy number in the table and is not relied on for the gate.

   Cross-family judging (Anthropic subject → OpenAI judge, and vice versa) was
   considered and **deliberately not adopted**, because it trades one bias for a
   worse one: each family would then be graded by a different judge, so a
   difference between families could not be separated from a difference between
   judges. Consistency across cells is what makes the six cells comparable to
   each other and to a fixed threshold. Switching mid-experiment would also have
   invalidated the 97 records already purchased, so the judge was deliberately
   held fixed for the whole primary experiment. The intended fix is instead a
   cross-family **audit** on a subsample plus the human validation below —
   measuring the bias rather than swapping it for another one.
   `judge_model_family` and `self_judged` are recorded per record so the audit
   can be scoped precisely.

   *Planned audit, not a prerequisite for this result:* re-judge the 106
   self-judged Anthropic records with `openai:gpt-5` as judge, holding the judge
   prompt and rubric fixed, and report the per-record verdict delta and the
   shift in each Anthropic cell's pass rate. If the shift is small the
   self-judging caveat is bounded empirically; if it is large, the Anthropic
   cells are re-reported with both judges side by side. Either way the gate is
   unaffected, because the gate fired on a cross-family-judged cell. Cost: 106
   judge calls, no subject calls.
3. **No human validation of the judge — the harness exists, no labels do.**
   `results/prompt_ceiling/human_validation.csv` holds a 40-row stratified,
   failure-enriched sample with empty human columns; `make agreement` computes
   percent agreement and Cohen's kappa once graded. It currently reports
   `NOT YET GRADED` and refuses to emit a number. Kappa matters here because the
   classes are unbalanced: at ~85% passes, a grader who wrote "pass" on every
   row would score ~85% raw agreement while carrying no information. Every
   downstream number inherits this unvalidated judge until the sheet is graded.
4. **Deterministic checks trade recall for precision by design.** A leak phrased
   in unusual prose can pass them; that is why they never run alone.
5. **56 scenarios is small.** Adequate for a large effect, too small to resolve
   differences of a few points, and confidence intervals are not reported.
6. **The teacher's ceiling is the dataset's ceiling.** Training data is filtered
   frontier output, so the tuned model inherits the teacher's blind spots. The
   quality gate rejects; it cannot create quality that was never generated.
7. **The behavior is narrow on purpose.** Nothing here says the tuned model is a
   good tutor overall — only that it holds one measurable constraint more often.
8. **Synthetic learners.** Pressure is written by a model imitating a frustrated
   student. Real learners are messier.
9. **Base model regressions are unmeasured.** No general-capability benchmark is
   run, so it is unknown whether tuning degrades ordinary helpfulness.

---

## Repository layout

```
behavior/spec.yaml          the specification everything reads
scenarios/*.jsonl           56 evaluation scenarios, three isolated splits
prompting/strategies.py     zero_shot | few_shot | structured_system_prompt
models/                     provider-independent adapters (API, local HF, PEFT)
evaluation/                 checks, judge, evaluator, metrics, reproducibility,
                            resume, failure_analysis, human_validation, plots
generation/                 controlled dimension space + teacher
filtering/                  static checks, judge, dedupe, balance, quality gate
training/                   chat-format conversion, QLoRA, configs
ablations/                  prompt_ceiling, base_vs_tuned, data_efficiency
results/                    JSON, CSV, Markdown, plots, raw judge transcripts
scripts/                    scenario build, filtering, reanalysis, manifest,
                            preflight, analyze_prompt_ceiling, judge_agreement
tests/                      368 tests, no network, no GPU
eval.py                     the one-command entry point
```

One deviation from the brief's layout: the evaluation package is `evaluation/`,
not `eval/`, because a top-level `eval/` package and a root `eval.py` cannot
coexist — `import eval` would be ambiguous. The root `eval.py` entry point is
preserved exactly as specified.
