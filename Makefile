# Socratic Debug Tutor — experiment commands.
#
# Every target that needs credentials or a GPU fails with an explanation rather
# than a stack trace. Targets that touch nothing external are safe to run
# anywhere, any time.
#
# On Windows without GNU make, run the PYTHON commands directly — each recipe
# below is a single command line by design. See README "Reproduction".

PYTHON ?= python
DATASET_VERSION ?= v1
BASE_MODEL ?= Qwen/Qwen3-1.7B
JUDGE ?= anthropic:claude-opus-5
MODELS ?= anthropic:claude-opus-5 openai:gpt-5
CANDIDATES ?= 1200
RUN ?= socratic-$(DATASET_VERSION)
EVAL_SET ?= scenarios/heldout.jsonl

.PHONY: help setup test lint scenarios eval-smoke smoke-data prompt-ceiling \
        prompt-ceiling-mock reanalyze analyze dataset-plan plan agreement preflight \
        generate-data filter-data train train-dry \
        evaluate data-efficiency data-efficiency-plan manifest clean-demo

help:
	@echo "Offline (no credentials, no GPU):"
	@echo "  make setup                 install the package and dev extras"
	@echo "  make test                  run the full unit suite"
	@echo "  make scenarios             rebuild scenarios/*.jsonl from source"
	@echo "  make eval-smoke            end-to-end evaluation with mock model + offline judge"
	@echo "  make smoke-data            end-to-end data pipeline on a mock teacher"
	@echo "  make prompt-ceiling-mock   ablation pipeline on mocks (labelled MOCKED)"
	@echo "  make train-dry             validate training config and build the dataset"
	@echo "  make reanalyze             re-render reports from saved transcripts"
	@echo "  make analyze               failure modes, training distribution, plots"
	@echo "  make dataset-plan          rebuild the Dataset V1 generation plan"
	@echo "  make plan                  show which calls a run would purchase"
	@echo "  make agreement             score human-vs-judge agreement (needs grading)"
	@echo ""
	@echo "Needs API credentials:"
	@echo "  make preflight             one cheap call per provider; spends ~nothing"
	@echo "  make prompt-ceiling        the real ablation (MODELS=..., JUDGE=...)"
	@echo "  make generate-data         teacher generation (CANDIDATES=$(CANDIDATES))"
	@echo "  make filter-data           quality gate -> dataset $(DATASET_VERSION)"
	@echo "  make evaluate              base vs tuned on the held-out set"
	@echo ""
	@echo "Needs a CUDA GPU (compute capability >= 7.5):"
	@echo "  make train                 QLoRA fine-tune"
	@echo "  make data-efficiency       train + evaluate the whole sweep"

# ---------------------------------------------------------------- offline

setup:
	$(PYTHON) -m pip install -e ".[providers,analysis,dev]"

test:
	$(PYTHON) -m pytest tests/ -q

scenarios:
	$(PYTHON) scripts/build_scenarios.py

eval-smoke:
	$(PYTHON) eval.py --model mock:demo --eval-set $(EVAL_SET) --offline-judge --limit 8

smoke-data:
	$(PYTHON) -m generation.generate --count 200 --mock --dataset-version vdemo
	$(PYTHON) scripts/filter_data.py --dataset-version vdemo --mock

prompt-ceiling-mock:
	$(PYTHON) -m ablations.prompt_ceiling --mock --output results/prompt_ceiling_mock

train-dry:
	$(PYTHON) -m training.train --dry-run --run-name $(RUN)-dry

reanalyze:
	$(PYTHON) scripts/reanalyze.py --results-dir results/prompt_ceiling

analyze:
	$(PYTHON) scripts/analyze_prompt_ceiling.py

dataset-plan:
	$(PYTHON) scripts/build_dataset_plan.py

plan:
	$(PYTHON) -m ablations.prompt_ceiling --plan --models $(MODELS)

agreement:
	$(PYTHON) scripts/judge_agreement.py

data-efficiency-plan:
	$(PYTHON) -m ablations.data_efficiency --plan

manifest:
	$(PYTHON) scripts/build_manifest.py

# ------------------------------------------------------- needs credentials

preflight:
	$(PYTHON) scripts/preflight.py --models $(MODELS) $(JUDGE)

prompt-ceiling:
	$(PYTHON) -m ablations.prompt_ceiling --models $(MODELS) --judge $(JUDGE) --preflight

generate-data:
	$(PYTHON) -m generation.generate --count $(CANDIDATES) \
		--dataset-version $(DATASET_VERSION) --teacher $(JUDGE)

filter-data:
	$(PYTHON) scripts/filter_data.py --dataset-version $(DATASET_VERSION) --judge $(JUDGE)

evaluate:
	$(PYTHON) -m ablations.base_vs_tuned --base hf:$(BASE_MODEL) \
		--tuned "peft:$(BASE_MODEL)+outputs/$(RUN)" --judge $(JUDGE)

# --------------------------------------------------------------- needs GPU

train:
	$(PYTHON) -m training.train --run-name $(RUN)

data-efficiency:
	$(PYTHON) -m ablations.data_efficiency --train --evaluate --judge $(JUDGE)

# ------------------------------------------------------------------ chores

clean-demo:
	rm -rf data/candidates/vdemo.jsonl data/candidates/vdemo.stats.json \
	       data/accepted/vdemo.jsonl data/rejected/vdemo.jsonl \
	       data/versions/vdemo results/prompt_ceiling_mock outputs/*-dry
