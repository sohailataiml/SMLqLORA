"""Why the corrected N=600 baseline still fails both solved-state scenarios.

The pre-registered decision rule selected H-B (solved-state release) because
`WITHHELD_AFTER_SOLVED` fired twice and the solved split scored 0/2. That is a
reason to look here. It is not, by itself, evidence that Dataset V1 caused it.

So this module measures the frozen corpus rather than arguing from the rule. It
derives its categories from signals that actually recur in the 85 solved
examples, applies them identically to the two held-out failures, and tests each
data-gap hypothesis against counts. Every number here recomputes from
`data/versions/v1/selected.jsonl` and `scenarios/heldout.jsonl`; nothing is
carried over from a previous run's prose.

    python -m analysis.solved_state            # print the report
    python -m analysis.solved_state --write    # also write the JSON artifacts
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ANALYSIS_VERSION = "1.0.0"

V1_PATH = REPO_ROOT / "data/versions/v1/selected.jsonl"
HELDOUT_PATH = REPO_ROOT / "scenarios/heldout.jsonl"
TRANSCRIPTS_PATH = REPO_ROOT / "results/n600_v1_baseline/judge_transcripts.jsonl"

# ---------------------------------------------------------------------------
# Signals
#
# These four patterns were chosen after reading the corpus, not before. Each
# fires on a majority or near-majority of the 85 solved examples, which is what
# makes it a recurring feature rather than an imposed category. They are kept
# deliberately shallow and inspectable: a learner saying what they changed, why
# it was wrong, that it now works, and showing the output.
# ---------------------------------------------------------------------------

SIGNAL_PATTERNS: dict[str, str] = {
    "describes_code_change": (
        r"\bI\s+(?:just\s+|then\s+)?(?:changed|moved|added|made|replaced|swapped"
        r"|materiali[sz]ed|switched|rewrote|set|removed|wrapped|pulled|hoisted"
        r"|declared|assigned|reassigned|converted|fixed)\b"
    ),
    "states_diagnosis": (
        r"\b(?:because|so the|so it|that.?s why|since|it (?:hands|gives|returns)"
        r"\s+back|was only|never (?:caught|saw|got)|turns out)\b"
    ),
    "claims_success": (
        r"\b(?:now|and now)\b[^.]{0,90}?\b(?:work|works|working|prints?|returns?"
        r"|shows?|gives?|passes|right|correct|proper(?:ly)?|as expected)\b"
        r"|\b(?:fixed|solved|it works|works now|all good|no more)\b"
        r"|\bboth values are right\b"
    ),
    "shows_runtime_output": (
        r"(?:prints?|get|got|returns?|shows?|outputs?|gives?)\b[^.]{0,40}`[^`]+`"
        r"|`[^`]*(?:\{[^`]*\}|=[^`]*)`"
    ),
}

_COMPILED = {name: re.compile(p, re.I) for name, p in SIGNAL_PATTERNS.items()}

#: The three signals that together constitute an unambiguous solved report.
CORE_SIGNALS = ("states_diagnosis", "describes_code_change", "claims_success")


def signal_profile(text: str) -> dict[str, Any]:
    """Which solved-state signals a learner message carries."""
    profile: dict[str, Any] = {
        name: bool(rx.search(text or "")) for name, rx in _COMPILED.items()
    }
    profile["word_count"] = len((text or "").split())
    profile["core_signal_count"] = sum(1 for s in CORE_SIGNALS if profile[s])
    return profile


def recognition_category(profile: dict[str, Any]) -> str:
    """A transparent label built from which core signals are present.

    Named for what the learner supplied, so the same rule classifies training
    examples and held-out scenarios without either being special-cased.
    """
    present = tuple(s for s in CORE_SIGNALS if profile[s])
    if len(present) == 3:
        return "diagnosis_change_and_success"
    if present == ("states_diagnosis", "describes_code_change"):
        return "diagnosis_and_change"
    if present == ("states_diagnosis", "claims_success"):
        return "diagnosis_and_success"
    if present == ("describes_code_change", "claims_success"):
        return "change_and_success"
    if present == ("states_diagnosis",):
        return "diagnosis_only"
    if present == ("describes_code_change",):
        return "change_only"
    if present == ("claims_success",):
        return "success_only"
    return "no_core_signal"


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_v1_solved(path: Path | None = None) -> list[dict]:
    """The frozen solved-state training examples, in file order."""
    rows = read_jsonl(path or V1_PATH)
    return [r for r in rows if r["dimensions"]["pressure_type"] == "solved"]


def load_v1_all(path: Path | None = None) -> list[dict]:
    return read_jsonl(path or V1_PATH)


def load_heldout_solved(path: Path | None = None) -> list[dict]:
    rows = read_jsonl(path or HELDOUT_PATH)
    return [r for r in rows if r.get("student_has_solved")]


def v1_student_message(record: dict) -> str:
    return record["scenario"].get("student_message", "")


def v1_prior_turns(record: dict) -> int:
    """Prior turns actually present in the example's conversation history.

    Counted from the history rather than read off the `conversation_turns`
    dimension, so the figure describes the record instead of the request that
    generated it.
    """
    return len(record["scenario"].get("conversation_history") or [])


# ---------------------------------------------------------------------------
# Tutor-side behaviour
# ---------------------------------------------------------------------------

CONFIRMATION_PATTERN = re.compile(
    r"\b(?:that.?s (?:exactly )?(?:right|it)|exactly (?:right|it)|yes,? you"
    r"|you.?ve got it|you got it|correct|that.?s the right fix|precisely"
    r"|spot on|nailed it)\b",
    re.I,
)
EXPLANATION_PATTERN = re.compile(
    r"\b(?:because|the reason|this is why|what happens is|under the hood"
    r"|evaluated once|by value|in CPython|the rule is)\b",
    re.I,
)
QUESTION_PATTERN = re.compile(r"\?")


def tutor_profile(response: str) -> dict[str, bool]:
    """Whether a tutor turn confirms, explains, and/or keeps questioning."""
    return {
        "confirms": bool(CONFIRMATION_PATTERN.search(response or "")),
        "explains": bool(EXPLANATION_PATTERN.search(response or "")),
        "asks_question": bool(QUESTION_PATTERN.search(response or "")),
    }


# ---------------------------------------------------------------------------
# Deterministic lexical similarity (no external service, no embeddings)
# ---------------------------------------------------------------------------

_TOKEN = re.compile(r"[a-z_][a-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall((text or "").lower())


def tfidf_vectors(
    documents: Sequence[str],
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Plain TF-IDF with L2 normalisation. Deterministic and inspectable."""
    tokenized = [tokenize(d) for d in documents]
    df: Counter[str] = Counter()
    for tokens in tokenized:
        df.update(set(tokens))
    n = len(documents)
    idf = {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}
    vectors: list[dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf = Counter(tokens)
        vec = {t: (c / len(tokens)) * idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors, idf


def cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def nearest_neighbours(
    query: str, corpus: Sequence[str], k: int = 5
) -> list[tuple[int, float]]:
    """Indices of the k most similar corpus entries, ties broken by index.

    The query is appended to the corpus so IDF is computed over one shared
    vocabulary rather than two.
    """
    vectors, _ = tfidf_vectors(list(corpus) + [query])
    qvec = vectors[-1]
    scored = [(i, cosine(qvec, vectors[i])) for i in range(len(corpus))]
    scored.sort(key=lambda pair: (-pair[1], pair[0]))
    return scored[:k]


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda kv: (-kv[1], str(kv[0]))))


def _lengths(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {}
    return {
        "n": len(values),
        "min": min(values),
        "median": float(statistics.median(values)),
        "max": max(values),
        "mean": round(statistics.fmean(values), 2),
    }


def solved_corpus_statistics(solved: Sequence[dict] | None = None) -> dict[str, Any]:
    """Everything measurable about the 85 frozen solved examples."""
    solved = list(solved if solved is not None else load_v1_solved())
    profiles = [signal_profile(v1_student_message(r)) for r in solved]
    tutors = [tutor_profile(r["tutor_response"]) for r in solved]
    categories = [recognition_category(p) for p in profiles]

    return {
        "count": len(solved),
        "prior_turn_distribution": _distribution(v1_prior_turns(r) for r in solved),
        "declared_conversation_turns": _distribution(
            r["dimensions"]["conversation_turns"] for r in solved
        ),
        "language": _distribution(r["dimensions"]["language"] for r in solved),
        "difficulty": _distribution(r["dimensions"]["difficulty"] for r in solved),
        "bug_category": _distribution(r["dimensions"]["bug_category"] for r in solved),
        "hint_strength": _distribution(r["dimensions"]["hint_strength"] for r in solved),
        "student_progress": _distribution(
            r["dimensions"]["student_progress"] for r in solved
        ),
        "student_message_words": _lengths([p["word_count"] for p in profiles]),
        "signal_counts": {
            name: sum(1 for p in profiles if p[name]) for name in SIGNAL_PATTERNS
        },
        "core_signal_count_distribution": _distribution(
            p["core_signal_count"] for p in profiles
        ),
        "recognition_categories": _distribution(categories),
        "tutor_behaviour": {
            "confirms": sum(1 for t in tutors if t["confirms"]),
            "explains": sum(1 for t in tutors if t["explains"]),
            "confirms_and_explains": sum(
                1 for t in tutors if t["confirms"] and t["explains"]
            ),
            "asks_question": sum(1 for t in tutors if t["asks_question"]),
            "confirms_without_question": sum(
                1 for t in tutors if t["confirms"] and not t["asks_question"]
            ),
        },
    }


def heldout_solved_profiles(
    heldout: Sequence[dict] | None = None,
) -> list[dict[str, Any]]:
    """The two held-out solved scenarios under the same classifier."""
    scenarios = list(heldout if heldout is not None else load_heldout_solved())
    out = []
    for s in scenarios:
        profile = signal_profile(s["student_message"])
        out.append({
            "id": s["id"],
            "language": s["language"],
            "bug_category": s["bug_category"],
            "difficulty": s["difficulty"],
            "prior_turns": len(s.get("conversation_history") or []),
            "student_message": s["student_message"],
            "signals": {k: bool(profile[k]) for k in SIGNAL_PATTERNS},
            "word_count": profile["word_count"],
            "core_signal_count": profile["core_signal_count"],
            "recognition_category": recognition_category(profile),
        })
    return out


# ---------------------------------------------------------------------------
# The corrected run's own behaviour
# ---------------------------------------------------------------------------

def corrected_transcripts(path: Path | None = None) -> list[dict]:
    return read_jsonl(path or TRANSCRIPTS_PATH)


def confirmation_behaviour(records: Sequence[dict] | None = None) -> dict[str, Any]:
    """How often the corrected model confirms anything, across all 20 outputs.

    The single most diagnostic number in this analysis. Its training targets for
    solved states confirm 82 times in 85; if the model confirms nothing at all,
    the solved failures are not a solved-state phenomenon.
    """
    records = list(records if records is not None else corrected_transcripts())
    profiles = [tutor_profile(r["model_response"]) for r in records]
    return {
        "outputs": len(records),
        "confirms": sum(1 for p in profiles if p["confirms"]),
        "asks_question": sum(1 for p in profiles if p["asks_question"]),
        "confirms_on_solved": sum(
            1 for r, p in zip(records, profiles)
            if p["confirms"] and r["scenario_id"].count("solved")
        ),
    }


def neighbour_report(k: int = 5) -> list[dict[str, Any]]:
    """Nearest V1 solved examples for each held-out solved scenario."""
    solved = load_v1_solved()
    corpus = [v1_student_message(r) for r in solved]
    train_responses = _training_split_responses()
    out = []
    for held in heldout_solved_profiles():
        neighbours = []
        for idx, score in nearest_neighbours(held["student_message"], corpus, k):
            record = solved[idx]
            tutor = tutor_profile(record["tutor_response"])
            neighbours.append({
                "v1_id": record["id"],
                "similarity": round(score, 4),
                "prior_turns": v1_prior_turns(record),
                "bug_category": record["dimensions"]["bug_category"],
                "language": record["dimensions"]["language"],
                "recognition_category": recognition_category(
                    signal_profile(corpus[idx])
                ),
                "in_training_split": record["tutor_response"] in train_responses
                if train_responses else None,
                "student_excerpt": corpus[idx][:200],
                "tutor_excerpt": record["tutor_response"][:200],
                "tutor_confirms": tutor["confirms"],
                "tutor_explains": tutor["explains"],
                "tutor_asks_question": tutor["asks_question"],
            })
        out.append({
            "heldout_id": held["id"],
            "heldout_bug_category": held["bug_category"],
            "heldout_recognition_category": held["recognition_category"],
            "neighbours": neighbours,
        })
    return out


def _training_split_responses() -> set[str]:
    """Assistant turns the model actually trained on, if the split is present."""
    path = REPO_ROOT / "outputs/socratic-v1-n600-bestckpt/data/train.jsonl"
    if not path.exists():
        return set()
    responses = set()
    for row in read_jsonl(path):
        for message in reversed(row["messages"]):
            if message["role"] == "assistant":
                responses.add(message["content"])
                break
    return responses


def solved_examples_in_training_split() -> dict[str, Any]:
    train = _training_split_responses()
    if not train:
        return {"available": False}
    solved = load_v1_solved()
    in_train = sum(1 for r in solved if r["tutor_response"] in train)
    return {
        "available": True,
        "solved_total": len(solved),
        "solved_in_training_split": in_train,
        "solved_in_validation_split": len(solved) - in_train,
    }


__all__ = [
    "ANALYSIS_VERSION",
    "CORE_SIGNALS",
    "confirmation_behaviour",
    "corrected_transcripts",
    "neighbour_report",
    "solved_examples_in_training_split",
    "SIGNAL_PATTERNS",
    "cosine",
    "heldout_solved_profiles",
    "load_heldout_solved",
    "load_v1_all",
    "load_v1_solved",
    "nearest_neighbours",
    "read_jsonl",
    "recognition_category",
    "signal_profile",
    "solved_corpus_statistics",
    "tfidf_vectors",
    "tokenize",
    "tutor_profile",
    "v1_prior_turns",
    "v1_student_message",
]
