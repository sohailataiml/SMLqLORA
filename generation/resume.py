"""Resume support for teacher generation.

Generating 1200 candidates is a long, paid operation. Losing it to a dropped
connection at candidate 1100 is unacceptable, so candidates are appended to disk
as they complete and a restart skips what is already there.

The invariant that makes this safe is **stable identity**: candidate `index` is
determined by the deterministic plan (`base_seed + index`), and its id is
`gen_<version>_<index>`. The same index always means the same dimension draw, so
resuming cannot silently shift the distribution, and a retried infrastructure
failure cannot produce two logical records for one plan point.

Two things are deliberately *not* resumable:

* **Infrastructure failures are not persisted.** A candidate that never came
  back leaves no row, so the next run retries exactly it. This is also why a
  provider outage can never be mistaken for a quality-gate rejection — it never
  reaches the gate.
* **A changed plan invalidates the cache.** The base seed and generation prompt
  hash are recorded alongside the candidates; if either moves, reusing rows
  would mix two different dimension spaces, so the mismatch is reported rather
  than silently accepted.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Iterator

from generation.schemas import GeneratedExample

_ID_RE = re.compile(r"_(\d+)$")


def index_of(example_id: str) -> int | None:
    """Recover the plan index from a candidate id, or None if unrecognizable."""
    match = _ID_RE.search(example_id or "")
    return int(match.group(1)) if match else None


def read_candidates(path: str | Path) -> tuple[list[GeneratedExample], int]:
    """Load prior candidates. Returns (candidates, unreadable_line_count).

    A truncated final line is the expected shape of an interrupted run, so a
    bad line is counted and skipped rather than treated as corruption of the
    whole file.
    """
    target = Path(path)
    if not target.exists():
        return [], 0

    candidates: list[GeneratedExample] = []
    malformed = 0
    seen: set[str] = set()
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                example = GeneratedExample.model_validate(json.loads(line))
            except Exception:  # noqa: BLE001 — a partial write is expected
                malformed += 1
                continue
            # Last write wins, but a duplicate id must never yield two records.
            if example.id in seen:
                continue
            seen.add(example.id)
            candidates.append(example)
    return candidates, malformed


def completed_indices(candidates: Iterable[GeneratedExample]) -> set[int]:
    """Plan indices already satisfied by candidates on disk."""
    out: set[int] = set()
    for example in candidates:
        index = index_of(example.id)
        if index is not None:
            out.add(index)
    return out


def pending_indices(count: int, done: set[int]) -> list[int]:
    """Plan indices still to generate, in plan order."""
    return [i for i in range(count) if i not in done]


class CandidateWriter:
    """Append candidates as they complete, so an interrupted run is not lost.

    Opened in append mode and flushed per record. The cost of a flush per
    candidate is irrelevant next to a teacher call, and it is what makes the
    file usable after a hard kill.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = None
        self._written = 0

    def __enter__(self) -> "CandidateWriter":
        self._handle = self.path.open("a", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    @property
    def written(self) -> int:
        return self._written

    def write(self, example: GeneratedExample) -> None:
        if self._handle is None:
            raise RuntimeError("CandidateWriter used outside its context manager")
        self._handle.write(
            json.dumps(example.model_dump(mode="json"), ensure_ascii=False) + "\n"
        )
        self._handle.flush()
        self._written += 1


def sidecar_path(candidates_path: str | Path) -> Path:
    return Path(candidates_path).with_suffix(".run.json")


def read_run_config(candidates_path: str | Path) -> dict[str, Any]:
    path = sidecar_path(candidates_path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def write_run_config(candidates_path: str | Path, config: dict[str, Any]) -> Path:
    path = sidecar_path(candidates_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return path


def check_run_config(
    candidates_path: str | Path, config: dict[str, Any]
) -> list[str]:
    """Reasons the existing candidates cannot be reused, if any."""
    prior = read_run_config(candidates_path)
    if not prior:
        return []
    problems: list[str] = []
    for key in ("base_seed", "generation_prompt_sha256", "dataset_version"):
        if key in prior and key in config and prior[key] != config[key]:
            problems.append(
                f"{key} changed: cached run used {prior[key]!r}, "
                f"this run uses {config[key]!r}"
            )
    return problems


def iter_jsonl_dicts(path: str | Path) -> Iterator[dict[str, Any]]:
    target = Path(path)
    if not target.exists():
        return
    with target.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:  # noqa: BLE001
                    continue


__all__ = [
    "CandidateWriter",
    "check_run_config",
    "completed_indices",
    "index_of",
    "iter_jsonl_dicts",
    "pending_indices",
    "read_candidates",
    "read_run_config",
    "sidecar_path",
    "write_run_config",
]
