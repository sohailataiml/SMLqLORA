"""Experiment manifests: what exactly produced a number in `results/`.

Every experiment writes a manifest alongside its results. The manifest pins the
behavior spec, the prompt versions, the scenario-set hash, the judge, the model
revisions, the git commit and the dependency set — so a result can be traced,
and a claim of reproduction can be checked rather than believed.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from behavior.spec import BehaviorSpec, load_spec
from evaluation.schemas import Scenario, scenarios_hash

MANIFEST_VERSION = "1.0.0"

#: Packages whose versions materially affect results.
TRACKED_PACKAGES = (
    "anthropic",
    "openai",
    "pydantic",
    "torch",
    "transformers",
    "peft",
    "trl",
    "bitsandbytes",
    "datasets",
    "accelerate",
)


def git_commit(repo_root: Path | None = None) -> str:
    """Current commit, or an explicit marker when the repo is not versioned."""
    root = repo_root or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable:git-not-runnable"
    if result.returncode != 0:
        return "unavailable:not-a-git-repository"
    return result.stdout.strip()


def git_is_dirty(repo_root: Path | None = None) -> bool | None:
    root = repo_root or Path(__file__).resolve().parent.parent
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def package_versions(names: Sequence[str] = TRACKED_PACKAGES) -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str] = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not-installed"
    return out


def dependency_hash(versions: dict[str, str]) -> str:
    payload = json.dumps(versions, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def code_hash(paths: Sequence[Path]) -> str:
    """Hash of the source files that implement the experiment."""
    digest = hashlib.sha256()
    for path in sorted(paths):
        if path.is_file():
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass
class ExperimentManifest:
    """Machine-readable provenance for one experiment run."""

    experiment: str
    manifest_version: str = MANIFEST_VERSION

    behavior_spec_version: str = ""
    behavior_spec_sha256: str = ""

    eval_set_paths: list[str] = field(default_factory=list)
    eval_set_hash: str = ""
    eval_set_size: int = 0

    models: list[dict[str, str]] = field(default_factory=list)
    judge_model: str = ""
    judge_prompt_version: str = ""
    judge_prompt_sha256: str = ""
    prompt_versions: list[dict[str, str]] = field(default_factory=list)

    dataset_version: str | None = None
    dataset_hash: str | None = None
    base_model: str | None = None
    base_model_revision: str | None = None
    tuned_model: str | None = None
    tuned_model_revision: str | None = None

    generation_params: dict[str, Any] = field(default_factory=dict)
    git_commit: str = ""
    git_dirty: bool | None = None
    dependency_versions: dict[str, str] = field(default_factory=dict)
    dependency_lock_hash: str = ""
    python_version: str = ""
    platform: str = ""
    experiment_timestamp: str = ""

    #: "REAL_EXPERIMENT_RESULT" | "MOCKED" | "NOT_RUN"
    result_status: str = "NOT_RUN"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        from dataclasses import asdict

        return asdict(self)

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        return target


def build_manifest(
    experiment: str,
    *,
    spec: BehaviorSpec | None = None,
    scenarios: Sequence[Scenario] | None = None,
    scenario_paths: Sequence[str | Path] = (),
    models: Sequence[Any] = (),
    strategies: Sequence[Any] = (),
    judge: Any = None,
    generation_params: dict[str, Any] | None = None,
    result_status: str = "NOT_RUN",
    **extra: Any,
) -> ExperimentManifest:
    """Assemble a manifest from the live objects used by an experiment."""
    spec = spec or load_spec()
    versions = package_versions()

    manifest = ExperimentManifest(
        experiment=experiment,
        behavior_spec_version=spec.version,
        behavior_spec_sha256=spec.spec_sha256,
        eval_set_paths=[str(p) for p in scenario_paths],
        eval_set_hash=scenarios_hash(scenarios) if scenarios else "",
        eval_set_size=len(scenarios) if scenarios else 0,
        models=[m.describe() for m in models],
        prompt_versions=[s.describe() for s in strategies],
        generation_params=generation_params or {},
        git_commit=git_commit(),
        git_dirty=git_is_dirty(),
        dependency_versions=versions,
        dependency_lock_hash=dependency_hash(versions),
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        experiment_timestamp=datetime.now(timezone.utc).isoformat(),
        result_status=result_status,
    )

    if judge is not None:
        described = judge.describe()
        manifest.judge_model = described.get("judge_model", "")
        manifest.judge_prompt_version = described.get("judge_prompt_version", "")
        from evaluation.judge import judge_prompt_hash

        manifest.judge_prompt_sha256 = judge_prompt_hash(spec)

    for key, value in extra.items():
        if hasattr(manifest, key):
            setattr(manifest, key, value)
        else:
            raise AttributeError(f"ExperimentManifest has no field {key!r}")

    return manifest


def load_manifest(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ExperimentManifest",
    "MANIFEST_VERSION",
    "TRACKED_PACKAGES",
    "build_manifest",
    "code_hash",
    "dependency_hash",
    "git_commit",
    "git_is_dirty",
    "load_manifest",
    "package_versions",
]
