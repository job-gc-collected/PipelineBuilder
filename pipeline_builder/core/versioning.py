"""Pipeline versioning — detect and classify changes between pipeline code and session state.

When a session is resumed after the pipeline code changed, baton classifies the
delta as one of three kinds:

* **Identical** — fingerprints match, resume transparently.
* **Safe change** — pipeline changed but all previously-completed stages still
  exist (e.g. new stages added, parameter tweaks, stage logic fixes).  Resume
  with a warning.
* **Breaking change** — a stage that was completed in a previous run no longer
  exists in the current pipeline.  Raises :class:`CompatibilityError`.

Renames are handled by adding ``aliases=["old_name"]`` to the new stage
decorator.  Any completed stage whose name matches an alias is transparently
remapped to the new canonical name.
"""
from __future__ import annotations

import hashlib
import json
import warnings
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass


class CompatibilityError(Exception):
    """Raised when a session cannot be resumed due to an incompatible pipeline change.

    The error message is actionable: it names the specific stage(s) that are
    missing and suggests how to fix the problem.
    """


def compute_fingerprint(name: str, hierarchy: list[str], steps: list) -> str:
    """Stable 16-char hex hash of the pipeline structure.

    Only structural aspects are hashed (stage name, reads, writes, hierarchy).
    Changes to stage logic, workers, timeout, etc. do NOT change the fingerprint
    — those are safe to change between runs.
    """
    stage_info = []
    for s in steps:
        if getattr(s, "_baton_type", None) == "stage":
            stage_info.append({
                "name": s.__name__,
                "reads": sorted(getattr(s, "_baton_reads", [])),
                "writes": sorted(getattr(s, "_baton_writes", [])),
            })
    payload = {"name": name, "hierarchy": hierarchy, "stages": stage_info}
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def check_resume_compatibility(
    session_id: str,
    completed_stages: list[str],
    stored_fingerprint: str,
    current_fingerprint: str,
    current_stage_names: set[str],
    aliases: dict[str, str],            # old_name → canonical_name
    current_hierarchy: list[str],
    stored_hierarchy: list[str] | None,
) -> list[str]:
    """Validate that a session is compatible with the current pipeline.

    Returns the (possibly alias-remapped) completed_stages list.
    Issues :class:`CompatibilityWarning` for safe changes.
    Raises :class:`CompatibilityError` for breaking changes.
    """
    if stored_fingerprint == current_fingerprint:
        return completed_stages  # identical — nothing to check

    # ── Hierarchy change is always breaking ──────────────────────────────
    if stored_hierarchy is not None and stored_hierarchy != current_hierarchy:
        raise CompatibilityError(
            f"Session '{session_id}' cannot be resumed: "
            f"the pipeline hierarchy changed from {stored_hierarchy} to {current_hierarchy}. "
            f"Create a new session or restore the original hierarchy."
        )

    # ── Remap completed stages via aliases ───────────────────────────────
    remapped: list[str] = []
    for stage in completed_stages:
        if stage in current_stage_names:
            remapped.append(stage)
        elif stage in aliases:
            canonical = aliases[stage]
            warnings.warn(
                f"[baton] Session '{session_id}': stage '{stage}' was renamed to "
                f"'{canonical}' (matched via aliases). "
                "Treating it as completed.",
                UserWarning,
                stacklevel=4,
            )
            remapped.append(canonical)
        else:
            raise CompatibilityError(
                f"Session '{session_id}' cannot be resumed: "
                f"stage '{stage}' was completed in a previous run but no longer exists "
                f"in the current pipeline.\n\n"
                f"  If you renamed this stage, add  aliases=['{stage}']  to the new "
                f"stage decorator so baton can map the old name to the new one.\n"
                f"  If you intentionally removed it, delete or abandon this session."
            )

    # ── Safe change ───────────────────────────────────────────────────────
    warnings.warn(
        f"[baton] Session '{session_id}': pipeline fingerprint changed "
        f"({stored_fingerprint!r} → {current_fingerprint!r}). "
        "All previously completed stages are still present — resuming with updated pipeline.",
        UserWarning,
        stacklevel=4,
    )
    return remapped
