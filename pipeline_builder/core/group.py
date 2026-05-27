"""Stage-group DAG utilities — pure functions with no side effects.

Design principle
---------------
All logic that reasons about *which stages can run in parallel* and
*in what order* lives here.  These are pure functions: they take data
in, return data out, and touch no I/O, no state, no pipeline config.

By keeping DAG logic pure and centralised:

* Both ``Pipeline._validate()`` and ``PipelineExecutor._run_stage_group()``
  call the same code paths — no divergence, no duplication.
* The functions are trivially unit-testable in isolation.
* Future scheduling improvements (weighted priority, resource limits, …)
  have a single home.
"""
from __future__ import annotations

from typing import Any, Callable


def can_reach(deps: dict[str, list[str]], src: str, dst: str) -> bool:
    """Return True if ``dst`` is transitively reachable from ``src`` in ``deps``.

    ``deps`` maps each stage name to the list of stage names it depends on
    (its *direct* predecessors).  This function follows those edges forwards
    (src → … → dst) to determine transitive reachability.

    Used to distinguish genuinely parallel stages (no path between them) from
    sequentially-chained stages (one depends on the other, directly or not).
    """
    visited: set[str] = set()
    queue = list(deps.get(src, []))
    while queue:
        cur = queue.pop()
        if cur == dst:
            return True
        if cur not in visited:
            visited.add(cur)
            queue.extend(deps.get(cur, []))
    return False


def infer_group_deps(group: list[Callable]) -> dict[str, list[str]]:
    """Infer dependency relationships within a group of stages.

    Rules (applied in order, registration order wins):

    1. **reads/writes inference**: if stage B reads level X and stage A
       (registered *before* B) writes level X, then B depends on A.
       Only the *closest* predecessor for each level is recorded.

    2. **explicit overrides**: any names listed in a stage's
       ``_baton_depends_on`` attribute are added as additional direct
       dependencies (useful for cross-level dependencies that
       reads/writes cannot express).

    Returns a dict mapping each stage name to its list of direct
    predecessor names within this group.
    """
    group_names = {s.__name__ for s in group}
    deps: dict[str, list[str]] = {s.__name__: [] for s in group}

    for i, s in enumerate(group):
        # Rule 1: closest earlier writer of each read level
        for level in s._baton_reads:
            for j in range(i - 1, -1, -1):
                earlier = group[j]
                if level in earlier._baton_writes:
                    if earlier.__name__ not in deps[s.__name__]:
                        deps[s.__name__].append(earlier.__name__)
                    break   # only the closest predecessor per level

        # Rule 2: explicit cross-level dependencies
        for explicit in getattr(s, "_baton_depends_on", []):
            if explicit in group_names and explicit not in deps[s.__name__]:
                deps[s.__name__].append(explicit)

    return deps


def collect_stage_groups(steps: list[Callable]) -> list[list[Callable]]:
    """Split a flat step list into consecutive-stage groups.

    A *group* is a maximal run of ``stage``-type steps between two
    barriers (checkpoint / router / loop / goal_check).  Stages inside
    a group may run in parallel; barriers run sequentially and separate
    groups from one another.

    Example::

        steps = [stage_a, stage_b, checkpoint, stage_c]
        #         └──── group 1 ────┘              └── group 2 ──┘
    """
    groups: list[list[Callable]] = []
    current: list[Callable] = []
    for step in steps:
        if getattr(step, "_baton_type", None) == "stage":
            current.append(step)
        else:
            if current:
                groups.append(current)
                current = []
    if current:
        groups.append(current)
    return groups


def levels_to_clear(
    group: list[Callable],
    deps: dict[str, list[str]],
) -> set[str]:
    """Return hierarchy levels that should be cleared before a fresh group run.

    A level is cleared when two or more stages in the same group both
    write to it *and* neither transitively depends on the other — they
    will run in parallel and their results must be merged (extend_nodes
    semantics), so the level must start empty.

    Single-writer levels, and levels shared by stages that have a
    dependency path between them, are left untouched so that data
    written by previous groups (or partially-completed resumes) is
    preserved.
    """
    level_writers: dict[str, list[str]] = {}
    for s in group:
        for lv in s._baton_writes:
            level_writers.setdefault(lv, []).append(s.__name__)

    to_clear: set[str] = set()
    for lv, writers in level_writers.items():
        if len(writers) > 1 and any(
            not (can_reach(deps, a, b) or can_reach(deps, b, a))
            for i, a in enumerate(writers)
            for b in writers[i + 1:]
        ):
            to_clear.add(lv)
    return to_clear


def batch_extend_mode(
    to_run: list[Callable],
) -> dict[str, bool]:
    """For each stage in a ready batch, decide whether to use extend_nodes.

    extend_nodes is needed when two stages in the *same concurrent batch*
    both write to the same level (they run in parallel and must append).
    set_nodes is used in all other cases (sequential / single-writer).

    Returns a dict mapping stage name → use_extend (bool).
    """
    batch_writers: dict[str, list[str]] = {}
    for s in to_run:
        for lv in s._baton_writes:
            batch_writers.setdefault(lv, []).append(s.__name__)

    return {
        s.__name__: any(
            len(batch_writers.get(lv, [])) > 1 for lv in s._baton_writes
        )
        for s in to_run
    }
