from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import Pipeline


def _step_label(step) -> str:
    """Human-readable label for a step (stage or checkpoint)."""
    btype = getattr(step, "_baton_type", "?")
    name = step.__name__

    if btype == "stage":
        reads = step._baton_reads
        writes = step._baton_writes
        workers = getattr(step, "_baton_workers", 1)
        fanout = getattr(step, "_baton_fanout", "auto")

        arrow = "  ".join(reads) + " → " + "  ".join(writes) if reads or writes else ""
        parallel = f" ⚡×{workers}" if workers > 1 else (" ⚡∞" if workers == -1 else "")
        manual = " [manual]" if fanout == "manual" else ""
        return f"[STAGE] {name}{parallel}{manual}  {arrow}"

    if btype == "checkpoint":
        on_reject = getattr(step, "_baton_on_reject", None)
        rollback = f"  ↩ {on_reject}" if on_reject else ""
        return f"[CHECK] {name}{rollback}"

    if btype == "router":
        targets = getattr(step, "_baton_targets", [])
        t_str = " | ".join(targets) if targets else "?"
        return f"[ROUTE] {name} → {t_str}"

    if btype == "loop":
        rollback_to = getattr(step, "_baton_rollback_to", "?")
        exit_on = getattr(step, "_baton_exit_on", [])
        max_rounds = getattr(step, "_baton_max_rounds", 3)
        exit_str = ", ".join(exit_on) if exit_on else "—"
        return f"[LOOP]  {name}  ↻ rollback:{rollback_to}  exit:{exit_str}  max:{max_rounds}"

    return f"[?] {name}"


def ascii_diagram(pipe: Pipeline) -> str:
    hierarchy_str = " → ".join(pipe.hierarchy)
    lines: list[str] = [
        f'Pipeline "{pipe.name}"  ·  {hierarchy_str}',
        "",
        f"  [IN]   {pipe.hierarchy[0]}",
    ]

    for step in pipe._steps:
        lines += ["    │", "    ▼", f"  {_step_label(step)}"]

    lines += ["    │", "    ▼", "  [END]", ""]

    if pipe._goal_checks:
        lines += ["  ── Goal Checks (post-stage, periodic) ──"]
        for gc in pipe._goal_checks:
            interval = getattr(gc, "_baton_gc_interval", "?")
            rollback_to = getattr(gc, "_baton_gc_rollback_to", None)
            max_c = getattr(gc, "_baton_gc_max_checks", "?")
            rb = f"  ↩ {rollback_to}" if rollback_to else ""
            lines.append(f"  [GOAL] {gc.__name__}  every:{interval}  max:{max_c}{rb}")
        lines.append("")

    return "\n".join(lines)


def mermaid_diagram(pipe: Pipeline) -> str:
    nodes: list[str] = []
    edges: list[str] = []

    # Input node
    nodes.append(f'    _in(["{pipe.hierarchy[0]} · input"])')

    prev = "_in"
    for step in pipe._steps:
        btype = getattr(step, "_baton_type", "?")
        nid = step.__name__

        if btype == "stage":
            reads = "  ".join(step._baton_reads)
            writes = "  ".join(step._baton_writes)
            workers = getattr(step, "_baton_workers", 1)
            parallel = f" ⚡×{workers}" if workers > 1 else (" ⚡∞" if workers == -1 else "")
            label = f"STAGE: {nid}{parallel}\\n{reads} → {writes}"
            nodes.append(f'    {nid}["{label}"]')
            edges.append(f"    {prev} --> {nid}")

        elif btype == "checkpoint":
            on_reject = getattr(step, "_baton_on_reject", None)
            nodes.append(f'    {nid}{{"{nid}\\n⏸ checkpoint"}}')
            edges.append(f"    {prev} --> {nid}")
            if on_reject:
                edges.append(f'    {nid} -->|"↩ reject"| {on_reject}')

        elif btype == "router":
            targets = getattr(step, "_baton_targets", [])
            nodes.append(f'    {nid}{{"{nid}\\n⬡ router"}}')
            edges.append(f"    {prev} --> {nid}")
            for target in targets:
                edges.append(f'    {nid} -->|"{target}"| {target}')

        elif btype == "loop":
            rollback_to = getattr(step, "_baton_rollback_to", "?")
            max_rounds = getattr(step, "_baton_max_rounds", 3)
            nodes.append(f'    {nid}{{"{nid}\\n↻ loop (max {max_rounds})"}}')
            edges.append(f"    {prev} --> {nid}")
            edges.append(f'    {nid} -->|"↩ rollback"| {rollback_to}')

        prev = nid

    nodes.append('    _end(["END"])')
    edges.append(f"    {prev} -->|confirm| _end")

    # Goal checks appear as annotations (dashed nodes) not in the main flow
    for gc in getattr(pipe, "_goal_checks", []):
        interval = getattr(gc, "_baton_gc_interval", "?")
        rollback_to = getattr(gc, "_baton_gc_rollback_to", None)
        gc_id = f"_gc_{gc.__name__}"
        nodes.append(f'    {gc_id}[/"{gc.__name__}\\n↺ every {interval} stages"/]')
        if rollback_to:
            edges.append(f'    {gc_id} -. "↩ rollback" .-> {rollback_to}')

    block = "\n".join(nodes) + "\n\n" + "\n".join(edges)
    return f"```mermaid\nflowchart TD\n{block}\n```"
