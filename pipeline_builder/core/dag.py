from __future__ import annotations

import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class DAGNode:
    """A unit of work inside a DAGSpec.

    fn must be a zero-argument callable — use a lambda or functools.partial
    to capture parameters::

        DAGNode(
            id="fetch_ddl_order",
            fn=lambda: fetch_ddl("dw.order"),
            depends_on=["origin_search_order"],
        )
    """
    id: str
    fn: Callable[[], Any]
    depends_on: list[str] = field(default_factory=list)


class DAGResult:
    """Return value of DAGSpec.run()."""

    def __init__(self, results: dict[str, Any], errors: dict[str, Exception]) -> None:
        self.results = results
        self._errors = errors

    def ok(self, node_id: str) -> bool:
        return node_id in self.results

    def get(self, node_id: str, default: Any = None) -> Any:
        return self.results.get(node_id, default)

    def failed(self) -> list[str]:
        """Node IDs that raised an exception (excluding skipped dependents)."""
        return [nid for nid, exc in self._errors.items() if "_upstream_failed" not in str(exc)]

    def skipped(self) -> list[str]:
        """Node IDs skipped because an upstream dependency failed."""
        return [nid for nid, exc in self._errors.items() if "_upstream_failed" in str(exc)]

    def all_ok(self) -> bool:
        return len(self._errors) == 0


class DAGSpec:
    """Intra-stage dependency graph.

    Use inside a @pipe.stage when nodes have inter-dependencies that
    baton's simple fan-out can't express::

        @pipe.stage(reads=["probe"], writes=["probe"], fanout="manual")
        def execute_p4_dag(probes: list[Probe], state: State, ai) -> list[Probe]:
            dag = DAGSpec()

            for probe in probes:
                table = probe.table
                # Dedup: one fetch_ddl node per table
                if not dag.has(f"ddl_{table}"):
                    dag.add(DAGNode(
                        id=f"ddl_{table}",
                        fn=lambda t=table: fetch_ddl(t),
                    ))
                dag.add(DAGNode(
                    id=f"analyze_{probe.id}",
                    fn=lambda p=probe, ai=ai: ai.run_structured(...),
                    depends_on=[f"ddl_{table}"],
                ))

            result = dag.run(workers=4)
            return [update_probe(p, result) for p in probes]
    """

    def __init__(self) -> None:
        self._nodes: dict[str, DAGNode] = {}

    def add(self, node: DAGNode) -> "DAGSpec":
        if node.id in self._nodes:
            raise ValueError(f"Duplicate node id '{node.id}'")
        self._nodes[node.id] = node
        return self

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def __len__(self) -> int:
        return len(self._nodes)

    def run(self, workers: int = 1) -> DAGResult:
        """Execute the DAG respecting dependencies.

        Nodes whose dependencies all succeeded run in parallel (up to workers).
        If a node fails, its dependents are skipped (not failed) so the rest
        of the graph can still complete.

        workers=-1 → one thread per ready batch (unbounded)
        """
        self._validate()

        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        remaining = set(self._nodes.keys())

        while remaining:
            # A node is ready when every dependency is resolved (success or failure)
            ready = [
                nid for nid in remaining
                if all(dep in results or dep in errors for dep in self._nodes[nid].depends_on)
            ]

            if not ready:
                cycle_nodes = ", ".join(sorted(remaining))
                raise RuntimeError(
                    f"DAG cycle or unresolvable dependency detected. "
                    f"Stuck nodes: {cycle_nodes}"
                )

            # Split: nodes with a failed upstream are skipped immediately
            to_skip = [
                nid for nid in ready
                if any(dep in errors for dep in self._nodes[nid].depends_on)
            ]
            to_run = [nid for nid in ready if nid not in to_skip]

            for nid in to_skip:
                errors[nid] = RuntimeError(f"_upstream_failed: dependency of '{nid}' failed")
                remaining.discard(nid)

            if not to_run:
                continue

            if workers == 1 or len(to_run) == 1:
                for nid in to_run:
                    try:
                        results[nid] = self._nodes[nid].fn()
                    except Exception as exc:
                        errors[nid] = exc
                    remaining.discard(nid)
            else:
                max_w = None if workers == -1 else workers
                with ThreadPoolExecutor(max_workers=max_w) as executor:
                    futures = {executor.submit(self._nodes[nid].fn): nid for nid in to_run}
                    for future in as_completed(futures):
                        nid = futures[future]
                        try:
                            results[nid] = future.result()
                        except Exception as exc:
                            errors[nid] = exc
                        remaining.discard(nid)

        return DAGResult(results=results, errors=errors)

    async def run_async(self, workers: int = 1) -> DAGResult:
        """Async variant of run().

        Coroutine nodes are awaited directly; sync nodes are run in a thread
        pool via asyncio.to_thread so they don't block the event loop.

        workers=-1 → unbounded concurrency within each ready batch.
        """
        self._validate()

        results: dict[str, Any] = {}
        errors: dict[str, Exception] = {}
        remaining = set(self._nodes.keys())

        while remaining:
            ready = [
                nid for nid in remaining
                if all(dep in results or dep in errors for dep in self._nodes[nid].depends_on)
            ]
            if not ready:
                raise RuntimeError(
                    f"DAG cycle or unresolvable dependency detected. "
                    f"Stuck nodes: {', '.join(sorted(remaining))}"
                )

            to_skip = [
                nid for nid in ready
                if any(dep in errors for dep in self._nodes[nid].depends_on)
            ]
            to_run = [nid for nid in ready if nid not in to_skip]

            for nid in to_skip:
                errors[nid] = RuntimeError(f"_upstream_failed: dependency of '{nid}' failed")
                remaining.discard(nid)

            if not to_run:
                continue

            async def _run_one(nid: str) -> tuple[str, Any]:
                fn = self._nodes[nid].fn
                if inspect.iscoroutinefunction(fn):
                    val = await fn()
                else:
                    val = await asyncio.to_thread(fn)
                return nid, val

            max_concurrent = len(to_run) if workers == -1 else min(workers, len(to_run))
            sem = asyncio.Semaphore(max_concurrent)

            async def _guarded(nid: str) -> tuple[str, Any | None, Exception | None]:
                async with sem:
                    try:
                        n, v = await _run_one(nid)
                        return n, v, None
                    except Exception as exc:
                        return nid, None, exc

            gathered = await asyncio.gather(*[_guarded(nid) for nid in to_run])
            for nid, val, exc in gathered:
                if exc is not None:
                    errors[nid] = exc
                else:
                    results[nid] = val
                remaining.discard(nid)

        return DAGResult(results=results, errors=errors)

    def _validate(self) -> None:
        known = set(self._nodes.keys())
        for node in self._nodes.values():
            for dep in node.depends_on:
                if dep not in known:
                    raise ValueError(
                        f"Node '{node.id}' depends on '{dep}' which is not in the DAG"
                    )
