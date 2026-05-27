"""Parallel execution speedup — workers=N vs workers=1.

Demonstrates:
- workers=N: N nodes processed concurrently with asyncio.gather
- Timing shows parallel ≈ max(per-node time), not sum

4 nodes × 0.1s each:
  workers=1 → ~0.4s  (sequential)
  workers=4 → ~0.1s  (parallel)

Run from the project root:
    python examples/parallel_demo.py
"""
from __future__ import annotations

import asyncio
import time

from pydantic import BaseModel
from pipeline_builder import Pipeline, State


class Item(BaseModel):
    id: int


class Result(BaseModel):
    item_id: int
    elapsed_ms: int


def make_pipeline(workers: int) -> Pipeline:
    pipe = Pipeline(f"parallel_w{workers}", hierarchy=["item", "result"])

    @pipe.stage(reads=["item"], writes=["result"], workers=workers)
    async def process(item: Item, state: State) -> list[Result]:
        t0 = time.monotonic()
        await asyncio.sleep(0.1)   # simulates an async AI call
        return [Result(item_id=item.id, elapsed_ms=int((time.monotonic() - t0) * 1000))]

    return pipe


async def main() -> None:
    items = [Item(id=i) for i in range(4)]

    for workers in [1, 4]:
        pipe = make_pipeline(workers)
        t0 = time.monotonic()
        result = await pipe.run_async(item=items)
        total = time.monotonic() - t0

        results = result.get_nodes("result")
        print(f"workers={workers}  total={total:.2f}s  "
              f"item_ids={[r.item_id for r in results]}")

    print("\n✓ workers=4 should be ~4× faster than workers=1")


if __name__ == "__main__":
    asyncio.run(main())
