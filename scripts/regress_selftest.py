"""Controls for eval_regress.py's grading path (client venv; docker): every canonical solution must pass,
and a wrong answer and an os._exit(0) "solution" must fail, for HumanEval+ and MBPP+.

  $ZROFORM_CLIENT_VENV/bin/python scripts/regress_selftest.py [--limit N]
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eval_regress  # noqa: E402
import safe_grader  # noqa: E402


def canonical(bench: str) -> dict[str, str]:
    import pyarrow.parquet as pq
    f = eval_regress.dataset_files(bench)[0]
    rows = pq.read_table(f).to_pylist()
    if bench == "humanevalplus":
        return {r["task_id"]: r["prompt"] + r["canonical_solution"] for r in rows}
    return {str(r["task_id"]): r["code"] for r in rows}


async def main(a) -> None:
    for bench in ("humanevalplus", "mbppplus"):
        tasks = eval_regress.load_bench(bench)[: a.limit or None]
        sols = canonical(bench)

        async def check(task, code):
            ok, det = await safe_grader.check_function_calls(code, task["entry"], task["inputs"], eval_regress.matcher(task))
            return ok, det

        good = await asyncio.gather(*[check(t, sols[t["id"]]) for t in tasks])
        wrong = await asyncio.gather(*[check(t, f"def {t['entry']}(*a, **k):\n    return None\n") for t in tasks[:20]])
        hack = await asyncio.gather(*[check(t, "import os\nos._exit(0)\n") for t in tasks[:20]])
        fails = [(t["id"], d) for t, (ok, d) in zip(tasks, good) if not ok]
        print(f"{bench}: canonical pass {sum(ok for ok, _ in good)}/{len(tasks)}; wrong pass {sum(ok for ok, _ in wrong)}/20; "
              f"os._exit pass {sum(ok for ok, _ in hack)}/20")
        for f in fails[:10]:
            print("   canonical FAIL:", f)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    asyncio.run(main(ap.parse_args()))
