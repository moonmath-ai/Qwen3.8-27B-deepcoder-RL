"""Coding regression evals: HumanEval+ and MBPP+ through the safe grader.

The evalplus `test` field defines `inputs`, `results` and an `assertion(out, exp, atol)` with float tolerances.
Here the program runs on the inputs in a fresh locked-down container (safe_grader.check_function_calls) and
the recorded outputs are compared OUTSIDE it with evalplus's own assertion(); expected results never enter
the sandbox. (The cookbook's MBPP benchmark scores `exit_code == 0`, the same flaw as code_rl's grader.)
Tasks whose expected values come from a reference function at test time (6 HumanEval+, 3 MBPP+) are
skipped, as is HumanEval/32 (a property check, not expected values): 157 HumanEval+ and 375 MBPP+ remain. Single turn: the prompt asks for the function in a ```python block; code is taken with the cookbook's
extract_code_from_model.

  run:     $ZROFORM_CLIENT_VENV/bin/python scripts/eval_regress.py run --bench humanevalplus \\
               [--sampler-path tinker://...] --out <dir>
  compare: $ZROFORM_CLIENT_VENV/bin/python scripts/eval_regress.py compare <base_dir> <adapter_dir>
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import glob
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
DATA = {"humanevalplus": "evalplus/humanevalplus", "mbppplus": "evalplus/mbppplus"}


def dataset_files(name: str) -> list[str]:
    """The benchmark's parquet files: from the HF cache, downloading them on first use."""
    from huggingface_hub import snapshot_download
    return sorted(glob.glob(f"{snapshot_download(DATA[name], repo_type='dataset')}/data/*.parquet"))


def parse_tests(test_src: str):
    """(input reprs, expected results, atol, assertion function, called function name) from an evalplus `test`."""
    tree = ast.parse(test_src)
    lits, atol, callee = {}, 0.0, None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("inputs", "results"):
            lits[node.targets[0].id] = node.value
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "assertion" and len(node.args) >= 3:
            atol = ast.literal_eval(node.args[2])
            callee = getattr(getattr(node.args[0], "func", None), "id", None)  # assertion(<fn>(*inp), exp, atol)
    env = {"inf": math.inf, "nan": math.nan}
    inputs = [ast.get_source_segment(test_src, e) for e in lits["inputs"].elts]
    results = eval(compile(ast.Expression(lits["results"]), "<results>", "eval"), dict(env))
    # assertion() and its helpers: module-level imports and defs only (not check() / the test loop)
    defs = ast.Module(body=[n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom, ast.FunctionDef))
                            and getattr(n, "name", "") != "check"], type_ignores=[])
    ns = dict(env)
    exec(compile(defs, "<evalplus>", "exec"), ns)
    return inputs, results, atol, ns.get("assertion"), callee


def load_bench(name: str) -> list[dict]:
    import pyarrow.parquet as pq
    f = dataset_files(name)[0]
    out = []
    for r in pq.read_table(f).to_pylist():
        if "results =" not in r["test"]:
            continue  # 9 tasks (6 HumanEval+, 3 MBPP+) compute expected values with a ref_func at test time; skipped
        inputs, results, atol, assertion, callee = parse_tests(r["test"])
        if assertion is None:
            continue  # HumanEval/32 (find_zero) checks a property (poly(x) ~ 0), not expected values; skipped
        entry = r["entry_point"] if callee == "candidate" else callee  # HumanEval+ tests call `candidate`
        if name == "humanevalplus":
            prompt = ("Complete the following Python function. Reply with the complete function (including the "
                      "signature and any imports) in a single ```python code block.\n\n```python\n" + r["prompt"] + "```")
        else:
            prompt = (f"{r['prompt']}\nYour code should pass this test:\n{r['test_list'][0]}\n\nWrite the function "
                      f"`{entry}` in Python and reply with it in a single ```python code block.")
        out.append({"id": str(r["task_id"]), "prompt": prompt, "entry": entry, "inputs": inputs,
                    "results": results, "atol": atol, "assertion": assertion})
    return out


def matcher(task):
    """compare(i, output) for check_function_calls: evalplus's assertion() raises on a mismatch."""
    def compare(i, got):
        task["assertion"](got, task["results"][i], task["atol"])
        return True
    return compare


async def run(a) -> None:
    import safe_grader
    import tinker
    from tinker import types as t
    from tinker_cookbook import renderers
    from tinker_cookbook.recipes.code_rl.code_grading import extract_code_from_model
    from tinker_cookbook.renderers import get_text_content
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    os.makedirs(a.out, exist_ok=True)
    tasks = load_bench(a.bench)[: a.limit or None]
    renderer = renderers.get_renderer(a.renderer, get_tokenizer(a.model))
    svc = tinker.ServiceClient(base_url=a.base_url)
    sc = (svc.create_sampling_client(model_path=a.sampler_path) if a.sampler_path
          else svc.create_sampling_client(base_model=a.model))
    sp = t.SamplingParams(max_tokens=a.max_tokens, temperature=a.temperature, stop=renderer.get_stop_sequences())
    done = {json.loads(l)["id"] for l in open(f"{a.out}/results.jsonl")} if os.path.exists(f"{a.out}/results.jsonl") else set()
    res_f, sem, t0 = open(f"{a.out}/results.jsonl", "a"), asyncio.Semaphore(a.concurrency), time.time()

    async def one(task):
        async with sem:
            prompt = renderer.build_generation_prompt([{"role": "user", "content": task["prompt"]}])
            out = await sc.sample_async(prompt, a.n, sp)
            correct, truncated = 0, 0
            for seq in out.sequences:
                truncated += seq.stop_reason == "length"
                msg, _ = renderer.parse_response(seq.tokens)
                code = extract_code_from_model(get_text_content(msg))
                if not code:
                    continue
                ok, _ = await safe_grader.check_function_calls(code, task["entry"], task["inputs"], matcher(task))
                correct += ok
        res_f.write(json.dumps({"id": task["id"], "n": a.n, "correct": correct, "truncated": truncated}) + "\n")
        res_f.flush()

    await asyncio.gather(*[one(x) for x in tasks if x["id"] not in done])
    svc.close("success").result()
    rows = [json.loads(l) for l in open(f"{a.out}/results.jsonl")]
    summary = {"bench": a.bench, "problems": len(rows), "pass@1": round(sum(r["correct"] / r["n"] for r in rows) / len(rows), 4),
               "truncated": sum(r["truncated"] for r in rows), "seconds": round(time.time() - t0)}
    json.dump(summary, open(f"{a.out}/summary.json", "w"), indent=1)
    print(json.dumps(summary))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--bench", required=True, choices=sorted(DATA))
    r.add_argument("--out", required=True)
    r.add_argument("--sampler-path")
    r.add_argument("--n", type=int, default=2)
    r.add_argument("--temperature", type=float, default=1.0)
    r.add_argument("--base-url", default="http://127.0.0.1:8300")
    r.add_argument("--model", default="Qwen/Qwen3.8-27B")
    r.add_argument("--renderer", default="qwen3_8_low_reasoning")
    r.add_argument("--max-tokens", type=int, default=12288)
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--limit", type=int, default=0)
    c = sub.add_parser("compare")
    c.add_argument("base")
    c.add_argument("adapter")
    a = ap.parse_args()
    if a.cmd == "run":
        asyncio.run(run(a))
    else:  # paired by problem, the same statistics as eval_code.compare
        import eval_code
        eval_code.compare(a.base, a.adapter, key="id")
