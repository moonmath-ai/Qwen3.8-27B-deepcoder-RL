"""Coding-RL reward that the program under test can't fake.

The cookbook grader (recipes/code_rl, lcb_utils.TEST_CODE) exec()s the generated program inside the grader
process and scores a pass by exit code 0, so a "solution" that calls os._exit(0) earns reward 1.0
(not covered by upstream PR #983). Here the program never shares a process, a file
system or a verdict channel with the expected outputs:

  1. Run: a fresh container per program (zf-code-runner image; no network, read-only root, tmpfs work dir,
     uid 65534, no capabilities, memory/pid/cpu caps, removed afterwards) executes the program on each test
     INPUT exactly as the cookbook's run_test does and returns the raw outputs. Expected outputs never
     enter the container, and nothing persists between programs.
  2. Grade: a trusted replay program that returns/prints those recorded outputs (as JSON data, never code)
     is graded by the cookbook's own run_test in a local subprocess, so comparison semantics are unchanged.

No valid result record (early exit, crashed runner, garbage) means fail; so does an empty test list.
`check_correctness` is a drop-in for tinker_cookbook.recipes.code_rl.code_grading.sandbox_check_correctness.
Needs the docker CLI (group `docker`) and the image: docker build -t zf-code-runner:1 -f scripts/runner.Dockerfile .
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from typing import Any

IMAGE = os.environ.get("ZF_RUNNER_IMAGE", "zf-code-runner:1")
CONCURRENCY = int(os.environ.get("ZF_GRADER_CONCURRENCY", "16"))
DOCKER_LIMITS = ["--network", "none", "--read-only", "--tmpfs", "/work:size=64m,exec,uid=65534,gid=65534",
                 "--tmpfs", "/tmp:size=64m,uid=65534,gid=65534", "--workdir", "/work", "--memory", "2g",
                 "--memory-swap", "2g", "--pids-limit", "128", "--cpus", "1", "--cap-drop", "ALL",
                 "--security-opt", "no-new-privileges", "--user", "65534:65534"]
MARKER = "<<ZF_RUN_RESULT>>"
_sem: asyncio.Semaphore | None = None

# Runs INSIDE the container. The execution steps mirror the cookbook's run_test (TEST_UTIL, tinker-cookbook
# 0.5.7 lcb_utils.py): same import preamble, same stdin-program rewrite into code(), same single module or
# Solution instance across tests, same per-test alarm and stdin mocking (call_method). It records outputs
# instead of comparing them. Values use a type-preserving JSON form (tuples, int-keyed dicts) so the
# cookbook's comparison quirks reproduce exactly in the replay.
RUNNER = r'''
import ast, json, signal, sys
payload = json.loads(sys.stdin.read())
real_stdout = sys.stdout
util = {}
exec(payload["test_util"], util)  # the cookbook's TEST_UTIL: RuntimeModule, call_method, Capturing, ...
test, timeout = payload["program"], payload["timeout"]
in_outs = json.loads(payload["input_output"])
fn_name = in_outs.get("fn_name")
records = []

def enc(x):
    if isinstance(x, tuple):
        return {"__t": [enc(v) for v in x]}
    if isinstance(x, list):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {"__d": [[enc(k), enc(v)] for k, v in x.items()]}
    return x

def done(status, **kw):
    real_stdout.write("\n" + payload["marker"] + json.dumps({"status": status, "records": records, **kw}) + "\n")
    real_stdout.flush()
    raise SystemExit(0)

util["reliability_guard"]()
sol = payload["preamble"]
if fn_name is not None:
    sol += test
    signal.alarm(timeout)
    try:
        mod = util["RuntimeModule"].from_string("tmp_sol", "", sol)
        tmp = mod if "class Solution" not in test else mod.Solution()
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        done("compile_error", error=repr(e))
    method_name = fn_name
else:
    try:
        astree = ast.parse(test)
        last_block = astree.body[-1]
        if isinstance(last_block, ast.If):
            if ast.unparse(last_block.test).strip() == "__name__ == '__main__'":
                test = ast.unparse(astree.body[:-1]) + "\n" + ast.unparse(last_block.body)
    except Exception:
        pass
    new_test = []
    for x in test.split("\n"):
        new_test.append(x + "\n" if x.startswith("from ") or x.startswith("import ") else "\t" + x + "\n")
    tmp_test, started = "", False
    for i in new_test:
        if i.startswith("\t") and not started:
            tmp_test += "stdin = sys.stdin\nstdout = sys.stdout\ndef code():\n" + i
            started = True
        elif started and (i.startswith("from ") or i.startswith("import ")):
            tmp_test += "\t" + i
        else:
            tmp_test += i
    sol += tmp_test
    signal.alarm(timeout)
    try:
        tmp = util["RuntimeModule"].from_string("tmp_sol", "", sol)
        signal.alarm(0)
    except Exception as e:
        signal.alarm(0)
        done("compile_error", error=repr(e))
    method_name = "code"
try:
    method = getattr(tmp, method_name)
except Exception as e:
    done("no_method", error=repr(e))

for inputs in in_outs["inputs"]:
    if fn_name is not None:
        inputs = [json.loads(line) for line in inputs.split("\n")]
        try:
            if isinstance(inputs[0], dict):
                inputs = [{int(k): v for k, v in inputs[0].items()}]
        except Exception:
            pass
        signal.alarm(timeout)
        try:
            output = method(*inputs)
            json.dumps(output)  # the cookbook does this too; unserialisable output is a runtime error there
            signal.alarm(0)
            records.append({"ok": True, "out": enc(output)})
        except Exception as e:
            signal.alarm(0)
            records.append({"ok": False, "error": repr(e)})
            done("ran")
    else:
        if isinstance(inputs, list):
            inputs = "\n".join(inputs)
        signal.alarm(timeout)
        with util["Capturing"]() as output:
            try:
                util["call_method"](method, inputs)
                signal.alarm(0)
                ok, err = True, None
            except Exception as e:
                signal.alarm(0)
                ok, err = False, repr(e)
        if not ok:
            records.append({"ok": False, "error": err})
            done("ran")
        records.append({"ok": True, "out": output[0]})
done("ran")
'''

# Runs INSIDE the container, for function-call benchmarks (HumanEval+, MBPP+): evaluates each input from its
# repr (inputs are not secret), calls the entry point, records outputs type-preserved. Expected outputs never
# enter the container; the caller compares them with the benchmark's own assertion() (check_function_calls).
RUNNER_CALL = r'''
import json, math, signal, sys
payload = json.loads(sys.stdin.read())
real_stdout = sys.stdout
records = []

class _Timeout(Exception):
    pass

def _alarm(signum, frame):
    raise _Timeout()

signal.signal(signal.SIGALRM, _alarm)

def enc(x):
    if isinstance(x, tuple):
        return {"__t": [enc(v) for v in x]}
    if isinstance(x, list):
        return [enc(v) for v in x]
    if isinstance(x, dict):
        return {"__d": [[enc(k), enc(v)] for k, v in x.items()]}
    if isinstance(x, set):
        return {"__s": [enc(v) for v in x]}
    if isinstance(x, float) and not math.isfinite(x):
        return {"__f": repr(x)}
    if isinstance(x, complex):
        return {"__c": [enc(x.real), enc(x.imag)]}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if type(x).__module__ == "numpy" and hasattr(x, "tolist"):
        return enc(x.tolist())
    return {"__o": type(x).__name__}  # e.g. re.Match: an opaque, non-None stand-in (evalplus checks `is not None`)

def done(status, **kw):
    real_stdout.write("\n" + payload["marker"] + json.dumps({"status": status, "records": records, **kw}) + "\n")
    real_stdout.flush()
    raise SystemExit(0)

ns = {"__name__": "solution"}
signal.alarm(payload["timeout"])
try:
    exec(payload["program"], ns)
    signal.alarm(0)
    fn = ns[payload["entry_point"]]
except BaseException as e:
    signal.alarm(0)
    done("compile_error", error=repr(e)[:300])
env = {"inf": math.inf, "nan": math.nan}
for src in payload["inputs"]:
    args = eval(src, dict(env))
    signal.alarm(payload["timeout"])
    try:
        out = fn(*args)
        signal.alarm(0)
        json.dumps(enc(out))
        records.append({"ok": True, "out": enc(out)})
    except Exception as e:
        signal.alarm(0)
        records.append({"ok": False, "error": repr(e)[:300]})
        done("ran")
done("ran")
'''


class Opaque:
    """A recorded object that isn't plain data (e.g. re.Match): not None, and equal to nothing but itself."""

    def __init__(self, type_name: str):
        self.type_name = type_name

    def __repr__(self):
        return f"<{self.type_name}>"


def decode(x):
    """Inverse of the runners' enc()."""
    if isinstance(x, list):
        return [decode(v) for v in x]
    if isinstance(x, dict):
        if "__t" in x:
            return tuple(decode(v) for v in x["__t"])
        if "__d" in x:
            return {decode(k): decode(v) for k, v in x["__d"]}
        if "__s" in x:
            return {decode(v) for v in x["__s"]}
        if "__f" in x:
            return float(x["__f"])
        if "__c" in x:
            return complex(decode(x["__c"][0]), decode(x["__c"][1]))
        if "__o" in x:
            return Opaque(x["__o"])
    return x


async def check_function_calls(program: str, entry_point: str, input_reprs: list[str], compare,
                               timeout: int = 6) -> tuple[bool, dict]:
    """Run `entry_point(*args)` for each input in a fresh locked-down container; `compare(i, output)` -> bool
    is called locally for each recorded output (expected values stay outside the sandbox). Fails closed."""
    global _sem
    if not input_reprs:
        return False, {"error": "no test cases"}
    marker = f"{MARKER}{uuid.uuid4().hex}:"
    payload = {"program": program, "entry_point": entry_point, "inputs": input_reprs, "timeout": timeout,
               "marker": marker}
    name = f"zf-run-{uuid.uuid4().hex[:16]}"
    _sem = _sem or asyncio.Semaphore(CONCURRENCY)
    async with _sem:
        out, rc = await _run(["docker", "run", "--rm", "-i", "--name", name, *DOCKER_LIMITS, IMAGE,
                              "python", "-I", "-c", RUNNER_CALL], payload, min((timeout + 1) * len(input_reprs) + 15, 180),
                             kill=["docker", "kill", name])
    rec = _last_record(out, marker)
    if rec is None or rec.get("status") != "ran" or not isinstance(rec.get("records"), list):
        return False, {"error": "no valid run record", "status": (rec or {}).get("status"), "exit_code": rc}
    records = rec["records"]
    if len(records) != len(input_reprs) or not all(isinstance(r, dict) and r.get("ok") is True for r in records):
        return False, {"error": "runtime error or missing outputs", "n_ok": sum(bool(r.get("ok")) for r in records)}
    for i, r in enumerate(records):
        try:
            if not compare(i, decode(r["out"])):
                return False, {"error": "wrong answer", "index": i}
        except Exception as e:  # noqa: BLE001 - a comparison that raises is a failure
            return False, {"error": f"compare raised {type(e).__name__}", "index": i}
    return True, {"n": len(records)}


# Runs LOCALLY in a throwaway subprocess (run_test's reliability_guard disables os functions in-process).
# Only trusted code executes here: the cookbook's TEST_UTIL and our replay program.
GRADE = r'''
import json, sys
p = json.loads(sys.stdin.read())
util = {}
exec(p["test_util"], util)
results, meta = util["run_test"]({"input_output": p["input_output"]}, test=p["replay"], timeout=p["timeout"])
sys.__stdout__.write("\n" + p["marker"] + json.dumps({"results": [r is True for r in results], "meta": meta},
                                                    default=str) + "\n")
'''

_DEC = '''
def _zf_dec(x):
    if isinstance(x, list):
        return [_zf_dec(v) for v in x]
    if isinstance(x, dict) and "__t" in x:
        return tuple(_zf_dec(v) for v in x["__t"])
    if isinstance(x, dict) and "__d" in x:
        return {_zf_dec(k): _zf_dec(v) for k, v in x["__d"]}
    return x
'''


def _preamble(test_util: str) -> str:
    """The import preamble the cookbook's run_test prepends to every program (taken from its source)."""
    start = test_util.index('sol = "') + len('sol = "')
    return json.loads('"' + test_util[start:test_util.index('"\n', start)] + '"')


def replay_program(records: list[dict], fn_name: str | None, class_based: bool) -> str:
    """Trusted stand-in for the program: returns (call-based) or prints (stdin) the recorded outputs in
    order, raising where the program raised. Records are embedded as a JSON string literal (data only)."""
    data = repr(json.dumps(records))
    if fn_name is not None:
        body = (f"_ZF_R = json.loads({data})\n" + _DEC +
                "def _zf_next():\n    r = _ZF_R.pop(0)\n    if not r['ok']:\n        raise RuntimeError(r['error'])\n"
                "    return _zf_dec(r['out'])\n")
        if class_based:
            return "import json\n" + body + f"class Solution:\n    def {fn_name}(self, *a):\n        return _zf_next()\n"
        return "import json\n" + body + f"def {fn_name}(*a):\n    return _zf_next()\n"
    # stdin programs are re-wrapped by run_test into code(), called once per test: keep the cursor on sys.
    return ("import json\nimport sys\n"
            f"_zf_r = json.loads({data})[getattr(sys, '_zf_i', 0)]; sys._zf_i = getattr(sys, '_zf_i', 0) + 1\n"
            "sys.stdout.write(_zf_r['out']) if _zf_r['ok'] else (_ for _ in ()).throw(RuntimeError(_zf_r['error']))\n")


def _last_record(stdout: bytes, marker: str) -> dict | None:
    text = stdout.decode("utf-8", "replace")
    i = text.rfind(marker)
    if i < 0:
        return None
    try:
        rec = json.loads(text[i + len(marker):].strip().splitlines()[0])
    except (ValueError, IndexError):
        return None
    return rec if isinstance(rec, dict) else None


async def _run(argv: list[str], payload: dict, timeout: float, kill: list[str] | None = None) -> tuple[bytes, int | None]:
    proc = await asyncio.create_subprocess_exec(*argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.DEVNULL)
    try:
        out, _ = await asyncio.wait_for(proc.communicate(json.dumps(payload).encode()), timeout)
        return out, proc.returncode
    except asyncio.TimeoutError:
        if kill:
            k = await asyncio.create_subprocess_exec(*kill, stdout=asyncio.subprocess.DEVNULL,
                                                     stderr=asyncio.subprocess.DEVNULL)
            await k.wait()
        proc.kill()
        await proc.wait()
        return b"", None


async def check_correctness(sample: list[dict[str, Any]], generation: str, timeout: int = 6,
                            backend: Any = None) -> tuple[bool, dict[str, Any]]:
    """Same signature and test format as the cookbook's sandbox_check_correctness (`backend` is ignored)."""
    global _sem
    from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample
    from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_UTIL

    if not sample:
        return False, {"error": "no test cases"}
    io = postprocess_lcb_sample(sample)["input_output"]
    in_outs = json.loads(io)
    n = len(in_outs["inputs"])
    if n == 0:
        return False, {"error": "no test cases"}
    inputs_only = json.dumps({"inputs": in_outs["inputs"], **({"fn_name": in_outs["fn_name"]} if "fn_name" in in_outs else {})})
    marker = f"{MARKER}{uuid.uuid4().hex}:"
    run_payload = {"test_util": TEST_UTIL, "preamble": _preamble(TEST_UTIL), "program": generation,
                   "input_output": inputs_only, "timeout": timeout, "marker": marker}
    name = f"zf-run-{uuid.uuid4().hex[:16]}"
    _sem = _sem or asyncio.Semaphore(CONCURRENCY)
    async with _sem:
        out, rc = await _run(["docker", "run", "--rm", "-i", "--name", name, *DOCKER_LIMITS, IMAGE,
                              "python", "-I", "-c", RUNNER], run_payload, (timeout + 1) * n + 15,
                             kill=["docker", "kill", name])
    rec = _last_record(out, marker)
    if rec is None or rec.get("status") != "ran" or not isinstance(rec.get("records"), list):
        return False, {"error": "no valid run record", "status": (rec or {}).get("status"), "exit_code": rc}
    records = rec["records"]
    if not all(isinstance(r, dict) and isinstance(r.get("ok"), bool) for r in records):
        return False, {"error": "malformed run record"}
    replay = replay_program(records, in_outs.get("fn_name"), "class Solution" in generation)
    grade_payload = {"test_util": TEST_UTIL, "replay": replay, "input_output": io, "timeout": timeout, "marker": marker}
    gout, _ = await _run([sys.executable, "-I", "-c", GRADE], grade_payload, 60)
    g = _last_record(gout, marker)
    if g is None:
        return False, {"error": "grader failed"}
    results = g.get("results") or []
    return bool(results) and len(results) == n and all(results), {"results": results, "meta": g.get("meta")}
