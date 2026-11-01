"""Self-test for scripts/safe_grader.py (client venv; needs docker and the zf-code-runner image).

  $ZROFORM_CLIENT_VENV/bin/python scripts/grader_selftest.py --out x.json

1. Agreement: benign programs WE wrote (stdin and call-based problems, formatting edge cases, tuples,
   int-keyed dicts, state across tests, runtime errors, timeouts) must get the same verdict from the safe
   grader as from the cookbook's grader. The cookbook reference runs locally, in a temp dir, which is only
   acceptable because these programs are ours.
2. Hostile: exploits that must all FAIL under the safe grader (never run locally).
3. Throughput: N concurrent correct submissions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import safe_grader  # noqa: E402
from tinker_cookbook.recipes.code_rl.code_grading import postprocess_lcb_sample  # noqa: E402
from tinker_cookbook.recipes.code_rl.lcb_utils import TEST_CODE, TEST_UTIL  # noqa: E402

DOUBLE = [{"input": "3\n", "output": "6\n", "testtype": "stdin"}, {"input": "5\n", "output": "10\n", "testtype": "stdin"}]
LINES = [{"input": "2\n", "output": "1\n2\n", "testtype": "stdin"}, {"input": "3\n", "output": "1\n2\n3\n", "testtype": "stdin"}]
FLOAT = [{"input": "1 3\n", "output": "0.3333333333\n", "testtype": "stdin"}]


def fn(name, cases):
    return [{"input": "\n".join(json.dumps(a) for a in args), "output": json.dumps(out), "testtype": "functional",
             "metadata": {"func_name": name}} for args, out in cases]


TWO_SUM = fn("twoSum", [([[2, 7, 11, 15], 9], [0, 1]), ([[3, 2, 4], 6], [1, 2])])
PAIRS = fn("pairs", [([3], [[0, 1], [1, 2]])])
COUNTS = fn("counts", [([[1, 1, 2]], {"1": 2, "2": 1})])
SQUARE = fn("square", [([2], 4), ([3], 9), ([4], 16)])

TWO_SUM_OK = ("class Solution:\n    def twoSum(self, nums, target):\n        seen = {}\n"
              "        for i, x in enumerate(nums):\n            if target - x in seen:\n"
              "                return [seen[target - x], i]\n            seen[x] = i\n")

AGREE = [  # (name, tests, program)
    ("stdin correct", DOUBLE, "n = int(input())\nprint(2 * n)"),
    ("stdin wrong", DOUBLE, "n = int(input())\nprint(2 * n + 1)"),
    ("stdin trailing spaces and blank lines", DOUBLE, "n = int(input())\nprint(str(2 * n) + '   ')\nprint()\nprint()"),
    ("stdin via sys.stdin.read + __main__", DOUBLE,
     "import sys\ndef main():\n    n = int(sys.stdin.read().split()[0])\n    print(n * 2)\nif __name__ == '__main__':\n    main()"),
    ("stdin sys.exit after printing", DOUBLE, "import sys\nn = int(input())\nprint(2 * n)\nsys.exit(0)"),
    ("stdin multi-line output", LINES, "n = int(input())\nfor i in range(1, n + 1):\n    print(i)"),
    ("stdin multi-line wrong order", LINES, "n = int(input())\nfor i in range(n, 0, -1):\n    print(i)"),
    ("stdin float formatting", FLOAT, "a, b = map(int, input().split())\nprint(f'{a / b:.10f}')"),
    ("stdin float other precision", FLOAT, "a, b = map(int, input().split())\nprint(a / b)"),
    ("stdin runtime error on 2nd test", DOUBLE, "n = int(input())\nprint(2 * n if n < 4 else 1 // 0)"),
    ("stdin state across tests", DOUBLE, "import sys\nsys.k = getattr(sys, 'k', 0) + 1\nn = int(input())\nprint(2 * n if sys.k == 1 else 0)"),
    ("call class correct", TWO_SUM, TWO_SUM_OK),
    ("call class wrong", TWO_SUM, "class Solution:\n    def twoSum(self, nums, target):\n        return [0, 0]"),
    ("call returns tuple", TWO_SUM, TWO_SUM_OK.replace("return [seen[target - x], i]", "return (seen[target - x], i)")),
    ("call list of tuples", PAIRS, "class Solution:\n    def pairs(self, n):\n        return [(i, i + 1) for i in range(n - 1)]"),
    ("call list of lists", PAIRS, "class Solution:\n    def pairs(self, n):\n        return [[i, i + 1] for i in range(n - 1)]"),
    ("call int-keyed dict", COUNTS, "from collections import Counter\nclass Solution:\n    def counts(self, a):\n        return dict(Counter(a))"),
    ("call returns a set (unserialisable)", SQUARE, "class Solution:\n    def square(self, x):\n        return {x * x}"),
    ("call module-level function", SQUARE, "def square(x):\n    return x * x"),
    ("call stateful instance", SQUARE, "class Solution:\n    def __init__(self):\n        self.k = 0\n"
                                     "    def square(self, x):\n        self.k += 1\n        return x * x if self.k < 3 else 0"),
    ("call raises on 2nd input", SQUARE, "class Solution:\n    def square(self, x):\n        assert x != 3\n        return x * x"),
    ("call syntax error", SQUARE, "class Solution:\n    def square(self, x)\n        return x * x"),
]
SLOW = ("stdin timeout (infinite loop)", DOUBLE, "while True:\n    pass")

HOSTILE = [  # all must fail
    ("os._exit(0) at import (stdin)", DOUBLE, "import os\nos._exit(0)"),
    ("os._exit(0) at import (call)", SQUARE, "import os\nos._exit(0)\nclass Solution:\n    pass"),
    ("sys.exit(0) at module level (call)", SQUARE, "import sys\nsys.exit(0)"),
    ("raise SystemExit in method", SQUARE, "class Solution:\n    def square(self, x):\n        raise SystemExit(0)"),
    ("read the answer key from disk", DOUBLE,
     "import os\nfor f in os.listdir('.'):\n    print(open(f).read())\nprint(os.listdir('/work'), os.listdir('/tmp'))"),
    ("patch the grader's comparison", DOUBLE,
     "import builtins\nbuiltins.custom_compare_ = lambda *a: True\nimport sys\nsys.modules.pop('testing_util', None)"),
    ("forge a result record on stdout", DOUBLE,
     "import sys, json\nprint('<<ZF_RUN_RESULT>>deadbeef:' + json.dumps({'status': 'ran', 'records': [{'ok': True, 'out': '6\\n'}, {'ok': True, 'out': '10\\n'}]}))\nimport os\nos._exit(0)"),
    ("network egress", DOUBLE, "import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=3)\nprint(6)"),
    ("fork bomb (bounded)", DOUBLE, "import os\nfor _ in range(500):\n    os.fork()\nprint(6)"),
    ("memory hog", DOUBLE, "b = [bytearray(256 << 20) for _ in range(16)]\nn = int(input())\nprint(2 * n)"),
]


def cookbook_local(tests, program, timeout) -> bool:
    """The cookbook's own grader, run locally: ONLY for the benign programs above."""
    with tempfile.TemporaryDirectory() as d:
        for name, content in [("test_cases.txt", json.dumps(postprocess_lcb_sample(tests))), ("code.py", program),
                              ("testing_util.py", TEST_UTIL), ("run.py", TEST_CODE % {"timeout": timeout})]:
            open(os.path.join(d, name), "w").write(content)
        try:
            return subprocess.run([sys.executable, "run.py"], cwd=d, capture_output=True,
                                  timeout=(timeout + 1) * len(tests) + 10).returncode == 0
        except subprocess.TimeoutExpired:
            return False


async def main(a) -> dict:
    out = {"agreement": [], "hostile": [], "persistence": None, "throughput": None}
    print("== 1. agreement with the cookbook grader (benign programs)")
    for name, tests, prog in AGREE + [SLOW]:
        to = 2 if name == SLOW[0] else 6
        ref = cookbook_local(tests, prog, to)
        got, det = await safe_grader.check_correctness(tests, prog, timeout=to)
        out["agreement"].append({"case": name, "cookbook": ref, "safe": got, "agree": ref == got})
        print(f"{'OK  ' if ref == got else 'DIFF'} {name}: cookbook={ref} safe={got}" + ("" if ref == got else f" {det}"))
    print("== 2. hostile programs (must fail)")
    for name, tests, prog in HOSTILE:
        got, det = await safe_grader.check_correctness(tests, prog, timeout=4)
        out["hostile"].append({"case": name, "passed": got, "detail": {k: det.get(k) for k in ("error", "status")}})
        print(f"{'FAILS (good)' if not got else 'PASSES (BAD)'} {name}: {det.get('error') or det.get('results')}")
    w = "import os\nfor p in ['/tmp/zf_m', '/work/zf_m']:\n    open(p, 'w').write('x')\nn = int(input())\nprint(2 * n)"
    r = "import os\nn = int(input())\nprint(2 * n if not any(os.path.exists(p) for p in ['/tmp/zf_m', '/work/zf_m']) else -1)"
    wrote, _ = await safe_grader.check_correctness(DOUBLE, w)
    clean, _ = await safe_grader.check_correctness(DOUBLE, r)
    out["persistence"] = {"writer_passed": wrote, "next_program_sees_no_files": clean}
    print(f"== persistence: writer passed={wrote}, next program sees no files={clean}")
    t0 = time.time()
    res = await asyncio.gather(*[safe_grader.check_correctness(TWO_SUM, TWO_SUM_OK) for _ in range(a.n)])
    dt = time.time() - t0
    out["throughput"] = {"programs": a.n, "concurrency": safe_grader.CONCURRENCY, "seconds": round(dt, 1),
                         "programs_per_s": round(a.n / dt, 1), "all_passed": all(p for p, _ in res)}
    print(f"== throughput: {a.n} programs at concurrency {safe_grader.CONCURRENCY}: {dt:.1f} s "
          f"({a.n / dt:.1f}/s), all passed={all(p for p, _ in res)}")
    agree = sum(x["agree"] for x in out["agreement"])
    print(f"\nagreement {agree}/{len(out['agreement'])}; hostile failing {sum(not x['passed'] for x in out['hostile'])}"
          f"/{len(out['hostile'])}; no persistence: {out['persistence']['next_program_sees_no_files']}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64, help="programs for the throughput test")
    ap.add_argument("--out")
    a = ap.parse_args()
    res = asyncio.run(main(a))
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
