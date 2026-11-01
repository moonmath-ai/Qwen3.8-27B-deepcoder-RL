"""Machine portability: profile paths come from the environment, and SGLang runs on the profile's GPUs."""

import asyncio
import os
from pathlib import Path

import pytest

from zroform.config import load_profile, restrict_visible_gpus
from zroform.sampler import SamplerError, SGLangSampler

PROFILES = sorted((Path(__file__).parents[2] / "profiles").glob("*.yaml"))


@pytest.mark.parametrize("path", PROFILES, ids=lambda p: p.name)
def test_every_shipped_profile_loads_with_its_store_under_zroform_data(path, monkeypatch):
    monkeypatch.setenv("ZROFORM_DATA", "/data/zf")
    assert load_profile(path).store_dir == "/data/zf/store"


def test_env_expands_only_dir_settings_and_an_unset_variable_is_an_error(tmp_path, monkeypatch):
    p = tmp_path / "p.yaml"
    p.write_text('name: "${NAME}"\nstore_dir: ${ZF}/s\nmodels: {m: {}}\n')
    monkeypatch.setenv("ZF", "/x")
    monkeypatch.setenv("NAME", "evil")
    prof = load_profile(p)
    assert (prof.store_dir, prof.name) == ("/x/s", "${NAME}")
    monkeypatch.delenv("ZF")
    with pytest.raises(ValueError, match="ZF"):
        load_profile(p)


class _ExitedProc:
    returncode = 1

    def poll(self):
        return 1


def _launch(monkeypatch, tmp_path, gpus, parent_visible):
    seen = {}

    def fake_popen(cmd, env, **kw):
        seen.update(cmd=cmd, visible=env["CUDA_VISIBLE_DEVICES"])
        return _ExitedProc()

    monkeypatch.setattr("zroform.sampler.subprocess.Popen", fake_popen)
    if parent_visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent_visible)
    s = SGLangSampler("m", 30100, 16, ["q_proj"], 2, 0.5, 1024, str(tmp_path / "sglang.log"), gpus=gpus)
    with pytest.raises(SamplerError):  # the fake process has exited: start() reports it
        asyncio.run(s.start(timeout=5))
    return seen["visible"], seen["cmd"][seen["cmd"].index("--tp-size") + 1]


def test_sglang_gets_the_profile_gpus_and_one_tp_rank_per_gpu(monkeypatch, tmp_path):
    assert _launch(monkeypatch, tmp_path, [4, 5, 6, 7], None) == ("4,5,6,7", "4")


def test_sampler_gpu_indices_are_relative_to_the_servers_own_visible_devices(monkeypatch, tmp_path):
    # Server started with CUDA_VISIBLE_DEVICES=2,3,6: its GPU 1 is physical GPU 3.
    assert _launch(monkeypatch, tmp_path, [1], "2,3,6") == ("3", "1")


@pytest.mark.parametrize("parent, trainer, sampler, visible, expect", [
    (None, "cuda:0", [4, 5], "0,4,5", ("cuda:0", [1, 2])),     # node8x-style split
    ("2,3,6", "cuda:0", [2], "2,6", ("cuda:0", [1])),          # indices relative to inherited devices
    (None, "cuda", [], "0", ("cuda:0", [0])),                  # colocated default (dev box)
    (None, "cuda:3", [], "3", ("cuda:0", [0])),                # colocated on another GPU
])
def test_server_sees_only_the_profile_gpus_renumbered_from_zero(monkeypatch, parent, trainer, sampler,
                                                                 visible, expect):
    # The first cross-GPU copy opens a CUDA context on every visible GPU, so hide the unused ones.
    if parent is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", parent)
    assert restrict_visible_gpus(trainer, sampler) == expect
    assert os.environ["CUDA_VISIBLE_DEVICES"] == visible


def test_launcher_restricts_gpus_before_anything_can_initialise_cuda():
    # The driver reads CUDA_VISIBLE_DEVICES once; transformers/peft imports already initialise it.
    import subprocess
    import sys
    code = "import sys, zroform.launch; assert not {'torch', 'transformers', 'zroform.api'} & set(sys.modules)"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_sglang_never_inherits_the_trainers_allocator_setting(monkeypatch, tmp_path):
    monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    seen = {}

    def fake_popen(cmd, env, **kw):
        seen.update(env)
        return _ExitedProc()

    monkeypatch.setattr("zroform.sampler.subprocess.Popen", fake_popen)
    s = SGLangSampler("m", 30100, 16, ["q_proj"], 2, 0.5, 1024, str(tmp_path / "sglang.log"), gpus=[0])
    with pytest.raises(SamplerError):
        asyncio.run(s.start(timeout=5))
    assert "PYTORCH_CUDA_ALLOC_CONF" not in seen
