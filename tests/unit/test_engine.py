"""Per-model seq_id ordering and retry dedupe."""

import asyncio

import pytest

from zroform.engine import Engine, UserError
from zroform.futures import Futures, Result
from tests.conftest import TINY, make_profile


class RecordingTrainer:
    base_model = TINY

    def __init__(self):
        self.order, self.steps = [], {}

    def create_slot(self, slot, spec, seed=None):
        self.steps[slot] = 0

    def optim_step(self, slot, adam):
        self.order.append(adam["tag"])
        return {"grad_norm": 0.0}


@pytest.fixture
async def engine(tmp_path):
    from zroform.store import LocalStore
    prof = make_profile(tmp_path)
    eng = Engine(prof, RecordingTrainer(), sampler=None, store=LocalStore(prof.store_dir), futures=Futures(60))
    sid = eng.create_session("owner", None)
    mid = await eng.create_model("owner", {"session_id": sid, "base_model": TINY, "model_seq_id": 0,
                                     "lora_config": {"rank": 4}})
    return eng, mid


async def _done(eng, rid):
    return await eng.futures.wait(rid, 5)


async def test_runs_ops_in_seq_order_even_when_they_arrive_out_of_order(engine):
    eng, mid = engine
    r3 = eng.submit("owner", mid, 3, "optim_step", {"tag": 3})
    r2 = eng.submit("owner", mid, 2, "optim_step", {"tag": 2})
    await asyncio.sleep(0.05)
    assert eng.trainer.order == []                     # waiting for seq 1
    r1 = eng.submit("owner", mid, 1, "optim_step", {"tag": 1})
    for rid in (r1, r2, r3):
        assert isinstance(await _done(eng, rid), Result)
    assert eng.trainer.order == [1, 2, 3]


async def test_a_retried_seq_id_returns_the_same_request_and_runs_once(engine):
    eng, mid = engine
    first = eng.submit("owner", mid, 1, "optim_step", {"tag": 1})
    await _done(eng, first)
    assert eng.submit("owner", mid, 1, "optim_step", {"tag": 1}) == first
    assert eng.trainer.order == [1]


async def test_other_owners_cannot_submit_to_a_run(engine):
    eng, mid = engine
    with pytest.raises(UserError):
        eng.submit("someone-else", mid, 1, "optim_step", {"tag": 1})
