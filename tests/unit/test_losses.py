"""Losses match Tinker's documented formulas (sums over tokens) and input-key rules."""

import pytest
import torch

from zroform import losses

lp = torch.tensor([-1.0, -0.5, -2.0])
q = torch.tensor([-1.2, -0.5, -1.0])
adv = torch.tensor([1.0, -2.0, 0.5])
rl = {"logprobs": q, "advantages": adv}
ratio = torch.exp(lp - q)


def test_cross_entropy_is_weighted_sum_of_negative_logprobs():
    w = torch.tensor([0.0, 1.0, 1.0])
    assert losses.compute("cross_entropy", lp, {"weights": w}, {}) == pytest.approx(2.5)


def test_importance_sampling():
    assert losses.compute("importance_sampling", lp, rl, {}) == pytest.approx(float(-(ratio * adv).sum()))


def test_ppo_clips_ratio_with_default_bounds():
    clipped = ratio.clamp(0.8, 1.2)
    want = -torch.min(ratio * adv, clipped * adv).sum()
    assert losses.compute("ppo", lp, rl, {}) == pytest.approx(float(want))


def test_cispo_stops_gradient_through_clipped_ratio():
    x = lp.clone().requires_grad_(True)
    losses.compute("cispo", x, rl, {"clip_low_threshold": 0.0, "clip_high_threshold": 1.1}).backward()
    # d/dlp of -(sg(clip(r)) * lp * A) is -clip(r) * A
    torch.testing.assert_close(x.grad, -(ratio.clamp(0.0, 1.1) * adv))


def test_dro_quadratic_penalty_uses_beta():
    want = -(lp * adv - 0.5 * 0.1 * (lp - q) ** 2).sum()
    assert losses.compute("dro", lp, rl, {"beta": 0.1}) == pytest.approx(float(want))


@pytest.mark.parametrize("loss_fn,keys", [
    ("importance_sampling", {"target_tokens", "logprobs", "advantages", "weights"}),  # extra key
    ("ppo", {"target_tokens", "logprobs"}),                                            # missing key
    ("cross_entropy", {"target_tokens", "advantages"}),
    ("nope", {"target_tokens"}),
])
def test_rejects_wrong_input_keys(loss_fn, keys):
    with pytest.raises(losses.LossInputError):
        losses.check_inputs(loss_fn, keys)


# ---- Identities implied by the documented definitions (independent of how we coded them) ----

def _grad(loss_fn, lp, q, adv, config=None):
    x = lp.clone().requires_grad_(True)
    loss = losses.compute(loss_fn, x, {"logprobs": q, "advantages": adv}, config or {})
    loss.backward()
    return float(loss.detach()), x.grad


A = torch.tensor([1.0, -2.0, 0.5, -0.25])
LP = torch.tensor([-1.0, -0.5, -2.0, -0.1])


def test_on_policy_is_ppo_and_cispo_all_reduce_to_the_policy_gradient():
    # Trainer == sampler (ratio 1): every ratio-based RL loss gives gradient -A, and IS loss = -sum(A).
    for fn, cfg in [("importance_sampling", None), ("ppo", None), ("ppo", {"clip_low_threshold": 0.9,
                    "clip_high_threshold": 1.1}), ("cispo", None)]:
        _, g = _grad(fn, LP, LP.clone(), A, cfg)
        torch.testing.assert_close(g, -A, msg=fn)
    assert _grad("importance_sampling", LP, LP.clone(), A)[0] == pytest.approx(float(-A.sum()))


def test_ppo_equals_importance_sampling_while_ratios_stay_inside_the_clip_range():
    q = LP - torch.tensor([0.05, -0.1, 0.15, -0.02])          # ratios within [0.8, 1.2]
    l_is, g_is = _grad("importance_sampling", LP, q, A)
    l_ppo, g_ppo = _grad("ppo", LP, q, A)
    assert l_ppo == pytest.approx(l_is)
    torch.testing.assert_close(g_ppo, g_is)


def test_ppo_stops_the_gradient_once_the_ratio_is_clipped_in_the_advantage_direction():
    q = LP - torch.tensor([0.5, -0.5, 0.0, 0.0])               # token 0: r>1.2 with A>0; token 1: r<0.8 with A<0
    _, g = _grad("ppo", LP, q, A)
    assert g[0] == 0 and g[1] == 0 and g[2] != 0


def test_cispo_gradient_matches_importance_sampling_when_nothing_is_clipped():
    q = LP - torch.tensor([0.3, -0.4, 0.2, 0.1])               # ratios in (0, 4): CISPO's default range
    _, g_is = _grad("importance_sampling", LP, q, A)
    _, g_cispo = _grad("cispo", LP, q, A)
    torch.testing.assert_close(g_cispo, g_is)


def test_cispo_caps_the_coefficient_at_the_high_threshold():
    q = LP - 3.0                                               # ratio e^3 ~ 20 > 4
    _, g = _grad("cispo", LP, q, A)
    torch.testing.assert_close(g, -4.0 * A)


def test_dro_with_zero_beta_is_the_plain_policy_gradient_and_beta_pulls_toward_the_sampler():
    q = LP - 0.3
    _, g0 = _grad("dro", LP, q, A, {"beta": 0.0})
    torch.testing.assert_close(g0, -A)
    _, g = _grad("dro", LP, q, A, {"beta": 0.5})
    torch.testing.assert_close(g, -(A - 0.5 * (LP - q)))       # d/dlp of -(lp*A - beta/2 (lp-q)^2)


def test_cross_entropy_gradient_is_minus_the_weights():
    w = torch.tensor([0.0, 1.0, 0.5, 2.0])
    x = LP.clone().requires_grad_(True)
    losses.compute("cross_entropy", x, {"weights": w}, {}).backward()
    torch.testing.assert_close(x.grad, -w)
