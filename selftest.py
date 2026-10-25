"""CPU checks. Does not load FLUX.2 or CountGD."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core.cases import build_eval_cases, build_train_cases, interleave_by_target_rounds
from .core.config import load_experiment_config
from .core.lora import LoRAManager
from .core.prompt_builder import build_prompt_pair
from .core.reward import RewardEvaluator, RewardResult, count_reward, is_exact, reward_gate
from .core.sampler import GRPOMember, _need_teacher
from .core.teacher import propose_teacher, teacher_velocity
from .core.transition import (
    apply_transition,
    ddim_mean,
    ddim_sde_std,
    gaussian_log_prob,
    gaussian_mean_kl,
    ode_mean,
    plan_sde_std,
)
from .losses.distillation import weighted_velocity_mse
from .losses.grpo import grpo_clip_loss, group_advantages, importance_ratio
from .train import comparison_table


class _Attn(nn.Module):
    def __init__(self, dim=8, fused=False):
        super().__init__()
        if fused:
            self.to_qkv_mlp_proj = nn.Linear(dim, dim * 4, bias=False)
            self.to_out = nn.Linear(dim, dim, bias=False)
        else:
            self.to_q = nn.Linear(dim, dim, bias=False)
            self.to_k = nn.Linear(dim, dim, bias=False)
            self.to_v = nn.Linear(dim, dim, bias=False)
            self.to_out = nn.ModuleList([nn.Linear(dim, dim, bias=False), nn.Identity()])
            self.add_q_proj = nn.Linear(dim, dim, bias=False)
            self.add_k_proj = nn.Linear(dim, dim, bias=False)
            self.add_v_proj = nn.Linear(dim, dim, bias=False)
            self.to_add_out = nn.Linear(dim, dim, bias=False)


class _Block(nn.Module):
    def __init__(self, fused=False):
        super().__init__()
        self.attn = _Attn(fused=fused)


class _TinyFlux(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer_blocks = nn.ModuleList([_Block(fused=False)])
        self.single_transformer_blocks = nn.ModuleList([_Block(fused=True)])


def run_selftest() -> None:
    cfg_a = load_experiment_config("distill")
    cfg_b = load_experiment_config("fast_grpo")
    assert cfg_a["method"] == "reward_gated_distillation"
    assert cfg_b["method"] == "structure_assisted_flow_grpo_fast"
    assert cfg_a["use_grpo"] is False and cfg_a["use_sde"] is False
    assert cfg_b["use_grpo"] is True and cfg_b["use_sde"] is True
    assert cfg_b["group_size"] == 4
    assert cfg_b["teacher_trigger"] == "half_inexact"
    assert cfg_b["sde"]["kernel"] == "ddim_cps"
    assert cfg_a["group_size"] is None
    assert cfg_a["branch_after_completed_steps"] == cfg_b["branch_after_completed_steps"] == 2
    assert cfg_a["lora"]["rank"] == cfg_b["lora"]["rank"] == 16
    assert cfg_a["model"]["path"] == cfg_b["model"]["path"]
    assert cfg_a["model"]["guidance"] == cfg_b["model"]["guidance"] == 4.0
    assert cfg_a["model"]["steps"] == cfg_b["model"]["steps"] == 50
    assert cfg_a["prompt"]["categories"] == cfg_b["prompt"]["categories"]
    rows = comparison_table(cfg_a, cfg_b)
    assert any(r["field"] == "core_loss" for r in rows)

    apple10 = build_prompt_pair("apple", 10)
    assert apple10.plain == (
        "A studio photograph of exactly ten red apples on a plain light gray "
        "surface against a plain light gray background. All objects are fully "
        "visible. No other objects. No text."
    )
    assert apple10.structured == (
        "A studio photograph of exactly ten red apples on a plain light gray "
        "surface against a plain light gray background. The red apples are "
        "arranged in two horizontal rows, with exactly five red apples in the "
        "back row and exactly five red apples in the front row. All objects "
        "are fully visible. No other objects. No text."
    )
    assert apple10.plain not in apple10.structured or "two horizontal rows" in apple10.structured
    assert "two horizontal rows" not in apple10.plain

    cases_a = build_train_cases(cfg_a)
    cases_b = build_train_cases(cfg_b)
    eval_a = build_eval_cases(cfg_a)
    assert len(cases_a) == 240
    assert len(cases_b) == 96
    assert len(eval_a) == 60
    assert sum(1 for c in cases_a if c.target in (9, 10)) == 120
    assert sum(1 for c in cases_b if c.target in (9, 10)) == 56
    train_seeds = {c.seed for c in cases_a}
    assert train_seeds.isdisjoint({c.seed for c in eval_a})

    assert math.isclose(count_reward(10, 10), 1.3)
    assert math.isclose(count_reward(15, 10), 1 / 6)
    assert reward_gate(1.3, 1 / 6, 0.5) == 1.0
    assert reward_gate(0.2, 1.3, 0.5) == 0.0
    assert is_exact(10, 10) and not is_exact(12, 10)

    v_old = torch.tensor([0.0, 1.0])
    v_p = torch.tensor([0.0, 1.0])
    v_s = torch.tensor([2.0, 3.0])
    v_t = teacher_velocity(v_old, v_s, v_p, gamma=1.0)
    assert torch.allclose(v_t, v_s)
    v_old2 = torch.tensor([5.0, 5.0])
    v_t2 = teacher_velocity(v_old2, v_s, v_p, gamma=1.0)
    assert torch.allclose(v_t2, torch.tensor([7.0, 7.0]))
    prop = propose_teacher(v_old2, v_s, v_p)
    assert torch.allclose(prop.delta, v_s - v_p)

    x = torch.zeros(1, 4, 4)
    v = torch.ones_like(x)
    sigma = torch.tensor([0.8])
    sigma_next = torch.tensor([0.6])
    ode = apply_transition(x, v, sigma, sigma_next, use_sde=False)
    assert torch.allclose(ode.next_latents.float(), ode_mean(x, v, sigma_next - sigma))
    assert ode.log_prob is None
    torch.manual_seed(0)
    sde = apply_transition(x, v, sigma, sigma_next, use_sde=True)
    std = plan_sde_std(sigma, sigma_next)
    lp = gaussian_log_prob(sde.next_latents, sde.mean, sde.std)
    assert torch.allclose(lp, sde.log_prob)
    # Recompute log-prob of the same action under the same mean → identical.
    lp2 = gaussian_log_prob(sde.next_latents, sde.mean, std)
    assert torch.allclose(lp2, sde.log_prob)
    rho = importance_ratio(sde.log_prob, sde.log_prob)
    assert torch.allclose(rho, torch.ones_like(rho), atol=1e-5)

    rewards = torch.tensor([1.3, 0.5, 0.5, 0.2])
    adv, ok = group_advantages(rewards)
    assert ok and abs(float(adv.mean())) < 1e-6
    zero, ok0 = group_advantages(torch.ones(4))
    assert not ok0 and torch.allclose(zero, torch.zeros(4))
    loss = grpo_clip_loss(torch.zeros(4), torch.zeros(4), adv, clip_epsilon=0.1)
    assert loss.ndim == 0

    student = torch.zeros(2, 3)
    teacher = torch.ones(2, 3)
    w = torch.tensor([1.0, 0.0])
    mse = weighted_velocity_mse(student, teacher, w)
    assert math.isclose(float(mse), 0.5, rel_tol=1e-5)
    single = weighted_velocity_mse(torch.zeros(1, 3), torch.ones(1, 3), torch.tensor([0.4]))
    assert math.isclose(float(single), 0.4, rel_tol=1e-5)
    kl = gaussian_mean_kl(teacher[:1], student[:1], torch.tensor([0.5]))
    expected = (1.0) / (2.0 * 0.25)
    assert math.isclose(float(kl), expected, rel_tol=1e-5)

    lora_cfg = {
        "lora": {
            "rank": 2,
            "alpha": 4,
            "dropout": 0.0,
            "dual_suffixes": [
                "attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0",
                "attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj", "attn.to_add_out",
            ],
            "single_suffixes": ["attn.to_qkv_mlp_proj", "attn.to_out"],
        }
    }
    tiny = _TinyFlux()
    mgr = LoRAManager(tiny, lora_cfg)
    assert mgr.n_trainable() > 0
    x = torch.randn(2, 8)
    mgr.set_mode("base")
    y_base = tiny.transformer_blocks[0].attn.to_q(x)
    mgr.set_mode("student")
    with torch.no_grad():
        mgr.layers[0].lora_B.weight.add_(0.25)
    y_student = tiny.transformer_blocks[0].attn.to_q(x)
    assert y_base.shape == y_student.shape
    assert not torch.allclose(y_base, y_student)
    mgr.snapshot_old()
    with torch.no_grad():
        mgr.layers[0].lora_B.weight.add_(0.5)
    y_new = tiny.transformer_blocks[0].attn.to_q(x)
    mgr.set_mode("old")
    y_old = tiny.transformer_blocks[0].attn.to_q(x)
    assert torch.allclose(y_old, y_student, atol=1e-5)
    assert not torch.allclose(y_new, y_student)
    for p in mgr.trainable_parameters():
        assert p.requires_grad

    # Simulate runtime: wrap after the host module has already moved.
    fp16_host = _TinyFlux().to(dtype=torch.float16)
    mgr_fp16 = LoRAManager(fp16_host, lora_cfg)
    layer = mgr_fp16.layers[0]
    assert layer.lora_A.weight.dtype == torch.float16
    assert layer.old_A.dtype == torch.float16
    x_fp16 = torch.randn(2, 8, dtype=torch.float16)
    mgr_fp16.set_mode("old")
    y_fp16 = fp16_host.transformer_blocks[0].attn.to_q(x_fp16)
    assert y_fp16.dtype == torch.float16
    mgr_fp16.set_mode("student")
    y_fp16_s = fp16_host.transformer_blocks[0].attn.to_q(x_fp16)
    assert y_fp16_s.dtype == torch.float16
    if torch.cuda.is_available():
        cuda_host = _TinyFlux().to("cuda")
        mgr_cuda = LoRAManager(cuda_host, lora_cfg)
        x_cuda = torch.randn(2, 8, device="cuda")
        for mode in ("base", "student", "old"):
            mgr_cuda.set_mode(mode)
            y_cuda = cuda_host.transformer_blocks[0].attn.to_q(x_cuda)
            assert y_cuda.device.type == "cuda"
        assert mgr_cuda.layers[0].old_A.device.type == "cuda"

    x = torch.zeros(1, 4, 4)
    v = torch.ones_like(x)
    sigma = torch.tensor([0.8])
    sigma_next = torch.tensor([0.6])
    dt = sigma_next - sigma
    euler = ode_mean(x, v, dt)
    ddim0 = ddim_mean(x, v, sigma, sigma_next, std=0.0)
    assert torch.allclose(ddim0, euler, atol=1e-6)
    std = ddim_sde_std(sigma, sigma_next, eta=0.3)
    assert float(std) <= float(sigma_next)
    t_next = float(sigma_next)
    s = float(std)
    assert math.isclose((t_next ** 2 - s ** 2) + s ** 2, t_next ** 2, rel_tol=1e-6)
    torch.manual_seed(1)
    sde = apply_transition(x, v, sigma, sigma_next, use_sde=True, eta=0.3)
    assert torch.allclose(sde.log_prob, gaussian_log_prob(sde.next_latents, sde.mean, sde.std))

    mixed_a = interleave_by_target_rounds(cases_a, 4, 60)
    assert {c.id for c in mixed_a} == {c.id for c in cases_a}
    round0 = mixed_a[:60]
    counts = {t: sum(1 for c in round0 if c.target == t) for t in (5, 6, 7, 8, 9, 10)}
    assert counts == {5: 6, 6: 9, 7: 9, 8: 6, 9: 15, 10: 15}
    mixed_b = interleave_by_target_rounds(cases_b, 4, 24)
    round0b = mixed_b[:24]
    counts_b = {t: sum(1 for c in round0b if c.target == t) for t in (5, 6, 7, 8, 9, 10)}
    assert counts_b == {5: 2, 6: 3, 7: 3, 8: 2, 9: 7, 10: 7}

    dummy = torch.zeros(1)
    def _m(exact, reliable=True):
        return GRPOMember(dummy, dummy, dummy, dummy, 1.3 if exact else 0.3, 8 if exact else 10, exact, reliable)

    assert not _need_teacher("half_inexact", [_m(True), _m(True), _m(True), _m(False)])
    assert _need_teacher("half_inexact", [_m(True), _m(True), _m(False), _m(False)])
    try:
        _need_teacher("not_a_trigger", [_m(False)] * 4)
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    calls = {"n": 0}

    def mock_detect(**kwargs):
        calls["n"] += 1
        if calls["n"] < 2:
            return RewardResult(-1, 0.0, False, False, reason="detect_failed:boom")
        return RewardResult(8, 1.3, True, True, attempts=1)

    ev = RewardEvaluator({"reward": {"backend": "mock", "max_retries": 2}}, mock_fn=mock_detect)
    recovered = ev.evaluate_with_retry(image_path=None, target=8, object_text="red apples")
    assert recovered.reliable and recovered.attempts == 2 and recovered.count == 8

    g = 4.0
    c_p = torch.tensor([1.0, 2.0])
    c_s = torch.tensor([3.0, 5.0])
    u = torch.tensor([0.25, -0.5])
    v_plain = u + g * (c_p - u)
    v_struct = u + g * (c_s - u)
    assert torch.allclose(v_struct - v_plain, g * (c_s - c_p))

    print("count_rl selftest ok", flush=True)


if __name__ == "__main__":
    run_selftest()
