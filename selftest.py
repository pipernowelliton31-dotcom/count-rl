"""CPU checks. Does not load FLUX.2 or CountGD."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .core.cases import build_eval_cases, build_train_cases
from .core.config import load_experiment_config
from .core.lora import LoRAManager
from .core.prompt_builder import build_prompt_pair
from .core.reward import count_reward, is_exact, reward_gate
from .core.teacher import propose_teacher, teacher_velocity
from .core.transition import apply_transition, gaussian_log_prob, gaussian_mean_kl, ode_mean, plan_sde_std
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
    assert math.isclose(float(mse), 1.0, rel_tol=1e-5)
    kl = gaussian_mean_kl(teacher[:1], student[:1], torch.tensor([0.5]))
    expected = (1.0) / (2.0 * 0.25)
    assert math.isclose(float(kl), expected, rel_tol=1e-5)

    tiny = _TinyFlux()
    mgr = LoRAManager(
        tiny,
        {
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
        },
    )
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

    print("count_rl selftest ok", flush=True)


if __name__ == "__main__":
    run_selftest()
