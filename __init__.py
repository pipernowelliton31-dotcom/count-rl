"""Structure-conditioned counting training for FLUX.2 Klein Base 9B.

Two methods share the same runtime, prompts, LoRA targets, teacher residual,
and CountGD reward:

* Version A: reward-gated velocity distillation (no SDE / GRPO)
* Version B: Flow-GRPO-Fast + reward-gated Gaussian KL (group size 4)
"""

from .core.config import load_experiment_config

__all__ = ["load_experiment_config"]
