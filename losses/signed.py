"""Optional signed-v2 diagnostic objective; NOT used by the default joint runner."""
from __future__ import annotations
import math
import torch

def confidence_signed_velocity_mse(student: torch.Tensor, old: torch.Tensor,
                                   teacher: torch.Tensor, q: float) -> torch.Tensor:
    """|q| * MSE(student, old + sign(q)*(teacher-old)), not MSE(student,old+q*d).

    Stop-gradient on teacher / old; q==0 produces exactly zero gradient.
    Negative Teacher direction is an unvalidated experimental hypothesis.
    """
    q=float(q)
    if not math.isfinite(q) or not -1 <= q <= 1:
        raise ValueError("q must be finite and within [-1,1]")
    if q == 0:
        return 0.0 * student.float().square().mean()
    target = old.float().detach() + math.copysign(1.0,q)*(teacher.float().detach()-old.float().detach())
    return abs(q)*torch.nn.functional.mse_loss(student.float(),target)
