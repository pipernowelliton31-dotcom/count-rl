"""Structure residual teacher. Same x_s, freeze Base, residual-correct the student."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def structure_delta(v_base_structured, v_base_plain):
    return v_base_structured - v_base_plain


def teacher_velocity(v_old, v_base_structured, v_base_plain, gamma: float = 1.0):
    """v_T = v_old + gamma * (v_0^S - v_0^P).

    All three velocities must be the same CFG convention (guided or conditional).
    At initialization v_old == v_0^P, so v_T == v_0^S.
    """
    return v_old + float(gamma) * structure_delta(v_base_structured, v_base_plain)


@dataclass
class TeacherProposal:
    v_teacher: Any
    v_old: Any
    v_base_plain: Any
    v_base_structured: Any
    delta: Any
    gamma: float


def propose_teacher(v_old, v_base_structured, v_base_plain, gamma: float = 1.0) -> TeacherProposal:
    delta = structure_delta(v_base_structured, v_base_plain)
    v_t = v_old + float(gamma) * delta
    return TeacherProposal(
        v_teacher=v_t,
        v_old=v_old,
        v_base_plain=v_base_plain,
        v_base_structured=v_base_structured,
        delta=delta,
        gamma=float(gamma),
    )
