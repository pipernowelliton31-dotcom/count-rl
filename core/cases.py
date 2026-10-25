"""Stratified train/eval cases. Quotas overweight 9 and 10; eval seeds are held out."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterator

from .prompt_builder import CATEGORIES, PromptPair, build_prompt_pair


@dataclass(frozen=True)
class TrainingCase:
    id: str
    category: str
    target: int
    seed: int
    split: str
    object_phrase: str
    countgd_text: str
    layout: str
    plain_prompt: str
    structured_prompt: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _int_quota(raw: dict[Any, Any]) -> dict[int, int]:
    return {int(k): int(v) for k, v in raw.items()}


def _case(category: str, target: int, seed: int, split: str) -> TrainingCase:
    pair: PromptPair = build_prompt_pair(category, target)
    return TrainingCase(
        id=f"{category}_n{target:02d}_s{seed:010d}",
        category=category,
        target=target,
        seed=int(seed),
        split=split,
        object_phrase=pair.object_phrase,
        countgd_text=pair.countgd_text,
        layout=pair.layout,
        plain_prompt=pair.plain,
        structured_prompt=pair.structured,
    )


def iter_quota_assignments(categories: list[str], quota: dict[int, int], seed_start: int) -> Iterator[TrainingCase]:
    """Round-robin categories inside each target quota so 8 cannot dominate a category."""
    seed = int(seed_start)
    for target in sorted(quota):
        n = quota[target]
        for i in range(n):
            category = categories[i % len(categories)]
            yield _case(category, target, seed, "train")
            seed += 1


def build_train_cases(cfg: dict[str, Any]) -> list[TrainingCase]:
    prompt_cfg = cfg["prompt"]
    budget = cfg["budget"]
    categories = list(prompt_cfg["categories"])
    quota = _int_quota(budget["quota"])
    expected = int(budget.get("n_train_cases") or budget.get("n_train_groups") or sum(quota.values()))
    cases = list(iter_quota_assignments(categories, quota, int(prompt_cfg["train_seed_start"])))
    if len(cases) != expected:
        raise ValueError(f"Quota produced {len(cases)} cases, config expected {expected}")
    if sum(quota.values()) != len(cases):
        raise ValueError("Quota does not sum to the case list")
    return cases


def build_eval_cases(cfg: dict[str, Any]) -> list[TrainingCase]:
    prompt_cfg = cfg["prompt"]
    categories = list(prompt_cfg["categories"])
    targets = [int(t) for t in prompt_cfg["targets"]]
    seeds = [int(s) for s in prompt_cfg["eval_seeds"]]
    cases = [_case(cat, target, seed, "eval") for seed in seeds for target in targets for cat in categories]
    expected = int(cfg.get("eval", {}).get("n_images", len(cases)))
    if len(cases) != expected:
        raise ValueError(f"Eval set has {len(cases)} images, config expected {expected}")
    train_seeds = {c.seed for c in build_train_cases(cfg)}
    overlap = train_seeds.intersection(set(seeds))
    if overlap:
        raise ValueError(f"Eval seeds overlap training seeds: {sorted(overlap)}")
    return cases


def unique_prompts(cases: list[TrainingCase]) -> list[str]:
    texts = [""]
    for case in cases:
        texts.append(case.plain_prompt)
        texts.append(case.structured_prompt)
    return list(dict.fromkeys(texts))
