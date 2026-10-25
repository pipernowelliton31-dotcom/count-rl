"""Plain vs structured counting prompts.

Templates match `structured_exact_counting_5x6` in the 480-subset manifest.
The teacher prompt is the student prompt plus one layout sentence.
"""

from __future__ import annotations

from dataclasses import dataclass

NUMBER_WORDS = {
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
}

# Internal keys follow numerosity_1120; phrases follow the structured spec.
CATEGORIES = {
    "apple": {"object_phrase": "red apples", "countgd_text": "red apples"},
    "mug": {"object_phrase": "cups", "countgd_text": "cups"},
    "bottle": {"object_phrase": "bottles", "countgd_text": "bottles"},
    "tennis_ball": {"object_phrase": "tennis balls", "countgd_text": "tennis balls"},
    "toy_car": {"object_phrase": "toy cars", "countgd_text": "toy cars"},
}

LAYOUTS = {
    5: "1x5",
    6: "3+3",
    7: "4+3",
    8: "4+4",
    9: "3x3",
    10: "5+5",
}

SCENE = (
    "A studio photograph of exactly {word} {phrase} on a plain light gray "
    "surface against a plain light gray background."
)
VISIBILITY = "All objects are fully visible. No other objects. No text."


def structure_sentence(target: int, phrase: str) -> str:
    if target == 5:
        return (
            f"The {phrase} are arranged in a single horizontal row of exactly "
            f"five {phrase}."
        )
    if target == 6:
        return (
            f"The {phrase} are arranged in two horizontal rows, with exactly "
            f"three {phrase} in the back row and exactly three {phrase} in the front row."
        )
    if target == 7:
        return (
            f"The {phrase} are arranged in two horizontal rows, with exactly "
            f"four {phrase} in the back row and exactly three {phrase} in the front row."
        )
    if target == 8:
        return (
            f"The {phrase} are arranged in two horizontal rows, with exactly "
            f"four {phrase} in the back row and exactly four {phrase} in the front row."
        )
    if target == 9:
        return (
            f"The {phrase} are arranged in a 3-by-3 grid, with exactly three rows "
            f"and exactly three {phrase} in each row."
        )
    if target == 10:
        return (
            f"The {phrase} are arranged in two horizontal rows, with exactly "
            f"five {phrase} in the back row and exactly five {phrase} in the front row."
        )
    raise ValueError(f"No frozen layout template for target={target}")


def plain_prompt(target: int, phrase: str) -> str:
    return f"{SCENE.format(word=NUMBER_WORDS[target], phrase=phrase)} {VISIBILITY}"


def structured_prompt(target: int, phrase: str) -> str:
    return (
        f"{SCENE.format(word=NUMBER_WORDS[target], phrase=phrase)} "
        f"{structure_sentence(target, phrase)} {VISIBILITY}"
    )


@dataclass(frozen=True)
class PromptPair:
    category: str
    target: int
    object_phrase: str
    countgd_text: str
    layout: str
    plain: str
    structured: str


def build_prompt_pair(category: str, target: int) -> PromptPair:
    if category not in CATEGORIES:
        raise KeyError(f"Unknown category {category!r}")
    if target not in NUMBER_WORDS:
        raise KeyError(f"Unsupported target {target}")
    meta = CATEGORIES[category]
    phrase = meta["object_phrase"]
    return PromptPair(
        category=category,
        target=int(target),
        object_phrase=phrase,
        countgd_text=meta["countgd_text"],
        layout=LAYOUTS[int(target)],
        plain=plain_prompt(int(target), phrase),
        structured=structured_prompt(int(target), phrase),
    )
