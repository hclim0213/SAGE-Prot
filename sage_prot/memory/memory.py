"""
Copyright (c) 2025 Hocheol Lim.
"""

import random
from functools import total_ordering
from itertools import product
from typing import Any, List, Optional, Tuple

import numpy as np

@total_ordering
class StorageElement:
    def __init__(
        self,
        sequence: str,
        score: float,
        expert_id: Optional[int] = None,
    ):
        self.sequence = sequence
        self.score = score
        self.expert_id = expert_id

    def __eq__(self, other):
        return np.isclose(self.score, other.score)

    def __lt__(self, other):
        return self.score < other.score

    def __hash__(self):
        return hash(self.sequence)


class MaxRewardPriorityMemory:
    def __init__(
        self,
    ) -> None:
        self.elements: List[StorageElement] = []

    def __len__(self) -> int:
        return len(self.elements)

    def add_list(
        self,
        sequences: List[str],
        scores: List[float],
        expert_id: Optional[int] = None,
    ) -> None:
        new_elements = [
            StorageElement(
                sequence=sequence,
                score=score,
                expert_id=expert_id,
            )
            for sequence, score in zip(sequences, scores)
        ]

        self.elements.extend(new_elements)
        self.elements = list(set(self.elements))

    def get_elements(
        self,
    ) -> Tuple[List[str], List[float], List[Any]]:
        return unravel_elements(self.elements)

    def squeeze_by_rank(self, top_k: int) -> None:
        top_k = min(top_k, len(self.elements))
        self.elements = sorted(self.elements, reverse=True)[:top_k]

    def sample_batch(self, batch_size: int) -> Tuple[List[str], List[float], List[Any]]:
        sampled_elements = random.choices(population=self.elements, k=batch_size)
        return unravel_elements(sampled_elements)


def unravel_elements(
    elements: List[StorageElement],
) -> Tuple[List[str], List[float], List[Any]]:
    return tuple(  # type: ignore
        map(
            list,
            zip(
                *[
                    (element.sequence, element.score, element.expert_id)
                    for element in elements
                ]
            ),
        )
    )
