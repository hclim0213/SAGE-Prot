"""
Copyright (c) 2025 Hocheol Lim.
"""

import sys
import os

from sage_prot.scoring.common_scoring_functions import (
    ReferenceScoringFunction,
)
from sage_prot.utils.goal_directed_benchmark import GoalDirectedBenchmark
from sage_prot.utils.goal_directed_score_contributions import uniform_specification

from sage_prot.scoring.scoring_function import (
    MoleculewiseScoringFunction,
)

from sage_prot.scoring.descriptors import (
    score_identity,
    score_similarity,
)

class ThresholdedImprovementScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, objective, constraint, threshold, offset):
        super().__init__()
        self.objective = objective
        self.constraint = constraint
        self.threshold = threshold
        self.offset = offset

    def raw_score(self, sequences):
        score = (
            self.corrupt_score
            if (self.constraint.score(sequences) < self.threshold)
            else (self.objective.score(sequences) + self.offset)
        )
        return score

def identity(
    target: str,
    name: str
) -> GoalDirectedBenchmark:

    benchmark_name = f"{name} identity"
    scoring_function = ReferenceScoringFunction(
        target=target, descriptor=score_identity
    )
    specification = uniform_specification(1)
    
    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=scoring_function,
        contribution_specification=specification,
    )

def similarity(
    target: str,
    name: str
) -> GoalDirectedBenchmark:

    benchmark_name = f"{name} similarity"
    scoring_function = ReferenceScoringFunction(
        target=target, descriptor=score_similarity
    )
    specification = uniform_specification(1, 10, 100)
    
    return GoalDirectedBenchmark(
        name=benchmark_name,
        objective=scoring_function,
        contribution_specification=specification,
    )
