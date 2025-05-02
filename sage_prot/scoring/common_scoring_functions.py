"""
Copyright (c) 2025 Hocheol Lim.
"""

from typing import Callable, List, Union

from sage_prot.scoring.score_modifier import ScoreModifier, MinGaussianModifier, MaxGaussianModifier, GaussianModifier
from sage_prot.scoring.scoring_function import MoleculewiseScoringFunction
from sage_prot.utils.math import arithmetic_mean, geometric_mean
from sage_prot.scoring.filters import clean_sequence

class SequenceScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, descriptor: Callable[[str], float], score_modifier: ScoreModifier = None, **kwargs) -> None:
        """
        Args:
            descriptor: molecular descriptors, such as the ones in descriptors.py
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)
        self.descriptor = descriptor
        self.kwargs = kwargs

    def raw_score(self, sequence: str) -> float:
        return self.score_seq(sequence)

    def score_seq(self, sequence: str) -> float:
        return self.descriptor(sequence, **self.kwargs)
    
class ReferenceScoringFunction(MoleculewiseScoringFunction):
    def __init__(self, target: str, descriptor: Callable[[str], float], score_modifier: ScoreModifier = None, **kwargs) -> None:
        """
        Args:
            descriptor: molecular descriptors, such as the ones in descriptors.py
            score_modifier: score modifier
        """
        super().__init__(score_modifier=score_modifier)
        self.target = target
        self.descriptor = descriptor
        self.kwargs = kwargs

    def raw_score(self, sequence: str) -> float:
        return self.score_seq(sequence)

    def score_seq(self, sequence: str) -> float:
        return self.descriptor(sequence, target=self.target, **self.kwargs)
    