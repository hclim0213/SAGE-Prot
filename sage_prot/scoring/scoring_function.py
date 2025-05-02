"""
Copyright (c) 2025 Hocheol Lim.
"""
import logging
import numpy as np
from abc import abstractmethod
from typing import List, Optional, Union

from sage_prot.scoring.score_modifier import ScoreModifier, LinearModifier
from sage_prot.utils.math import geometric_mean

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class InvalidMolecule(Exception):
    pass

class ScoringFunction:
    """
    Base class for an objective function.

    In general, do not inherit directly from this class. Prefer `MoleculewiseScoringFunction` or `BatchScoringFunction`.
    """

    def __init__(self, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """
        self.score_modifier = score_modifier
        self.corrupt_score = -1.0

    @property
    def score_modifier(self):
        return self._score_modifier

    @score_modifier.setter
    def score_modifier(self, modifier: Optional[ScoreModifier]):
        self._score_modifier = LinearModifier() if modifier is None else modifier

    def modify_score(self, raw_score: float) -> float:
        return self._score_modifier(raw_score)

    @abstractmethod
    def score(self, sequences: str) -> float:
        """
        Score a single molecule as sequences
        """
        raise NotImplementedError

class MoleculewiseScoringFunction(ScoringFunction):
    """
    Objective function that is implemented by calculating the score molecule after molecule.
    Rather use `BatchScoringFunction` than this if your objective function can process a batch of molecules
    more efficiently than by trivially parallelizing the `score` function.

    Derived classes must only implement the `raw_score` function.
    """

    def __init__(self, score_modifier: ScoreModifier = None) -> None:
        """
        Args:
            score_modifier: Modifier to apply to the score. If None, will be LinearModifier()
        """
        super().__init__(score_modifier=score_modifier)

    def score(self, sequences: str) -> float:
        try:
            return self.modify_score(self.raw_score(sequences))
        except InvalidMolecule:
            return self.corrupt_score
        except Exception:
            logger.warning(f'Unknown exception thrown during scoring of {sequences}')
            return self.corrupt_score

    @abstractmethod
    def raw_score(self, sequences: str) -> float:
        """
        Get the objective score before application of the modifier.

        For invalid molecules, `InvalidMolecule` should be raised.
        For unsuccessful score calculations, `ScoreCannotBeCalculated` should be raised.
        """
        raise NotImplementedError
    
    @abstractmethod
    def score_seq(self, seq: str) -> float:
        """
        Calculate the sequence score

        """
        raise NotImplementedError

