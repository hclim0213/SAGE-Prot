"""
Copyright (c) 2025 Hocheol Lim.
"""

from typing import List, Optional, Union

import torch
from sage_prot.utils.goal_directed_generator import GoalDirectedGenerator
from sage_prot.scoring.scoring_function import ScoringFunction
from joblib import Parallel
from tqdm import tqdm
import numpy as np
from sage_prot.memory import Recorder
from sage_prot.runners.trainer import Trainer

class Generator(GoalDirectedGenerator):
    def __init__(
        self,
        trainer: Trainer,
        recorder: Recorder,
        num_steps: int,
        device: torch.device,
        scoring_num_list: List[int],
        num_jobs: int,
        dataset_type: Optional[str] = None,
    ) -> None:
        self.trainer = trainer
        self.recorder = recorder
        self.num_steps = num_steps
        self.device = device
        self.scoring_num_list = scoring_num_list
        self.dataset_type = dataset_type

        self.pool = Parallel(n_jobs=num_jobs)
        
    def generate_optimized_molecules(
        self,
        scoring_function: ScoringFunction,
        number_molecules: int,
        starting_population: Optional[List[str]] = None,
    ) -> List[str]:
        self.trainer.init(
            scoring_function=scoring_function, device=self.device, pool=self.pool
        )
        
        for step in tqdm(range(self.num_steps)):
            sequences, scores = self.trainer.step(
                scoring_function=scoring_function, device=self.device, pool=self.pool
            )

            self.recorder.add_list(sequences=sequences, scores=scores)
            current_score = self.recorder.get_and_log_score()

        self.recorder.log_final()
        best_sequences, best_scores = self.recorder.get_topk(top_k=number_molecules)
        return best_sequences

