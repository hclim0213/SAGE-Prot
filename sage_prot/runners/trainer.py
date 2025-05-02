"""
Copyright (c) 2025 Hocheol Lim.
"""

import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from sage_prot.scoring.scoring_function import ScoringFunction
from joblib import Parallel
from torch_geometric.data import Batch

from sage_prot.data import SequenceCharDictionary
from sage_prot.logger.abstract_logger import AbstractLogger
from sage_prot.memory import FragmentLibrary, MaxRewardPriorityMemory
from sage_prot.models import (
    AbstractGeneratorHandler,
    GeneticOperatorHandler,
)

from sage_prot.models.apprentice import LSTMGenerator, TransformerDecoderGenerator
from sage_prot.utils.featurizer import CanonicalFeaturizer
from sage_prot.utils.sampling_handler import SamplingHandler
from sage_prot.utils.sequences import canonicalize_and_score_sequences
from sage_prot.scoring.filters import clean_sequence
from torch.utils.data import DataLoader, Dataset

class SEQDataset(Dataset):
    def __init__(self, seq_list):
        self.seq_list = seq_list
    
    def __len__(self):
        return len(self.seq_list)
    
    def __getitem__(self, idx):
        return self.seq_list[idx]

class Trainer:
    def __init__(
        self,
        apprentice_memory: MaxRewardPriorityMemory,
        expert_memory: MaxRewardPriorityMemory,
        apprentice_handler: AbstractGeneratorHandler,
        expert_handlers: List[GeneticOperatorHandler],
        char_dict: SequenceCharDictionary,
        num_keep: int,
        apprentice_sampling_batch_size: int,
        expert_sampling_batch_size: int,
        sampling_strategy: str,
        apprentice_training_batch_size: int,
        apprentice_training_steps: int,
        num_sequences_for_similarity: int,
        logger: AbstractLogger,
    ) -> None:
        self.apprentice_memory = apprentice_memory
        self.apprentice_mean_similarity = 1.0
        self.expert_memory = expert_memory

        self.apprentice_handler = apprentice_handler
        self.expert_handlers = expert_handlers

        self.char_dict = char_dict
        self.num_keep = num_keep
        self.apprentice_sampling_batch_size = apprentice_sampling_batch_size
        self.expert_sampling_batch_size = expert_sampling_batch_size
        self.apprentice_training_batch_size = apprentice_training_batch_size
        self.apprentice_training_steps = apprentice_training_steps

        num_experts = len(self.expert_handlers)
        self.num_experts = num_experts
        self.sampling_handler = SamplingHandler(
            num_experts, expert_sampling_batch_size, sampling_strategy
        )
        self.partial_query_sizes = (
            [  # Initialize the query sizes to uniform distribution at the beginning
                int(self.expert_sampling_batch_size / self.num_experts)
            ]
            * self.num_experts
        )

        self.logger = logger
        self.init_sequences = init_sequences
        self.num_sequences_for_similarity = num_sequences_for_similarity

    def init(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ) -> None:
        if len(self.init_sequences) > 0:  # type: ignore
            sequences, scores = canonicalize_and_score_sequences(
                sequences=self.init_sequences,  # type: ignore
                scoring_function=scoring_function,
                char_dict=self.char_dict,
                pool=pool,
            )
            self.apprentice_memory.add_list(sequences=sequences, scores=scores)
            self.expert_memory.add_list(sequences=sequences, scores=scores)

    def step(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ) -> Tuple[List[str], List[float]]:
        # Generate and record sequences from apprentice
        apprentice_sequences, apprentice_scores = self._update_memory_by_apprentice(
            scoring_function, device, pool
        )
        (
            best_apprentice_sequences,
            best_apprentice_scores,
            _,
        ) = self.apprentice_memory.get_elements()
        
        # Generate and record sequences from expert
        expert_sequences, expert_scores = self._update_memory_by_expert(
            scoring_function, device, pool
        )
        
        loss, fit_size = self._train_apprentice_step(device)
        self.logger.log_metric("loss_apprentice", loss)
        self.logger.log_metric("fit_size", fit_size)

        sequences = apprentice_sequences + expert_sequences
        scores = apprentice_scores + expert_scores

        return sequences, scores

    def _update_memory_by_apprentice(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ):
        
        with torch.no_grad():
            self.apprentice_handler.model.eval()
            
            try:
                context_sequences = self._get_all_sequences_from_memory()
            except:
                context_sequences = None

            sequences, _, _, _ = self.apprentice_handler.sample(
                num_samples=self.apprentice_sampling_batch_size,
                context_sequences=context_sequences,
                device=device,
            )
        
        canon_sequences, canon_scores = canonicalize_and_score_sequences(
            sequences=sequences,
            scoring_function=scoring_function,
            char_dict=self.char_dict,
            pool=pool,
        )
        
        self.apprentice_memory.add_list(sequences=canon_sequences, scores=canon_scores)
        self.apprentice_memory.squeeze_by_rank(top_k=self.num_keep)

        return canon_sequences, canon_scores

    def _update_memory_by_expert(
        self, scoring_function: ScoringFunction, device: torch.device, pool: Parallel
    ):
        
        apprentice_sequences, _, _ = self.apprentice_memory.sample_batch(
            self.expert_sampling_batch_size
        )
        canon_sequences: List[str] = []
        canon_scores: List[float] = []

        for expert_idx in range(self.num_experts):
            expert = self.expert_handlers[expert_idx]
            mating_pool = apprentice_sequences
            
            query_sequences = expert.query(
                query_size=self.partial_query_sizes[expert_idx],
                apprentice_mean_similarity=self.apprentice_mean_similarity,
                mating_pool=mating_pool,
                pool=pool,
            )
            
            partial_sequences, partial_scores = canonicalize_and_score_sequences(
                sequences=query_sequences,
                scoring_function=scoring_function,
                char_dict=self.char_dict,
                pool=pool,
            )
            
            self.expert_memory.add_list(
                sequences=partial_sequences, scores=partial_scores, expert_id=expert_idx
            )
            canon_sequences += partial_sequences
            canon_scores += partial_scores

        self.logger.log_metric("mutation_rate", self.expert_handlers[0].mutation_rate)
        self.expert_memory.squeeze_by_rank(top_k=self.num_keep)

        expert_ratios = [
            query_size / self.expert_sampling_batch_size
            for query_size in self.partial_query_sizes
        ]
        self.logger.log_values("expert_ratios", expert_ratios)

        # Update the partial query sizes for the next round
        _, _, expert_ids = self.expert_memory.get_elements()
        self.partial_query_sizes = self.sampling_handler.calculate_partial_query_size(
            expert_ids
        )

        return canon_sequences, canon_scores

    def _train_apprentice_step(self, device: torch.device) -> Tuple[float, int]:
        average_loss = 0.0

        all_sequences = self._get_all_sequences_from_memory()
        self.apprentice_handler.model.train()  # type: ignore
        
        dataset = SEQDataset(all_sequences)
        dataloader = DataLoader(dataset, batch_size=self.apprentice_training_batch_size, shuffle=True)
        
        for _ in range(self.apprentice_training_steps):
            for seq_batch in dataloader:
                loss = self.apprentice_handler.train_on_batch(sequences=seq_batch, device=device)
                average_loss += loss / len(dataloader)
        
        fit_size = len(all_sequences)

        return average_loss, fit_size

    def _get_all_sequences_from_memory(self) -> List[str]:
        apprentice_sequences, _, _ = self.apprentice_memory.get_elements()
        expert_sequences, _, _ = self.expert_memory.get_elements()
        all_sequences = list(set(apprentice_sequences + expert_sequences))
        return all_sequences