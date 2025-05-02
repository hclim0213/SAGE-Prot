"""
Copyright (c) 2025 Hocheol Lim.
"""

import gc
import random
from typing import List

import numpy as np
import torch
from joblib import Parallel, delayed

from sage_prot.models.expert import (
    crossover,
    mutate,
)

class GeneticOperatorHandler:
    def __init__(
        self,
        crossover_type: str,
        mutation_type: str,
        mutation_initial_rate: float,
    ) -> None:
        self.mutation_initial_rate = mutation_initial_rate
        self.mutation_rate = mutation_initial_rate

        if crossover_type == "SEQUENCES":
            self.crossover_func = crossover
        else:
            raise ValueError(f"'crossover_type' {crossover_type} is invalid")

        if mutation_type == "SEQUENCES":
            self.mutate_func = mutate
        else:
            raise ValueError(f"'mutation_type' {mutation_type} is invalid")

    def query(
        self,
        query_size: int,
        apprentice_mean_similarity: float,
        mating_pool: List[str],
        pool: Parallel,
    ) -> List[str]:

        original_sequences = random.choices(mating_pool, k=2 * query_size)
        sequences_a, sequences_b = (
            original_sequences[:query_size],
            original_sequences[query_size:],
        )
        sequences = pool(
            delayed(self.reproduce_seqs)(sequence_a, sequence_b, self.mutation_rate)
            for sequence_a, sequence_b in zip(sequences_a, sequences_b)
        )

        sequences_list = list(filter(lambda sequence: sequence is not None, sequences))
        gc.collect()
        return sequences_list

    def reproduce_seqs(
        self, parent_a: str, parent_b: str, mutation_rate: float, num_trials: int = 10
    ) -> List[str]:
        
        for _ in range(num_trials):
            try:
                parent_seqs = [parent_a, parent_b]
                new_child = None
                
                proba = random.random()
                if proba <= 0.8:
                    new_child = self.crossover_func(parent_a, parent_b)
                    if new_child is not None:
                        new_child = self.mutate_func(new_child, mutation_rate)
                else:
                    if random.random() <= 0.5:
                        new_child = self.mutate_func(parent_a, mutation_rate)
                    else:
                        new_child = self.mutate_func(parent_b, mutation_rate)
                
                if new_child is not None and new_child not in parent_seqs:
                    break
                    
            except:
                continue
        
        new_seq = (
            new_child
            if new_child is not None
            else None
        )
        return new_seq

