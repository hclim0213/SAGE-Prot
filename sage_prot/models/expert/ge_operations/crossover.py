"""
Copyright (c) 2025 Hocheol Lim.
"""

import random
from typing import Any, List, Optional
from Bio import Align
from Bio.Align import substitution_matrices
from sage_prot.scoring.filters import clean_sequence

def crossover(parent_a: str, parent_b: str, num_trials: int = 10) -> Optional[str]:
    clean_parent_a = clean_sequence(parent_a)
    clean_parent_b = clean_sequence(parent_b)
    
    parent_seqs = [clean_parent_a, clean_parent_b]
    
    for _ in range(num_trials):
        try:
            child_seq = None
            if random.random() <= 0.5:
                child_seq = sequence_crossover(clean_parent_a, clean_parent_b)
            else:
                child_seq = sequence_crossover(clean_parent_b, clean_parent_a)
    
            if child_seq is not None:
                clean_child_seq = clean_sequence(child_seq)
                if child_seq is not None and child_seq not in parent_seqs:
                    return child_seq
        except:
            continue

    return None
        

def sequence_crossover(
    parent_a: str, parent_b: str, num_trials: int = 10, mut_prob: float = 0.9
) -> Optional[str]:
    
    for _ in range(num_trials):
        try:
            child_seqs = []
            aligner = Align.PairwiseAligner()
            aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
            seq1, seq2 = aligner.align(parent_a, parent_b)[0]
            
            while True:
                if random.random() <= mut_prob:
                    temp_seq = ''
                    for a, b in zip(seq1, seq2):
                        temp_seq += random.choice([a, b])

                    child_seqs.append(clean_sequence(temp_seq))
                    mut_prob -= 0.1
                else:
                    break

            if len(child_seqs) > 0:
                child_seqs = list(set(child_seqs))
                return random.choice(child_seqs)
            
        except:
            continue
    
    return None

