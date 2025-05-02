"""
Copyright (c) 2025 Hocheol Lim.
"""
import os
from typing import Union, List
from sage_prot.scoring.filters import (
    clean_sequence,
)

def score_length(seq: str, length: int) -> float:
    difference = abs(len(seq)-length)
    
    score = 1. - difference/length
    
    return score

def score_identity(seq: str, target: str) -> float:
    from Bio.Align import PairwiseAligner, substitution_matrices
    
    try:
        aligner = PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        
        clean_ref = clean_sequence(target)
        clean_seq = clean_sequence(seq)
        
        alignments = aligner.align(clean_ref, clean_seq)
        top_alignment = alignments[0]
        
        aligned_target, aligned_query, target, query = top_alignment[0], top_alignment[1], top_alignment.target, top_alignment.query
        
        num_identical = sum(a == b and a != '-' for a, b in zip(aligned_target, aligned_query))
        #num_similar = sum(1 for a, b in zip(aligned_target, aligned_query) if a != '-' and b != '-' and aligner.substitution_matrix[(a, b)] > 0)
        
        length_alignment = len(aligned_target)  # Aligned length
        
        identity = num_identical / length_alignment
        #similarity = num_similar / length_alignment
        coverage_target = (len(target) / len(aligned_target.replace('-', '')))
        coverage_query = (len(query) / len(aligned_query.replace('-', '')))
    
        score = identity * coverage_target * coverage_query
        
        return float(score)
    except:
        return -1000

def score_similarity(seq: str, target: str) -> float:
    from Bio.Align import PairwiseAligner, substitution_matrices
    
    try:
        aligner = PairwiseAligner()
        aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
        
        clean_ref = clean_sequence(target)
        clean_seq = clean_sequence(seq)
        
        alignments = aligner.align(clean_ref, clean_seq)
        top_alignment = alignments[0]
        
        aligned_target, aligned_query, target, query = top_alignment[0], top_alignment[1], top_alignment.target, top_alignment.query
        
        #num_identical = sum(a == b and a != '-' for a, b in zip(aligned_target, aligned_query))
        num_similar = sum(1 for a, b in zip(aligned_target, aligned_query) if a != '-' and b != '-' and aligner.substitution_matrix[(a, b)] > 0)
        
        length_alignment = len(aligned_target)  # Aligned length
        
        #identity = num_identical / length_alignment
        similarity = num_similar / length_alignment
        coverage_target = (len(target) / len(aligned_target.replace('-', '')))
        coverage_query = (len(query) / len(aligned_query.replace('-', '')))
    
        score = similarity * coverage_target * coverage_query
        
        return float(score)
    except:
        return -1000

