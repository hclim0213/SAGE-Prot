"""
Copyright (c) 2025 Hocheol Lim.
"""

import os
from typing import List, Tuple
from itertools import islice
import numpy as np
from sage_prot.scoring.scoring_function import ScoringFunction
from sage_prot.scoring.filters import clean_sequence
from joblib import Parallel, delayed

from sage_prot.data.char_dict import SequenceCharDictionary
from concurrent.futures import ProcessPoolExecutor, TimeoutError

def score_wt_timeout(sequence, scoring_function, timeout=3600):
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(scoring_function.score, sequence)
            try:
                result = future.result(timeout=timeout)
                return result
            except TimeoutError:
                print(f"Timeout occurred for sequence: {sequence}")
                return -1
            except Exception as e:
                print(f"Error occurred for sequence: {sequence}, error: {str(e)}")
                return -1

def sequences_to_actions(char_dict: SequenceCharDictionary, seqs: List[str]):
    max_seq_length = char_dict.max_seq_len + 1
    enc_seqs = list(map(lambda seq: char_dict.encode(seq) + char_dict.END, seqs))
    actions = np.zeros((len(seqs), max_seq_length), dtype=np.int32)
    seq_lengths = np.zeros((len(seqs),), dtype=np.long)

    for i, enc_seq in list(enumerate(enc_seqs)):
        for c in range(len(enc_seq)):
            try:
                actions[i, c] = char_dict.char_idx[enc_seq[c]]
            except:
                print(char_dict.char_idx)
                print(enc_seq)
                print(enc_seq[c])
                assert False

        seq_lengths[i] = len(enc_seq)

    return actions, seq_lengths

def canonicalize_and_score_sequences(
    sequences: List[str],
    scoring_function: ScoringFunction,
    char_dict: SequenceCharDictionary,
    pool: Parallel,
) -> Tuple[List[str], List[float]]:
    canon_sequences = pool(
        delayed(lambda sequence: clean_sequence(sequence))(sequence)
        for sequence in sequences
    )

    canon_sequences = list(
        filter(
            lambda sequence: (sequence is not None) and char_dict.is_allowed(sequence),
            canon_sequences,
        )
    )
    
    canon_scores = Parallel(n_jobs=pool.n_jobs)(
        delayed(score_wt_timeout)(sequence, scoring_function) for sequence in canon_sequences
    )
    
    filted_sequences_and_scores = list(
        filter(
            lambda sequence_and_score: sequence_and_score[1]
            > scoring_function.scoring_function.corrupt_score,  # type: ignore
            zip(canon_sequences, canon_scores),
        )
    )

    canon_sequences, canon_scores = (
        map(list, zip(*filted_sequences_and_scores))  # type: ignore
        if len(filted_sequences_and_scores) > 0
        else ([], [])
    )

    return canon_sequences, canon_scores
