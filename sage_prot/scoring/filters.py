"""
Copyright (c) 2025 Hocheol Lim.
"""

import signal
import random
from typing import Optional, List, Iterable, Collection, Tuple, Any, Set
from sage_prot.utils.data import remove_duplicates

def handler(signum, frame):
    raise Exception()

def clean_sequence(seq):
    cleaned_seq = ""
    for aa in seq:
        if aa == "B":
            cleaned_seq += random.choice(["D", "N"])
        elif aa == "J":
            cleaned_seq += random.choice(["I", "L"])
        elif aa == "O":
            cleaned_seq += "K"
        elif aa == "U":
            cleaned_seq += "M"
        elif aa == "X":
            cleaned_seq += random.choice(list("ACDEFGHIKLMNPQRSTVWY"))
        elif aa == "Z":
            cleaned_seq += random.choice(["E", "Q"])
        elif aa in ["-", "*"]:
            continue
        else:
            cleaned_seq += aa
            
    return cleaned_seq

