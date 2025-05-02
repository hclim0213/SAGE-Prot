"""
Copyright (c) 2025 Hocheol Lim.
"""

from pathlib import Path
from typing import List
from sage_prot.data.char_dict import SequenceCharDictionary

def load_pretrain_dataset(char_dict: SequenceCharDictionary, sequences_path: str, code='train') -> List[str]:
    
    sequences_path = sequences_path + '/' + code + '.txt'
    processed_dataset_path = (
        str(Path(sequences_path).with_suffix("")) + "_processed.seq"
    )

    if Path(processed_dataset_path).exists():
        with open(processed_dataset_path, "r") as file:
            processed_dataset = file.read().splitlines()

    else:
        with open(sequences_path, "r") as file:
            dataset = file.read().splitlines()

        processed_dataset = list(filter(char_dict.is_allowed, dataset))
        with open(processed_dataset_path, "w") as file:
            file.writelines("\n".join(processed_dataset))

    return processed_dataset


def load_dataset(char_dict: SequenceCharDictionary, sequences_path: str) -> List[str]:
    processed_dataset_path = (
        str(Path(sequences_path).with_suffix("")) + "_processed.seq"
    )

    if Path(processed_dataset_path).exists():
        with open(processed_dataset_path, "r") as file:
            processed_dataset = file.read().splitlines()

    else:
        with open(sequences_path, "r") as file:
            dataset = file.read().splitlines()

        processed_dataset = list(filter(char_dict.is_allowed, dataset))
        with open(processed_dataset_path, "w") as file:
            file.writelines("\n".join(processed_dataset))

    return processed_dataset