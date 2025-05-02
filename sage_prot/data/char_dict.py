"""
Copyright (c) 2025 Hocheol Lim.
"""

import torch
from typing import Dict, List, Set, Union

PAD = " "
BEGIN = "j"
END = "\n"

class SequenceCharDictionary:
    def __init__(self, dataset: str, max_seq_len: int) -> None:
        self.max_seq_len = max_seq_len

        self.forbidden_symbols = get_forbidden_symbols(dataset)
        self.char_idx = get_char_idx(dataset)
        self.idx_char = {v: k for k, v in self.char_idx.items()}

        self.encode_dict = get_encode_dict(dataset)
        self.decode_dict = {v: k for k, v in self.encode_dict.items()}

    def is_allowed(self, sequences: str) -> bool:
        if len(sequences) > self.max_seq_len:
            return False

        for symbol in self.forbidden_symbols:
            if symbol in sequences:
                return False

        return True

    def encode(self, sequences: str) -> str:
        temp_sequences = sequences
        for symbol, token in self.encode_dict.items():
            temp_sequences = temp_sequences.replace(symbol, token)
        return temp_sequences

    def decode(self, sequences: str) -> str:
        temp_sequences = sequences
        for symbol, token in self.decode_dict.items():
            temp_sequences = temp_sequences.replace(symbol, token)
        return temp_sequences

    def get_char_num(self) -> int:
        return len(self.idx_char)

    @property
    def begin_idx(self) -> int:
        return self.char_idx[BEGIN]

    @property
    def end_idx(self) -> int:
        return self.char_idx[END]

    @property
    def pad_idx(self) -> int:
        return self.char_idx[PAD]

    @property
    def BEGIN(self):
        return BEGIN

    @property
    def END(self):
        return END

    @property
    def PAD(self):
        return PAD

    def matrix_to_sequences(self, array: torch.Tensor, seq_lengths: torch.Tensor):
        array_list = array.tolist()
        seqs = list(
            map(
                lambda item: self.vector_to_sequences(item[0], item[1]),
                zip(array_list, seq_lengths),  # type: ignore
            )
        )
        return seqs

    def vector_to_sequences(self, vec, seq_length):
        chars = list(map(self.idx_char.get, vec[:seq_length]))
        seq = "".join(chars)
        seq = self.decode(seq)
        return seq

def get_forbidden_symbols(dataset: str) -> Union[Dict, Set]:
    """
    Get forbidden symbols of the dataset
    """
    if dataset == "protein" or dataset == "benchmark":
        forbidden_symbols = {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "#",
            "%",
            "(",
            ")",
            "+",
            ".",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "=",
            "[",
            "]",
            "@",
            "/",
            "\\",
        }
    else:
        forbidden_symbols = {
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
            "#",
            "%",
            "(",
            ")",
            "+",
            ".",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "=",
            "[",
            "]",
            "@",
            "/",
            "\\",
        }
        #forbidden_symbols = set()

    return forbidden_symbols

def get_char_idx(dataset: str) -> Dict:
    if dataset == "protein" or dataset == "benchmark":
        char_idx = {
            PAD: 0,
            BEGIN: 1,
            END: 2,
            "A": 3,
            "B": 4,
            "C": 5,
            "D": 6,
            "E": 7,
            "F": 8,
            "G": 9,
            "H": 10,
            "I": 11,
            "J": 12,
            "K": 13,
            "L": 14,
            "M": 15,
            "N": 16,
            "O": 17,
            "P": 18,
            "Q": 19,
            "R": 20,
            "S": 21,
            "T": 22,
            "U": 23,
            "V": 24,
            "W": 25,
            "X": 26,
            "Y": 27,
            "Z": 28,
            "-": 29,
            "*": 30,
        }
    else:
        char_idx = {
            PAD: 0,
            BEGIN: 1,
            END: 2,
            "A": 3,
            "B": 4,
            "C": 5,
            "D": 6,
            "E": 7,
            "F": 8,
            "G": 9,
            "H": 10,
            "I": 11,
            "J": 12,
            "K": 13,
            "L": 14,
            "M": 15,
            "N": 16,
            "O": 17,
            "P": 18,
            "Q": 19,
            "R": 20,
            "S": 21,
            "T": 22,
            "U": 23,
            "V": 24,
            "W": 25,
            "X": 26,
            "Y": 27,
            "Z": 28,
            "-": 29,
            "*": 30,
        }
    return char_idx

def get_encode_dict(dataset: str) -> Dict:
    if dataset == "protein" or dataset == "benchmark":
        encode_dict = {}
    else:
        encode_dict = {}

    return encode_dict
