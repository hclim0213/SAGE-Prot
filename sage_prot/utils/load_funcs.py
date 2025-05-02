"""
Copyright (c) 2025 Hocheol Lim.
"""

import logging
from typing import List, Optional, Union

import torch

from sage_prot.logger import CommandLineLogger, NeptuneLogger
from sage_prot.models.apprentice import LSTMGenerator, TransformerDecoderGenerator

from sage_prot.models.handlers import (
    GeneticOperatorHandler,
    LSTMGeneratorHandler,
    TransformerDecoderGeneratorHandler,
)

def load_logger(args, tags=None):
    if args.logger_type == "Neptune":
        logger = NeptuneLogger(args, tags)
    elif args.logger_type == "CommandLine":
        logger = CommandLineLogger(args)
    else:
        raise NotImplementedError

    return logger

def load_neural_apprentice(args):
    if args.model_type == "LSTM":
        neural_apprentice = LSTMGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    elif args.model_type == "TransformerDecoder":
        neural_apprentice = TransformerDecoderGenerator.load(load_dir=args.apprentice_load_dir, best=args.use_best)
    else:
        raise ValueError(f"{args.model_type} is not a valid model-type")

    return neural_apprentice


def load_apprentice_handler(model, optimizer, char_dict, max_sampling_batch_size, args):
    if args.model_type == "LSTM":
        apprentice_handler = LSTMGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    elif args.model_type == "TransformerDecoder":
        apprentice_handler = TransformerDecoderGeneratorHandler(
            model=model,
            optimizer=optimizer,
            char_dict=char_dict,
            max_sampling_batch_size=max_sampling_batch_size,
        )
    return apprentice_handler


def load_genetic_experts(
    expert_types: List[str],
    args,
) -> List[GeneticOperatorHandler]:
    experts = []
    for ge_type in expert_types:
        expert_handler = GeneticOperatorHandler(
            crossover_type=ge_type,
            mutation_type=ge_type,
            mutation_initial_rate=args.mutation_initial_rate,
        )
        experts.append(expert_handler)
    return experts


def load_generator(input_size: int, args):
    if args.model_type == "LSTM":
        generator = LSTMGenerator(
            input_size=input_size,
            hidden_size=args.hidden_size,
            output_size=input_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    elif args.model_type == "TransformerDecoder":
        generator = TransformerDecoderGenerator(  # type: ignore
            n_token=input_size,
            n_embed=args.embed_size,
            n_head=args.n_head,
            n_hidden=args.hidden_size,
            n_layers=args.n_layers,
            dropout=args.dropout,
        )
    return generator
