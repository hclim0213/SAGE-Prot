"""
Copyright (c) 2022 Hocheol Lim.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import argparse
import datetime
import random
from pathlib import Path

import torch
from torch.optim import Adam
from sage_prot.data import SequenceCharDictionary, load_dataset
from sage_prot.memory import MaxRewardPriorityMemory, Recorder
from sage_prot.runners import Trainer, Generator
from sage_prot.benchmark.general_benchmarks import load_benchmark

from sage.utils.load_funcs import (
    load_apprentice_handler,
    load_genetic_experts,
    load_logger,
    load_neural_apprentice,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="normal_runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="./results/",
        help="Path to save the final neural apprentice and other models",
    )
    parser.add_argument(
        "--benchmark_id",
        type=int,
        default=0,
        help="Determines which benchmark to run against.",
    )
    parser.add_argument(
        "--benchmark_type",
        type=str,
        default='benchmark',
        help="Determines which benchmark to run against.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="protein",
        help="Sets the starting dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="/workspace/_ext/data/datasets/protein",
        help="Path to the dataset. Should correspond to the chosen 'dataset'",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="LSTM",
        help="Chooses the type of model for the neural apprentice.",
    )
    parser.add_argument(
        "--explainer_type",
        type=str,
        default=None,
        help="The model type of the explainer model",
    )
    parser.add_argument(
        "--max_sequences_length",
        type=int,
        default=1024,
        help="The maximum allowed sequences string lenght.",
    )
    parser.add_argument(
        "--apprentice_load_dir",
        type=str,
        default="/workspace/_ext/data/pretrained_models/original_benchmarks/zinc",
        help="Path to the pretrained neural apprentice.",
    )
    parser.add_argument(
        "--explainer_load_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--genetic_experts", type=str, nargs="+", default="SEQUENCES"
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        default="CommandLine",
    )
    parser.add_argument("--project_qualified_name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--mutation_initial_rate", type=float, default=1e-1)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--num_keep", type=int, default=1024)
    parser.add_argument("--max_sampling_batch_size", type=int, default=1024)
    parser.add_argument("--apprentice_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--expert_sampling_batch_size", type=int, default=8192)
    parser.add_argument("--apprentice_training_batch_size", type=int, default=256)
    parser.add_argument("--apprentice_training_steps", type=int, default=8)
    parser.add_argument("--num_sequences_for_similarity", type=int, default=100)
    parser.add_argument("--num_jobs", type=int, default=8)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--sampling_strategy", type=str, default="softmax")
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--seed", type=int, default=404)
    
    args = parser.parse_args()

    # Create save dir for the neural apprentice and explainer
    now = datetime.datetime.now()
    experiment_id = now.strftime("%y%m%d_%H%M%S")
    save_dir = args.save_root + "{}/".format(experiment_id)
    
    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    print(save_path)
    
    # Prepare CUDA device if is set
    if args.use_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")

    random.seed(args.seed)

    # Get a logger
    tags = [args.dataset, args.benchmark_id, args.model_type, *args.genetic_experts]
    logger = load_logger(args, tags)

    # Load benchmark, character dictionary
    if args.benchmark_type == 'benchmark':
        benchmark, scoring_num_list = load_benchmark(args.benchmark_id)
    
    char_dict = SequenceCharDictionary(
        dataset=args.dataset, max_seq_len=args.max_sequences_length
    )

    # Prepare max-reward memory
    apprentice_memory = MaxRewardPriorityMemory()
    expert_memory = MaxRewardPriorityMemory()

    # Load neural apprentice
    neural_apprentice = load_neural_apprentice(args)
    neural_apprentice.to(device)
    neural_apprentice.train()

    optimizer = Adam(params=neural_apprentice.parameters(), lr=args.learning_rate)

    apprentice_handler = load_apprentice_handler(
        model=neural_apprentice,
        optimizer=optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=args.max_sampling_batch_size,
        args=args,
    )

    # Load the genetic experts
    expert_handlers = load_genetic_experts(
        args.genetic_experts,
        args=args,
    )

    # Load the gegl-trainer
    trainer = Trainer(
        apprentice_memory=apprentice_memory,
        expert_memory=expert_memory,
        apprentice_handler=apprentice_handler,
        expert_handlers=expert_handlers,
        char_dict=char_dict,
        num_keep=args.num_keep,
        apprentice_sampling_batch_size=args.apprentice_sampling_batch_size,
        expert_sampling_batch_size=args.expert_sampling_batch_size,
        sampling_strategy=args.sampling_strategy,
        apprentice_training_batch_size=args.apprentice_training_batch_size,
        apprentice_training_steps=args.apprentice_training_steps,
        num_sequences_for_similarity=args.num_sequences_for_similarity,
        logger=logger,
    )

    # Load recorder
    recorder = Recorder(
        scoring_num_list=scoring_num_list,
        logger=logger,
        record_filtered=args.record_filtered,
        save_dir=save_dir,
    )

    generator = Generator(
        trainer=trainer,
        recorder=recorder,
        num_steps=args.num_steps,
        device=device,
        dataset_type=args.dataset,
        scoring_num_list=scoring_num_list,
        num_jobs=args.num_jobs,
    )
    
    final_result = benchmark.assess_model(generator)
    logger.log_metric("benchmark_score", final_result.score)
    
    print("benchmark_score")
    print(final_result.score)
    print(final_result.optimized_molecules)
    
    # Save the neural apprentice and explainer if used
    neural_apprentice.save(save_dir)
