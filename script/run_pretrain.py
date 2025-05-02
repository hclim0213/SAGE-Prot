"""
Copyright (c) 2022 Hocheol Lim.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

import sys
sys.path.append('/workspace/_ext')

import argparse
import datetime
import random
from pathlib import Path

import torch
from torch.optim import Adam

from sage.data.char_dict import SequenceCharDictionary
from sage.data.dataset import load_pretrain_dataset
from sage.runners import PreTrainer
from sage.utils.load_funcs import load_apprentice_handler, load_generator, load_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="pretrain", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dataset", type=str, default="protein")
    parser.add_argument(
        "--dataset_path", type=str, default="/workspace/_ext/data/datasets/protein"
    )
    parser.add_argument(
        "--model_type", type=str, default="LSTM"
    )
    parser.add_argument(
        "--logger_type",
        type=str,
        choices=["Neptune", "CommandLine"],
        default="CommandLine",
    )
    parser.add_argument("--project_qualified_name", type=str, default="")
    parser.add_argument("--max_sequences_length", type=int, default=1024)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--optimizer", type=str, default='adam')
    parser.add_argument("--save_root", default="/workspace/_ext/data/pretrained_models/models/")
    parser.add_argument("--seed", type=int, default=404)
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--use_valid", action="store_true", default=False)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()
    
    # Create a unique experimental id based on the execution timestamp
    now = datetime.datetime.now()
    experiment_id = now.strftime("%y%m%d_%H%M")
    
    print(args.dataset)
    
    save_dir = (
        args.save_root
        + "{}/".format(args.dataset)
        + "{}/".format(args.model_type)
        + "{}/".format(experiment_id)
    )
    save_path = Path(save_dir)
    if not save_path.exists():
        save_path.mkdir(parents=True)
    print(save_path)
    
    # Prepare CUDA device if set
    if args.use_cuda:
        device = torch.device(0)
    else:
        device = torch.device("cpu")
    random.seed(args.seed)

    # Initialize logger
    tags = [args.dataset, experiment_id]
    logger = load_logger(args, tags)

    # Load the dataset and valide characters
    char_dict = SequenceCharDictionary(
        dataset=args.dataset, max_seq_len=args.max_sequences_length
    )
    
    train_dataset = load_pretrain_dataset(char_dict=char_dict, sequences_path=args.dataset_path, code='train')
    valid_dataset = None

    if args.use_valid:
        valid_dataset = load_pretrain_dataset(char_dict=char_dict, sequences_path=args.dataset_path, code='valid')
    
    input_size = max(char_dict.char_idx.values()) + 1
    generator = load_generator(input_size, args)
    generator.to(device)
    
    optimizer = Adam(params=generator.parameters(), lr=args.learning_rate)

    generator_handler = load_apprentice_handler(
        model=generator,
        optimizer=optimizer,
        char_dict=char_dict,
        max_sampling_batch_size=0,
        args=args,
    )
    
    trainer = PreTrainer(
        char_dict=char_dict,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        generator_handler=generator_handler,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        save_dir=save_dir,
        device=device,
        num_workers=args.num_workers,
        logger=logger,
    )
    trainer.pretrain()
