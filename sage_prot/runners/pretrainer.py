"""
Copyright (c) 2025 Hocheol Lim.
"""

from typing import List, Optional

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sage_prot.data.char_dict import SequenceCharDictionary
from sage_prot.logger.abstract_logger import AbstractLogger
from sage_prot.models.handlers import AbstractGeneratorHandler
from sage_prot.utils.sequences import sequences_to_actions

class PreTrainer:
    def __init__(
        self,
        char_dict: SequenceCharDictionary,
        train_dataset: List[str],
        generator_handler: AbstractGeneratorHandler,
        num_epochs: int,
        batch_size: int,
        save_dir: str,
        num_workers: int,
        device: torch.device,
        logger: AbstractLogger,
        valid_dataset: Optional[List[str]] = None,
    ):
        self.generator_handler = generator_handler
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.device = device
        self.logger = logger
        
        action_dataset, _ = sequences_to_actions(char_dict=char_dict, seqs=train_dataset)
        action_dataset_ten = TensorDataset(torch.LongTensor(action_dataset))  # type: ignore
        self.dataset_loader: DataLoader = DataLoader(
            dataset=action_dataset_ten,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        
        self.valid_dataset = valid_dataset
        
        if valid_dataset is not None:
            self.best_valid_loss = float('inf')
        
            action_dataset_valid, _valid = sequences_to_actions(char_dict=char_dict, seqs=valid_dataset)
            action_dataset_ten_valid = TensorDataset(torch.LongTensor(action_dataset_valid))  # type: ignore
            self.dataset_loader_valid: DataLoader = DataLoader(
                dataset=action_dataset_ten_valid,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        

    def pretrain(self):
        
        if self.valid_dataset is not None:
            print('epoch', '\t', 'train_loss', '\t', 'valid_loss', '\t', 'best_valid')
        else:
            print('epoch', '\t', 'train_loss')
        
        for epoch in tqdm(range(self.num_epochs)):
            train_loss = []
            for actions in self.dataset_loader:
                loss = self.generator_handler.train_on_action_batch(
                    actions=actions[0], device=self.device
                )
                
                train_loss.append(loss)
                self.logger.log_metric("loss", loss)
            
            train_loss_ = sum([float(i) for i in train_loss]) / len(train_loss)
            
            if self.valid_dataset is not None:
                valid_loss = []
                best_valid = False
                for actions in self.dataset_loader_valid:
                    loss_valid = self.generator_handler.valid_on_action_batch(
                        actions=actions[0], device=self.device
                    )
                    valid_loss.append(loss_valid)
                    
                valid_loss_ = sum([float(i) for i in valid_loss]) / len(valid_loss)
                if valid_loss_ <= self.best_valid_loss:
                    self.best_valid_loss = valid_loss_
                    best_valid = True
            
            if self.valid_dataset is not None:
                print(epoch, '\t', train_loss_, '\t', valid_loss_, '\t', best_valid)
                if best_valid:
                    self.generator_handler.save(self.save_dir, best=True)
            else:
                print(epoch, '\t', train_loss_)
            
            self.generator_handler.save(self.save_dir)
