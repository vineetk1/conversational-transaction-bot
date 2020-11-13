'''
Vineet Kumar, sioom.ai
'''

import torch
from torch.utils.data import Dataset, TensorDataset, RandomSampler, DataLoader
import pytorch_lightning as pl
import logging
import sys
from typing import List, Dict
import pickle

logger = logging.getLogger(__name__)


def collate(examples: List[List[int]]):
    logger.debug('')
    return examples


class ctbData(pl.LightningDataModule):
    def __init__(self, args):
        logger.debug('')
        super().__init__()
        self.args = args

    def prepare_data(self) -> int:
        logger.debug('')
        if self.args.tokenizer == "gpt2":
            from utils.defaultFormat_to_gpt2Format import \
                    defaultFormat_to_gpt2Format
            data_info = defaultFormat_to_gpt2Format(self.args)
            for name, f_path in data_info['f_paths'].items():
                with f_path.open('rb') as file:
                    if name == 'train':
                        self.train_data = ctbDataset(pickle.load(file))
                    elif name == 'valid':
                        self.valid_data = ctbDataset(pickle.load(file))
                    elif name == 'test':
                        self.test_data = ctbDataset(pickle.load(file))
                    else:
                        assert False
            return data_info['len_tokenizer']
        else:
            logger.critical(f'unknown tokenizer: {self.args.tokenizer}')
            sys.exit()

    def setup(self):
        logger.debug('')

    def train_dataloader(self):
        logger.debug('')
        return DataLoader(self.train_data,
                          batch_size=8,
                          shuffle=False,
                          sampler=RandomSampler(self.train_data),
                          batch_sampler=None,
                          num_workers=0,
                          collate_fn=collate,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def val_dataloader(self):
        logger.debug('')
        return DataLoader(self.valid_data,
                          batch_size=8,
                          shuffle=False,
                          sampler=RandomSampler(self.valid_data),
                          batch_sampler=None,
                          num_workers=0,
                          collate_fn=collate,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def test_dataloader(self):
        logger.debug('')
        return DataLoader(self.test_data,
                          batch_size=8,
                          shuffle=False,
                          sampler=RandomSampler(self.test_data),
                          batch_sampler=None,
                          num_workers=0,
                          collate_fn=collate,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)


class ctbDataset(Dataset):
    # example = feature plus label
    def __init__(self, features: List[List[int]]):
        logger.debug('')
        self.features = features

    def __len__(self) -> int:
        logger.debug('')
        return len(self.features)

    def __getitem__(self, idx: int) -> List[int]:
        logger.debug('')
        return self.features[idx]
