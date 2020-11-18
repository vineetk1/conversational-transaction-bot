'''
Vineet Kumar, sioom.ai
'''

import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import pytorch_lightning as pl
import logging
import sys
from typing import List, Dict
import pickle

logger = logging.getLogger(__name__)


class ctbData(pl.LightningDataModule):
    def __init__(self, args):
        logger.debug('')
        super().__init__()
        self.args = args

    def prepare_data(self) -> int:
        logger.debug('')
        if self.args.tokenization == "gpt2":
            from utils.defaultFormat_to_gpt2Format import \
                    defaultFormat_to_gpt2Format
            data_info = defaultFormat_to_gpt2Format(self.args)
            self.tokenizer = data_info['tokenizer']
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
            return len(self.tokenizer)
        else:
            logger.critical(f'unknown tokenization: {self.args.tokenization}')
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
                          num_workers=6,
                          collate_fn=self.collater,
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
                          num_workers=6,
                          collate_fn=self.collater,
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
                          num_workers=6,
                          collate_fn=self.collater,
                          pin_memory=True,
                          drop_last=False,
                          timeout=0)

    def collater(self, examples: List[List[int]]) -> Dict[str, torch.Tensor]:
        logger.debug('')
        try:
            sep_idxs = [
                example.index(self.tokenizer.sep_token_id)
                for example in examples
            ]
        except ValueError:
            logger.critical('No sep_token in example')
            sys.exit()
        example_lens = [len(example) for example in examples]
        max_example_len = max(example_lens)
        assert self.tokenizer.padding_side == 'right'

        input_ids = torch.LongTensor([
            (example + [self.tokenizer.pad_token_id] *
             (max_example_len - example_len))
            for example, example_len in zip(examples, example_lens)
        ])

        attention_mask = torch.FloatTensor([[1] * example_len + [0] *
                                            (max_example_len - example_len)
                                            for example_len in example_lens])

        token_type_ids = torch.LongTensor([[0] * sep_idx + [1] *
                                           (max_example_len - sep_idx)
                                           for sep_idx in sep_idxs])

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids,
            'labels': input_ids
        }


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
