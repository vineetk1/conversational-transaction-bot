'''
Vineet Kumar, sioom.ai
'''

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
import pytorch_lightning as pl
import logging
import sys

logger = logging.getLogger(__name__)


class ctbData(pl.LightningDataModule):
    def __init__(self, args):
        logger.debug('')
        super().__init__()
        self.args = args

    def prepare_data(self):
        logger.debug('')
        if self.args.tokenizer == "gpt2":
            from utils.defaultFormat_to_gpt2Format import \
                    defaultFormat_to_gpt2Format
            defaultFormat_to_gpt2Format(self.args)
        else:
            logger.critical(f'unknown tokenizer: {self.args.tokenizer}')
            sys.exit()

    def setup(self):
        logger.debug('')

    def train_dataloader(self):
        logger.debug('')

    def val_dataloader(self):
        logger.debug('')

    def test_dataloader(self):
        logger.debug('')
