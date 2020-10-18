'''
Vineet Kumar, sioom.ai
'''

import torch
from torch.utils.data import TensorDataset, RandomSampler, DataLoader, random_split
import pytorch_lightning as pl
from transformers.data.processors.glue import MnliProcessor
from utils.dstc2_to_defaultFormat import dstc2_to_defaultFormat
import os
import urllib.request
import zipfile
import logging

logger = logging.getLogger(__name__)


class ctbData(pl.LightningDataModule):
    def __init__(self):
        logger.debug('')
        super().__init__()

    def prepare_data(self):
        logger.debug('')
        dstc2_to_defaultFormat()

    def setup(self):
        logger.debug('')

    def train_dataloader(self):
        logger.debug('')

    def val_dataloader(self):
        logger.debug('')

    def test_dataloader(self):
        logger.debug('')
