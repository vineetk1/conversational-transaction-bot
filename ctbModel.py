'''
Vineet Kumar, sioom.ai
'''

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

logger = logging.getLogger(__name__)


class ctbModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        logger.debug('')

    def forward(self):
        logger.debug('')

    def training_step(self):
        logger.debug('')

    def training_step_end(self):
        logger.debug('')

    def training_epoch_end(self):
        logger.debug('')

    def validation_step(self):
        logger.debug('')

    def validation_step_end(self):
        logger.debug('')

    def validation_epoch_end(self):
        logger.debug('')

    def test_step(self):
        logger.debug('')

    def test_step_end(self):
        logger.debug('')

    def test_epoch_end(self):
        logger.debug('')

    def configure_optimizers(self):
        logger.debug('')
        return torch.optim.Adam(
            [p for p in self.parameters() if p.requires_grad],
            lr=2e-05,
            eps=1e-08)
