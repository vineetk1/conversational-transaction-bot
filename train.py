'''
Vineet Kumar, sioom.ai
'''

import pytorch_lightning as pl
from ctbData import ctbData
from ctbModel import ctbModel
import logging
import utils.logging_config

logger = logging.getLogger(__name__)


def main():
    logger.debug('')
    data = ctbData()
    data.prepare_data()
    data.setup()
    model = ctbModel()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    main()
