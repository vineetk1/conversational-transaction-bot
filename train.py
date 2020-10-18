'''
Vineet Kumar, sioom.ai
'''

import pytorch_lightning as pl
from ctbData import ctbData
from ctbModel import ctbModel
import utils.logging_config


data = ctbData()
data.prepare_data()
data.setup()
model = ctbModel()
trainer = pl.Trainer(gpus=1)
trainer.fit(model, datamodule=data)
