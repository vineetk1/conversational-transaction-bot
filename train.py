'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from ctbData import ctbData
from ctbModel import ctbModel
from ast import literal_eval
from sys import argv
from pathlib import Path
from logging import getLogger
import utils.logging_config

logg = getLogger(__name__)


def main():
    logg.debug('')
    # last file in command-line has dictionaries of parameters
    params_file_path = argv[len(argv) - 1]
    with open(params_file_path, 'r') as paramF:
        param_dicts = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]
    seed_everything(63)
    data = ctbData(param_dicts[1])
    len_tokenizer = data.prepare_data()
    data.setup()
    model = ctbModel(param_dicts[2]['model_type'], len_tokenizer)
    tb_logger = TensorBoardLogger('ctb_lightning_logs',
                                  name=Path(params_file_path).name)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=2,
        period=1,
        filename='{epoch:02d}-{val_loss:.4f}')
    trainer = Trainer(logger=tb_logger,
                      num_sanity_val_steps=0,
                      callbacks=[checkpoint_callback],
                      **param_dicts[3])
    trainer.fit(model, datamodule=data)
    trainer.test()   # auto loads checkpoint file with lowest val loss


if __name__ == '__main__':
    main()
