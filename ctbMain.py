'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from ctbData import ctbData
from ctbModel import ctbModel
from ast import literal_eval
from sys import argv
from logging import getLogger
import utils.logging_config
import utils.NEW_TOKENS
import collections.abc

logg = getLogger(__name__)


def main():
    # last file in command-line has dictionaries of parameters
    params_file_path = argv[len(argv) - 1]
    with open(params_file_path, 'r') as paramF:
        param_dicts = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]

    tb_subDir = ",".join([
        f'{item}={param_dicts[1][item]}'
        for item in ['model_type', 'tokenizer_type'] if item in param_dicts[1]
    ])

    ckpt_filename = ""
    for item in ['optz', 'optz_params', 'lr_sched', 'lr_sched_params']:
        if item in param_dicts[1]:
            if isinstance(param_dicts[1][item], str):
                ckpt_filename += f'{item}={param_dicts[1][item]},'
            elif isinstance(param_dicts[1][item], collections.abc.Iterable):
                for k, v in param_dicts[1][item].items():
                    ckpt_filename += f'{k}={v},'
    ckpt_filename += '{epoch:02d}-{val_loss:.5f}'

    tb_logger = TensorBoardLogger('tensorboard_logs', name=tb_subDir)

    seed_everything(63)
    model = ctbModel(param_dicts[1], utils.NEW_TOKENS.SPECIAL_TOKENS,
                     utils.NEW_TOKENS.DSTC2_TOKENS)
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=param_dicts[0]['save_top_k']
        if 'save_top_k' in param_dicts[0] else 1,
        save_last=True,
        period=1,
        filename=ckpt_filename)
    lr_monitor = LearningRateMonitor(logging_interval='step',
                                     log_momentum=True)
    trainer = Trainer(logger=tb_logger,
                      deterministic=True,
                      num_sanity_val_steps=0,
                      log_every_n_steps=100,
                      callbacks=[checkpoint_callback, lr_monitor],
                      **param_dicts[3])

    data = ctbData(param_dicts[2])
    data.prepare_data(tokenizer=model.get_tokenizer(),
                      tokenizer_type=model.get_model_id()['tokenizer_type'])
    data.setup()

    trainer.tune(model, datamodule=data)
    trainer.fit(model, datamodule=data)
    if 'no_testing' not in param_dicts[
            0] or param_dicts[0]['no_testing'] is False:
        trainer.test()  # auto loads checkpoint file with lowest val loss


if __name__ == '__main__':
    main()
