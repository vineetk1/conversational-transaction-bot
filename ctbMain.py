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
import utils.NEW_TOKENS

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
    tb_logger = TensorBoardLogger('ctb_lightning_logs',
                                  name=Path(params_file_path).name)
    if 'chkpt' in param_dicts[0] and param_dicts[0]['chkpt'] is not None:
        model = ctbModel.load_from_checkpoint(
            checkpoint_path=param_dicts[0]['chkpt'])
        strng = (f'Checkpointed file has model_type='
                 f'{model.get_model_id()["model_type"]} and tokenizer_type='
                 f'{model.get_model_id()["tokenizer_type"]}')
        logg.info(strng)
        trainer = Trainer(logger=tb_logger,
                          checkpoint_callback=False,
                          **param_dicts[3])
    else:
        seed_everything(63)
        model = ctbModel(param_dicts[1], utils.NEW_TOKENS.SPECIAL_TOKENS,
                         utils.NEW_TOKENS.DSTC2_TOKENS)
        checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=param_dicts[0]['save_top_k'],
            period=1,
            filename='{epoch:02d}-{val_loss:.4f}')
        trainer = Trainer(logger=tb_logger,
                          deterministic=True,
                          num_sanity_val_steps=0,
                          callbacks=[checkpoint_callback],
                          **param_dicts[3])

    data = ctbData(param_dicts[2])
    data.prepare_data(tokenizer=model.get_tokenizer(),
                      tokenizer_type=model.get_model_id()['tokenizer_type'])
    data.setup()

    if 'chkpt' in param_dicts[0] and param_dicts[0]['chkpt'] is not None:
        trainer.test(model, test_dataloaders=data.test_dataloader())
    else:
        trainer.fit(model, datamodule=data)
        trainer.test()  # auto loads checkpoint file with lowest val loss


if __name__ == '__main__':
    main()
