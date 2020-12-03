'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
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
    data = ctbData(param_dicts[1])
    len_tokenizer = data.prepare_data(testing_only=True)
    data.setup()
    if param_dicts[0]['chkpt'] is None:
        model = ctbModel(param_dicts[2]['model_type'], len_tokenizer)
    else:
        model = ctbModel.load_from_checkpoint(
            checkpoint_path=param_dicts[0]['chkpt'])
    tb_logger = TensorBoardLogger('ctb_lightning_logs',
                                  name=Path(params_file_path).name)
    trainer = Trainer(logger=tb_logger,
                      checkpoint_callback=False,
                      **param_dicts[3])
    trainer.test(model=model, test_dataloaders=data.test_dataloader())


if __name__ == '__main__':
    main()
