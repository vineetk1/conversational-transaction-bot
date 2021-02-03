'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import Trainer
from ctbData import ctbData
from ctbModel import ctbModel
from ast import literal_eval
from sys import argv
from logging import getLogger
import utils.logging_config
import utils.NEW_TOKENS

logg = getLogger(__name__)


def main():
    # last file in command-line has dictionaries of parameters   
    params_file_path = argv[len(argv) - 1]
    with open(params_file_path, 'r') as paramF:
        param_dicts = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]
    if 'chkpt' in param_dicts[0] and param_dicts[0]['chkpt'] is not None:
        model = ctbModel.load_from_checkpoint(
            checkpoint_path=param_dicts[0]['chkpt'])
    elif 'model_type' in param_dicts[1] and param_dicts[1][
            'model_type'] is not None and 'tokenizer_type' in param_dicts[
                1] and param_dicts[1]['tokenizer_type'] is not None:
        model = ctbModel(param_dicts[1], utils.NEW_TOKENS.SPECIAL_TOKENS,
                         utils.NEW_TOKENS.DSTC2_TOKENS)
    else:
        logg.critical('No checkpoint file or pre-trained model to load')
        exit()

    strng = (f'Checkpoint file has model_type='
             f'{model.get_model_id()["model_type"]} and tokenizer_type='
             f'{model.get_model_id()["tokenizer_type"]}')
    logg.info(strng)
    trainer = Trainer(logger=True, checkpoint_callback=False, **param_dicts[3])

    data = ctbData(param_dicts[2])
    data.prepare_data(tokenizer=model.get_tokenizer(),
                      tokenizer_type=model.get_model_id()['tokenizer_type'],
                      testing_only=True)
    data.setup()

    trainer.tune(model, datamodule=data)  # determine batch-size
    if 'pass_fail_stat' in param_dicts[0] and param_dicts[0]['pass_fail_stat']:
        model.set_pass_fail_stat()
    trainer.test(model, test_dataloaders=data.test_dataloader())
    model.clear_pass_fail_stat()


if __name__ == '__main__':
    main()
