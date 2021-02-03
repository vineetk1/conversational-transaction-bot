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
import copy

logg = getLogger(__name__)


def main():
    # last file name in command-line has dictionaries of parameters
    params_file_path = argv[len(argv) - 1]
    with open(params_file_path, 'r') as paramF:
        param_dicts = [
            dictionary for line in paramF if line[0] == '{'
            and isinstance(dictionary := literal_eval(line), dict)
        ]

    seed_everything(63)
    if 'chkpt' in param_dicts[0] and param_dicts[0]['chkpt'] is not None:
        model = ctbModel.load_from_checkpoint(
            checkpoint_path=param_dicts[0]['chkpt'])
        if not ('no_training' in param_dicts[0]
                and param_dicts[0]['no_training']):
            # Train checkpointed model with new hyperparameters
            param_dicts[1]['model_type'] = model.get_model_id()['model_type'],
            param_dicts[1]['tokenizer_type'] = model.get_model_id(
            )['tokenizer_type'],
            # param_dicts[1] needed by tb_subDir and ckpt_filename, so it
            # must not change
            param_dicts1 = copy.deepcopy(param_dicts[1])
            model.change_hperparams(param_dicts1)
    else:
        model = ctbModel(param_dicts[1], utils.NEW_TOKENS.SPECIAL_TOKENS,
                         utils.NEW_TOKENS.DSTC2_TOKENS)
    strng = (f"model_type={param_dicts[1]['model_type']}, "
             f"tokenizer_type={param_dicts[1]['tokenizer_type']}")
    logg.info(strng)

    if not ('no_training' in param_dicts[0] and param_dicts[0]['no_training']):
        # Training: True, Testing: Don't care
        tb_subDir = ",".join([
            f'{item}={param_dicts[1][item]}'
            for item in ['model_type', 'tokenizer_type']
            if item in param_dicts[1]
        ])
        tb_logger = TensorBoardLogger('tensorboard_logs', name=tb_subDir)

        ckpt_filename = ""
        for item in ['optz', 'optz_params', 'lr_sched', 'lr_sched_params']:
            if item in param_dicts[1]:
                if isinstance(param_dicts[1][item], str):
                    ckpt_filename += f'{item}={param_dicts[1][item]},'
                elif isinstance(param_dicts[1][item],
                                collections.abc.Iterable):
                    for k, v in param_dicts[1][item].items():
                        ckpt_filename += f'{k}={v},'
        ckpt_filename += '{epoch:02d}-{val_loss:.5f}'

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
    elif not ('no_testing' in param_dicts[0] and param_dicts[0]['no_testing']):
        # Training: False, Testing: True
        trainer = Trainer(logger=True,
                          checkpoint_callback=False,
                          **param_dicts[3])
    else:
        # Training: False, Testing: False
        strng = ('User specified no-training and no-testing. Must do either'
                 'training or testing or both.')
        logg.critical(strng)
        exit()

    data = ctbData(param_dicts[2])
    data.prepare_data(tokenizer=model.get_tokenizer(),
                      tokenizer_type=model.get_model_id()['tokenizer_type'],
                      no_training=True if 'no_training' in param_dicts[0]
                      and param_dicts[0]['no_training'] else False,
                      no_testing=True if 'no_testing' in param_dicts[0]
                      and param_dicts[0]['no_testing'] else False)
    data.setup()

    trainer.tune(model, datamodule=data)
    if not ('no_training' in param_dicts[0] and param_dicts[0]['no_training']):
        # Training: True, Testing: Don't care
        trainer.fit(model, datamodule=data)
        if not ('no_testing' in param_dicts[0]
                and param_dicts[0]['no_testing']):
            if 'test_pass_fail_stat' in param_dicts[0] and param_dicts[0][
                    'test_pass_fail_stat']:
                model.set_pass_fail_stat()
            trainer.test()  # auto loads checkpoint file with lowest val loss
            model.clear_pass_fail_stat()
    elif not ('no_testing' in param_dicts[0] and param_dicts[0]['no_testing']):
        # Training: False, Testing: True
        if 'test_pass_fail_stat' in param_dicts[0] and param_dicts[0][
                'test_pass_fail_stat']:
            model.set_pass_fail_stat()
        trainer.test(model, test_dataloaders=data.test_dataloader())
        model.clear_pass_fail_stat()
    else:
        # Training: False, Testing: False
        logg.critical('Bug in the Logic')
        exit()


if __name__ == '__main__':
    main()
