'''
Vineet Kumar, sioom.ai
'''

from pytorch_lightning import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from ctbData import ctbData
from ctbModel import ctbModel
import logging
import utils.logging_config
from argparse import ArgumentParser

logg = logging.getLogger(__name__)


def main(args):
    logg.debug('')
    seed_everything(63)
    data = ctbData(args)
    len_tokenizer = data.prepare_data()
    data.setup()
    model = ctbModel(args, len_tokenizer)
    tb_logger = TensorBoardLogger('ctb_lightning_logs', name='ctb_model')
    #trainer = Trainer.from_argparse_args(args)
    trainer = Trainer(logger=tb_logger, gpus=1, num_sanity_val_steps=0)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--tokenization', type=str, default='gpt2')
    parser.add_argument(
        '--default_format_path',
        type=str,
        default='data/dialog-bAbI-tasks/dstc2/defaultFormat.train')
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
