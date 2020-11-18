'''
Vineet Kumar, sioom.ai
'''

import pytorch_lightning as pl
from ctbData import ctbData
from ctbModel import ctbModel
import logging
import utils.logging_config
from argparse import ArgumentParser

logger = logging.getLogger(__name__)


def main(args):
    logger.debug('')
    pl.seed_everything(63)
    data = ctbData(args)
    len_tokenizer = data.prepare_data()
    data.setup()
    model = ctbModel(args, len_tokenizer)
    trainer = pl.Trainer.from_argparse_args(args)
    trainer.fit(model, datamodule=data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--tokenization', type=str, default='gpt2')
    parser.add_argument(
        '--default_format_path',
        type=str,
        default='data/dialog-bAbI-tasks/dstc2/defaultFormat.train')
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    main(args)
