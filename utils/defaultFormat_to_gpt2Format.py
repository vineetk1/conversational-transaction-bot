'''
Vineet Kumar, sioom.ai
Change from default format to a format that gpt2 model understands
'''

import sys
import pathlib
import logging
import pickle
from typing import List, Dict
import copy
from transformers import GPT2Tokenizer

logger = logging.getLogger(__name__)


def defaultFormat_to_gpt2Format(args):
    logger.debug('')
    dirP = \
        pathlib.Path(args.default_format_path).parents[0].resolve(strict=True)
    stem = pathlib.Path(args.default_format_path).stem

    toFiles = (dirP.joinpath(f'{args.tokenizer}.train'),
               dirP.joinpath(f'{args.tokenizer}.valid'),
               dirP.joinpath(f'{args.tokenizer}.test'))
    for file in toFiles:
        try:
            file.touch(exist_ok=False)
        except FileExistsError:
            logger.debug(
                f'Conversion not needed. Following file already exists {file}')
            #return {"train": toTrainF, "valid": toValidF, "test": toTestF}

    fromFiles = (dirP.joinpath(f'{stem}.train'),
                 dirP.joinpath(f'{stem}.valid'), dirP.joinpath(f'{stem}.test'))
    for file in fromFiles:
        if not file.exists():
            strng = (
                f'Program ended prematurely. Following file does not exist '
                f'{file}')
            logger.critical(strng)
            sys.exit()

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
