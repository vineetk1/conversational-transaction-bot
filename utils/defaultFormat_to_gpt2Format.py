'''
Vineet Kumar, sioom.ai
Change from default format to a format that gpt2 model understands
'''

from sys import exit
import pathlib
from logging import getLogger
import pickle
from typing import Dict
from transformers import GPT2Tokenizer
import utils.NEW_TOKENS

logg = getLogger(__name__)


def defaultFormat_to_gpt2Format(tokenizer_type: str,
                                default_format_path: str) -> Dict:
    logg.debug('')
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    _ = tokenizer.add_special_tokens(utils.NEW_TOKENS.SPECIAL_TOKENS)
    _ = tokenizer.add_tokens(utils.NEW_TOKENS.TOKENS)

    dirP = pathlib.Path(default_format_path).parents[0].resolve(strict=True)
    stem = pathlib.Path(default_format_path).stem

    toFiles = (toTrainF := dirP.joinpath(f'{tokenizer_type}.train'), toValidF
               := dirP.joinpath(f'{tokenizer_type}.valid'), toTestF :=
               dirP.joinpath(f'{tokenizer_type}.test'))
    for file in toFiles:
        try:
            file.touch(exist_ok=False)
        except FileExistsError:
            logg.info(
                f'Conversion not needed. Following file already exists {file}')
            return {
                "f_paths": {
                    "train": toTrainF,
                    "valid": toValidF,
                    "test": toTestF
                },
                "tokenizer": tokenizer
            }

    fromFiles = (dirP.joinpath(f'{stem}.train'),
                 dirP.joinpath(f'{stem}.valid'), dirP.joinpath(f'{stem}.test'))
    for file in fromFiles:
        if not file.exists():
            strng = (
                f'Program ended prematurely. Following file does not exist '
                f'{file}')
            logg.critical(strng)
            exit()

    for fromFile, toFile in zip(fromFiles, toFiles):
        default_to_gpt2_format(tokenizer, fromFile, toFile)
    return {
        "f_paths": {
            "train": toTrainF,
            "valid": toValidF,
            "test": toTestF
        },
        "tokenizer": tokenizer
    }


def default_to_gpt2_format(tokenizer, fromFile: pathlib.PosixPath,
                           toFile: pathlib.PosixPath) -> None:
    lst_input_ids = []
    with fromFile.open('rb') as fromF:
        lst_dlgs = pickle.load(fromF)
        for dlg in lst_dlgs:
            assert len(dlg['user']) == len(dlg['bot'])
            # persona is not used, so it is ignored
            history = '<BOS>'
            for i, (u_str, b_str) in enumerate(zip(dlg['user'], dlg['bot'])):
                seq = " ".join([history, u_str, '<SEP>', b_str, '<EOS>'])
                history = " ".join([history, u_str, b_str])
                try:
                    idx = dlg['bot_idx'].index(i)
                    history = " ".join(
                        [history, " ".join(dlg['api_call_result'][idx])])
                except ValueError:
                    pass
                seq_ids = tokenizer(seq,
                                    return_length=True,
                                    return_token_type_ids=False,
                                    return_attention_mask=False)
                if seq_ids.length <=\
                        tokenizer.max_model_input_sizes['distilgpt2']:
                    # NOTE "copy.deepcopy" is not needed below
                    lst_input_ids.append(seq_ids['input_ids'])
                else:
                    # (1) Look at future turns to find what sequences of
                    # "api_call_result" are relevant; Remove all except those
                    # relevant sequences and a few irrelevant ones; (2) If
                    # point 1 doesn't work then remove all sequences in
                    # "api_call_result"; (3) If pt 2 doesn't work then remove
                    # the early turns involving both the user and the bot;
                    # Nothing is implemented yet except the seq_ids is NOT
                    # added to lst_input_ids
                    pass
    with toFile.open('wb') as toF:
        logg.info(f'Done writing to file {toFile}')
        pickle.dump(lst_input_ids, toF, protocol=pickle.HIGHEST_PROTOCOL)
