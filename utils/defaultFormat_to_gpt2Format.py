'''
Vineet Kumar, sioom.ai
Change from default format to a format that gpt2 model understands
'''

from sys import exit
import pathlib
from logging import getLogger
import pickle
from typing import Dict

logg = getLogger(__name__)


def defaultFormat_to_gpt2Format(tokenizer, tokenizer_type,
                                default_format_path: str) -> Dict:
    logg.debug('')

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

    # find max length of labels in train/val/test files
    labels_max_len = 0
    for fromFile in fromFiles:
        with fromFile.open('rb') as fromF:
            lst_dlgs = pickle.load(fromF)
            for dlg in lst_dlgs:
                for b_str in dlg['bot']:
                    label_ids = tokenizer(b_str,
                                          return_length=True,
                                          return_token_type_ids=False,
                                          return_attention_mask=False)
                    labels_max_len = max(labels_max_len, label_ids['length'])

    for fromFile, toFile in zip(fromFiles, toFiles):
        default_to_gpt2_format(tokenizer, fromFile, toFile, labels_max_len)
    return {
        "f_paths": {
            "train": toTrainF,
            "valid": toValidF,
            "test": toTestF
        },
    }


def default_to_gpt2_format(tokenizer, fromFile: pathlib.PosixPath,
                           toFile: pathlib.PosixPath,
                           labels_max_len: int) -> None:
    lst_input_ids = []
    with fromFile.open('rb') as fromF:
        lst_dlgs = pickle.load(fromF)
        for dlg in lst_dlgs:
            assert len(dlg['user']) == len(dlg['bot'])
            # persona is not used, so it is ignored
            history = ''
            for i, (u_str, b_str) in enumerate(zip(dlg['user'], dlg['bot'])):
                feature_ids = tokenizer(" ".join([history, u_str]),
                                        return_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)
                label_ids = tokenizer(b_str,
                                      return_length=False,
                                      return_token_type_ids=False,
                                      return_attention_mask=False)
                # Policy: If feature_ids is larger than max allowed by GPT2,
                # then truncate by picking tokens from the end (i.e. drop the
                # tokens in the beginning)
                feature_ids_trunc = feature_ids['input_ids'][-(
                    tokenizer.max_model_input_sizes['distilgpt2'] -
                    labels_max_len - 3):]
                # NOTE "copy.deepcopy" is not needed below
                lst_input_ids.append([tokenizer.bos_token_id] +
                                     feature_ids_trunc +
                                     [tokenizer.sep_token_id] +
                                     label_ids['input_ids'] +
                                     [tokenizer.eos_token_id])
                try:
                    idx = dlg['bot_idx'].index(i)
                    history = " ".join([
                        history, u_str, b_str,
                        " ".join(dlg['api_call_result'][idx])
                    ])
                except ValueError:
                    history = " ".join([history, u_str, b_str])

    with toFile.open('wb') as toF:
        pickle.dump(lst_input_ids, toF, protocol=pickle.HIGHEST_PROTOCOL)
        logg.info(f'Done writing to file {toFile}')
