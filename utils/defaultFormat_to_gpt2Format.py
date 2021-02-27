'''
Vineet Kumar, sioom.ai
Change from default format to a format that gpt2 model understands
'''

from sys import exit
import pathlib
from logging import getLogger
import pickle
from typing import Dict
import copy

logg = getLogger(__name__)


def defaultFormat_to_gpt2Format(tokenizer, tokenizer_type,
                                default_format_path: str) -> Dict:
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

    statistics_dir = pathlib.Path.cwd().joinpath('statistics')
    statistics_dir.mkdir(exist_ok=True)
    dlgs_meta_file_test = statistics_dir.joinpath('dlgs_metadata.test')
    dlgs_meta_file_test.touch()
    turns_meta_file_test = statistics_dir.joinpath('turns_metadata.test')
    turns_meta_file_test.touch()

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
        default_to_gpt2_format(tokenizer, fromFile, toFile, labels_max_len,
                               turns_meta_file_test, dlgs_meta_file_test)
    return {
        "f_paths": {
            "train": toTrainF,
            "valid": toValidF,
            "test": toTestF
        },
    }


def default_to_gpt2_format(tokenizer, fromFile: pathlib.PosixPath,
                           toFile: pathlib.PosixPath, labels_max_len: int,
                           turns_meta_file_test: pathlib.PosixPath,
                           dlgs_meta_file_test: pathlib.PosixPath) -> None:
    lst_input_ids = []
    dlgs_meta = []
    turns_meta = []
    test_file = fromFile.suffix == '.test'
    with fromFile.open('rb') as fromF:
        lst_dlgs = pickle.load(fromF)
        for dlg in lst_dlgs:
            assert len(dlg['user']) == len(dlg['bot'])
            if test_file:
                # metadata: (1) line # from the original dataset file where
                # this dlg started, (2) index of first turn in dlg
                dlgs_meta.append({
                    'lineno':
                    copy.deepcopy(dlg['dlg_start_lineno']),
                    'idx_first_trn':
                    len(lst_input_ids)
                })
            # persona is not used, so it is ignored
            history = ''
            max_gpt2_len = tokenizer.max_model_input_sizes[
                'distilgpt2'] - labels_max_len - 3
            for i, (u_str, b_str) in enumerate(zip(dlg['user'], dlg['bot'])):
                feature_ids = tokenizer(" ".join([history, u_str]),
                                        return_length=False,
                                        return_token_type_ids=False,
                                        return_attention_mask=False)

                if test_file:
                    # record info about this turn
                    trunc_u_str = trunc_i_str = None
                    u_ids = tokenizer(u_str,
                                      return_length=False,
                                      return_token_type_ids=False,
                                      return_attention_mask=False)
                    if (num_trunc :=
                            len(u_ids['input_ids']) - max_gpt2_len) > 0:
                        trunc_u_str = tokenizer.decode(
                            u_ids['input_ids'][:num_trunc])
                    elif (num_trunc :=
                          len(feature_ids['input_ids']) - max_gpt2_len) > 0:
                        trunc_i_str = tokenizer.decode(
                            feature_ids['input_ids'][:num_trunc])
                    turns_meta.append(
                        copy.deepcopy({
                            'u_str': u_str,
                            'trunc_u_str': trunc_u_str,
                            'trunc_i_str': trunc_i_str
                        }))

                label_ids = tokenizer(b_str,
                                      return_length=False,
                                      return_token_type_ids=False,
                                      return_attention_mask=False)
                # Policy: If feature_ids is larger than max allowed by GPT2,
                # then truncate by dropping the tokens in the beginning
                feature_ids_trunc = feature_ids['input_ids'][-max_gpt2_len:]
                lst_input_ids.append(
                    copy.deepcopy([tokenizer.bos_token_id] +
                                  feature_ids_trunc +
                                  [tokenizer.sep_token_id] +
                                  label_ids['input_ids'] +
                                  [tokenizer.eos_token_id]))
                try:
                    idx = dlg['bot_idx'].index(i)
                    history = " ".join([
                        history, u_str, b_str,
                        " ".join(dlg['api_call_result'][idx])
                    ])
                except ValueError:
                    history = " ".join([history, u_str, b_str])
        # last entry is not a new dlg but the previous dlg with the same
        # line # but index of last turn plus 1
        if test_file:
            dlgs_meta.append({
                'lineno': copy.deepcopy(dlg['dlg_start_lineno']),
                'idx_first_trn': len(lst_input_ids)
            })

    with toFile.open('wb') as toF:
        pickle.dump(lst_input_ids, toF, protocol=pickle.HIGHEST_PROTOCOL)
        logg.info(f'Done writing to file {toFile}')
    if test_file:
        with dlgs_meta_file_test.open('wb') as dF, turns_meta_file_test.open(
                'wb') as tF:
            pickle.dump(dlgs_meta, dF, protocol=pickle.HIGHEST_PROTOCOL)
            logg.info(f'Done writing to file {dlgs_meta_file_test}')
            pickle.dump(turns_meta, tF, protocol=pickle.HIGHEST_PROTOCOL)
            logg.info(f'Done writing to file {turns_meta_file_test}')
