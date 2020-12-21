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
                    # add 1 to length for EOS token
                    labels_max_len = max(labels_max_len, label_ids.length + 1)

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
    stat_num_examples = 0
    with fromFile.open('rb') as fromF:
        lst_dlgs = pickle.load(fromF)
        for dlg in lst_dlgs:
            assert len(dlg['user']) == len(dlg['bot'])
            # persona is not used, so it is ignored
            history = '<BOS>'
            for i, (u_str, b_str) in enumerate(zip(dlg['user'], dlg['bot'])):
                stat_num_examples += 1
                if fromFile.suffix != '.test':
                    seq = " ".join([history, u_str, '<SEP>', b_str, '<EOS>'])
                else:
                    seq = " ".join([history, u_str, '<SEP>'])
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
                label_ids = tokenizer(" ".join([b_str, '<EOS>']),
                                      return_length=True,
                                      return_token_type_ids=False,
                                      return_attention_mask=False)
                if fromFile.suffix == '.test':
                    if (seq_ids.length + labels_max_len) <=\
                            tokenizer.max_model_input_sizes['distilgpt2']:
                        # NOTE "copy.deepcopy" is not needed below
                        lst_input_ids.append(
                            (seq_ids['input_ids'], label_ids['input_ids']))
                else:
                    if seq_ids.length - label_ids.length + labels_max_len <=\
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
        pickle.dump(lst_input_ids, toF, protocol=pickle.HIGHEST_PROTOCOL)
        strng = (
            f'Done writing to file {toFile}: '
            f'# of examples: Total {stat_num_examples}, '
            f'Selected {len(lst_input_ids)}, '
            f'Discarded {stat_num_examples - len(lst_input_ids)}')
        logg.info(strng)
