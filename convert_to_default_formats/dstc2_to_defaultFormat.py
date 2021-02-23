'''
Vineet Kumar, sioom.ai
Convert dstc2 dataset files to a default format
'''

import sys
import pathlib
import logging
import pickle
from typing import List, Dict
import copy

logger = logging.getLogger(__name__)


def dstc2_to_defaultFormat() -> None:
    logger.debug('')
    fromDataP = pathlib.Path(__file__).parents[1].joinpath(
        'data', 'dialog-bAbI-tasks').resolve(strict=True)
    toDataP = fromDataP.joinpath('dstc2')
    toDataP.mkdir(exist_ok=True)

    toFiles = (toDataP.joinpath('defaultFormat.train'),
               toDataP.joinpath('defaultFormat.valid'),
               toDataP.joinpath('defaultFormat.test'))
    for file in toFiles:
        try:
            file.touch(exist_ok=False)
        except FileExistsError:
            logger.info(
                f'Conversion not needed. Following file already exists {file}')

    fromFiles = (fromDataP.joinpath('dialog-babi-task6-dstc2-trn.txt'),
                 fromDataP.joinpath('dialog-babi-task6-dstc2-dev.txt'),
                 fromDataP.joinpath('dialog-babi-task6-dstc2-tst.txt'))
    for file in fromFiles:
        if not file.exists():
            strng = (
                f'Program ended prematurely. Following file does not exist '
                f'{file}')
            logger.critical(strng)
            sys.exit()

    for fromFile, toFile in zip(fromFiles, toFiles):  # new file
        with fromFile.open('r') as fromF:
            dialogs_list = []  # list of dictionaries, where dialog is in dict
            user, bot, bot_idx, api_call_result = [], [], [], []
            start_of_file = True
            prev_line_apiCall = False
            dlg_lineno = 0
            for lineno, line in enumerate(fromF):  # new line
                if line == '\n':
                    continue
                user_utt = bot_utt = api_out = None
                num, _, line = line.rstrip().partition(' ')
                if not num.isdecimal():
                    logger.critical(
                        f"Missing decimal number at start of line {lineno}")
                    sys.exit()
                if num == '1':  # new dialog
                    if not start_of_file:
                        save_previous_dialog(dialogs_list, dlg_lineno, user,
                                             bot, bot_idx, api_call_result)
                    dlg_lineno = lineno
                    user.clear(), bot.clear(), bot_idx.clear()
                    api_call_result.clear()
                    start_of_file = False
                try:
                    user_utt, bot_utt = line.split('\t')
                    prev_line_apiCall = False
                except ValueError:
                    api_out = line

                # save line info
                if user_utt is not None:
                    assert (bot_utt is not None)
                    user.append(copy.deepcopy(user_utt))
                    bot.append(copy.deepcopy(bot_utt))
                if api_out is not None:
                    assert (user_utt is None and bot_utt is None)
                    if prev_line_apiCall is False:
                        bot_idx.append(len(bot) - 1)
                        api_call_result.append(copy.deepcopy([api_out]))
                        prev_line_apiCall = True
                    else:
                        api_call_result[len(bot_idx) - 1].append(
                            copy.deepcopy(api_out))
            else:  # for-else; EOF
                save_previous_dialog(dialogs_list, dlg_lineno, user, bot,
                                     bot_idx, api_call_result)
                with toFile.open('wb') as toF:
                    logger.info(f'Done writing to file {toFile}')
                    pickle.dump(dialogs_list,
                                toF,
                                protocol=pickle.HIGHEST_PROTOCOL)
                del dialogs_list


def save_previous_dialog(dialogs_list: List[Dict], dlg_lineno: int,
                         user: List[str], bot: List[str], bot_idx: List[int],
                         api_call_result: List[List[str]]) -> None:
    dialog = {
        'dlg_start_lineno': dlg_lineno,
        'persona': [],
        'user': user,
        'bot': bot,
        'bot_idx': bot_idx,
        'api_call_result': api_call_result
    }
    dialogs_list.append(copy.deepcopy(dialog))


if __name__ == '__main__':
    dstc2_to_defaultFormat()
