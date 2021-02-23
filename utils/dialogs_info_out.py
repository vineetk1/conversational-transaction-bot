'''
Vineet Kumar, sioom.ai
'''

from collections import Counter
import sys
from contextlib import redirect_stdout
import textwrap
from pathlib import Path
from typing import List


class DialogsInfoOut(object):
    def __init__(self):
        self.max_num_turns_in_dlg = 0
        self.count = Counter()
        # clear files if they exists; create files if not already created
        stat_dir = Path.cwd().joinpath('statistics')
        self.passF = stat_dir.joinpath('passed_dialogs_info.txt')
        self.passF.touch()
        self.passF.write_text('')  # empty the file
        self.failF = stat_dir.joinpath('failed_dialogs_info.txt')
        self.failF.touch()
        self.failF.write_text('')  # empty the file
        self.stdout = Path('/dev/null')
        self.write_out(strng='Abbrevations\n------------',
                       write_to=[self.passF, self.failF])
        strng = ('Turn (Tu); Truncated Input (TI) or Untruncated Input (UI); '
                 'Actual Output (AO); Predicted Output (PO); '
                 'Turn Passed (P); Turn Failed (F);')
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF],
                       next_lines_indent=1)
        pass

    def dlg_info(self, passed: bool, num_consec_turns_passed: int,
                 num_turns_in_dlg: int):
        self.max_num_turns_in_dlg = max(num_turns_in_dlg,
                                        self.max_num_turns_in_dlg)
        self.count[f'num_dlgs {passed}'] += 1
        self.count[f'num_turns_in_dlg {passed} {num_turns_in_dlg}'] += 1
        self.count[f'num_consec_turns_passed {num_consec_turns_passed}'] += 1

    def turn_info(self, dlg_passed: bool, turn_num_in_dlgs: int, passed: bool,
                  untrunc: bool, input: str, actual_output: str,
                  predicted_output: str):
        self.count[f'num_turns {passed} {untrunc}'] += 1
        self.count[f'turn_num_in_dlgs {passed} {turn_num_in_dlgs}'] += 1

        if turn_num_in_dlgs == 1:
            self.write_out(
                strng='',  # newline
                write_to=[self.passF if dlg_passed else self.failF])
        self.write_out(strng=(f"Tu{turn_num_in_dlgs}-"
                              f"{'UI' if untrunc else 'TI'}:"),
                       strng1=f"{input}",
                       write_to=[self.passF if dlg_passed else self.failF])
        self.write_out(strng=f"Tu{turn_num_in_dlgs}-AO:",
                       strng1=f"{actual_output}",
                       write_to=[self.passF if dlg_passed else self.failF])
        self.write_out(
            strng=f"Tu{turn_num_in_dlgs}-PO-{'P' if passed else 'F'}:",
            strng1=f"{predicted_output}",
            write_to=[self.passF if dlg_passed else self.failF])

    def print_statistics(self):
        strng = '\nStatistics on the test set\n--------------------------'
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout])

        # Number of turns
        num_turns = (self.count['num_turns False False']
                     if 'num_turns False False' in self.count else 0) + (
                         self.count['num_turns False True']
                         if 'num_turns False True' in self.count else 0) + (
                             self.count['num_turns True False']
                             if 'num_turns True False' in self.count else
                             0) + (self.count['num_turns True True'] if
                                   'num_turns True True' in self.count else 0)
        self.write_out(strng=f'Number of turns = {num_turns}',
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       next_lines_indent=1)

        # Percent of turns with truncated inputs
        num_turns_trunc = (self.count['num_turns False False']
                           if 'num_turns False False' in self.count else
                           0) + (self.count['num_turns True False'] if
                                 'num_turns True False' in self.count else 0)
        strng = (
            f'Percent of turns with truncated inputs = ({num_turns_trunc}/'
            f'{num_turns} x 100) = {(num_turns_trunc/num_turns * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_indent=1)

        # Percent of turns that passed
        num_turns_pass = (self.count['num_turns True True']
                          if 'num_turns True True' in self.count else
                          0) + (self.count['num_turns True False']
                                if 'num_turns True False' in self.count else 0)
        strng = (
            f'Percent of turns that passed = ({num_turns_pass}/{num_turns} x '
            f'100) = {(num_turns_pass/num_turns * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_indent=1)

        # Percent of truncated turns that passed
        if num_turns_trunc:
            num_turns_pass_trunc = self.count['num_turns True False'] \
                if 'num_turns True False' in self.count else 0
            strng = (f'Percent of turns that passed with truncated inputs = '
                     f'({num_turns_pass_trunc}/{num_turns_trunc} x 100) = '
                     f'{(num_turns_pass_trunc / num_turns_trunc * 100):.2f}%')
            self.write_out(strng=strng,
                           write_to=[self.passF, self.failF, self.stdout],
                           bullet=True,
                           first_line_indent_lev=2,
                           next_lines_indent=1)

        # Percent of untruncated turns that passed
        num_turns_untrunc = (self.count['num_turns False True']
                             if 'num_turns False True' in self.count else
                             0) + (self.count['num_turns True True'] if
                                   'num_turns True True' in self.count else 0)
        if num_turns_untrunc:
            num_turns_pass_untrunc = self.count['num_turns True True'] \
                if 'num_turns True True' in self.count else 0
            strng = (
                f'Percent of turns that passed with untruncated inputs = '
                f'({num_turns_pass_untrunc}/{num_turns_untrunc} x 100) = '
                f'{(num_turns_pass_untrunc / num_turns_untrunc * 100):.2f}%')
            self.write_out(strng=strng,
                           write_to=[self.passF, self.failF, self.stdout],
                           bullet=True,
                           first_line_indent_lev=2,
                           next_lines_indent=1)

        # Just checking if the numbers add up
        assert (num_turns == num_turns_untrunc + num_turns_trunc)
        assert (num_turns_pass == num_turns_pass_untrunc +
                num_turns_pass_trunc)
        num_turns_fail = (self.count['num_turns False True']
                          if 'num_turns False True' in self.count else
                          0) + (self.count['num_turns False False'] if
                                'num_turns False False' in self.count else 0)
        assert (num_turns == num_turns_pass + num_turns_fail)

        # Percent of turns that passed at each turn-number in dialogs
        first_time = True
        strng = (
            'Percent of turns that passed at each turn-number in dialogs -- ('
            'Turn # in dialogs: # of such turns that passed/total number of '
            'such turns x 100 = result) -- ')
        for turn_num_in_dlgs in range(1, self.max_num_turns_in_dlg + 1):
            num_passed = (
                self.count[f'turn_num_in_dlgs True {turn_num_in_dlgs}']
                if f'turn_num_in_dlgs True {turn_num_in_dlgs}' in self.count
                else 0)
            num_failed = (
                self.count[f'turn_num_in_dlgs False {turn_num_in_dlgs}']
                if f'turn_num_in_dlgs False {turn_num_in_dlgs}' in self.count
                else 0)
            if first_time:
                first_time = False
                stng = (f'({turn_num_in_dlgs}: {num_passed}/'
                        f'{num_passed + num_failed} = '
                        f'{(num_passed/(num_passed + num_failed) * 100):.2f}'
                        '%)')
            else:
                stng = (f', ({turn_num_in_dlgs}: {num_passed}/'
                        f'{num_passed + num_failed} = '
                        f'{(num_passed/(num_passed + num_failed) * 100):.2f}'
                        '%)')
            strng += stng
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_indent=1)

        # Number of dialogs
        num_dlgs = (self.count['num_dlgs True'] if 'num_dlgs True' in self.
                    count else 0) + (self.count['num_dlgs False']
                                     if 'num_dlgs False' in self.count else 0)
        self.write_out(strng=f'Number of dialogs = {num_dlgs}',
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       next_lines_indent=1)

        # Percent of dialogs that passed
        num_dlgs_passed = self.count[
            'num_dlgs True'] if 'num_dlgs True' in self.count else 0
        strng = (f'Percent of dialogs that passed= {num_dlgs_passed}/'
                 f'{num_dlgs} x 100 = {(num_dlgs_passed/num_dlgs * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_indent=1)

        # Percent of dialogs with specified number of turns that passed
        first_time = True
        strng = (
            '(# of turns in dialog: # of such dialogs that passed/total number'
            ' of such dialogs x 100 = result) -- ')
        for num_turns_in_dlg in range(1, self.max_num_turns_in_dlg + 1):
            num_passed = (
                self.count[f'num_turns_in_dlg True {num_turns_in_dlg}']
                if f'num_turns_in_dlg True {num_turns_in_dlg}' in self.count
                else 0)
            num_failed = (
                self.count[f'num_turns_in_dlg False {num_turns_in_dlg}']
                if f'num_turns_in_dlg False {num_turns_in_dlg}' in self.count
                else 0)
            if not (num_passed or num_failed):
                continue
            if first_time:
                first_time = False
                stng = (f'({num_turns_in_dlg}: {num_passed}/'
                        f'{num_passed + num_failed} = '
                        f'{(num_passed/(num_passed + num_failed) * 100):.2f}'
                        '%)')
            else:
                stng = (f', ({num_turns_in_dlg}: {num_passed}/'
                        f'{num_passed + num_failed} = '
                        f'{(num_passed/(num_passed + num_failed) * 100):.2f}'
                        '%)')
            strng += stng
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=2,
                       next_lines_indent=1)

        # Number of consecutive turns that passed, counting from beginning of
        # dialog
        first_time = True
        strng = ('(# of consecutive turns that passed, counting from beginning'
                 ' of dialog: # of occurrences of such consecutive turns) -- ')
        for num_consec_turns_passed in range(1, self.max_num_turns_in_dlg + 1):
            if (f'num_consec_turns_passed {num_consec_turns_passed}'
                    in self.count):
                count = self.count[
                    f'num_consec_turns_passed {num_consec_turns_passed}']
                if first_time:
                    first_time = False
                    stng = (f'({num_consec_turns_passed}: {count})')
                else:
                    stng = (f', ({num_consec_turns_passed}: {count})')
                strng += stng
        self.write_out(strng=strng,
                       write_to=[self.passF, self.failF, self.stdout],
                       bullet=True,
                       first_line_indent_lev=2,
                       next_lines_indent=1)

    def write_out(self,
                  strng: str,
                  strng1: str = "",
                  write_to: List = [sys.stdout],
                  bullet: bool = False,
                  first_line_indent_lev: int = 0,
                  next_lines_indent: int = 0):
        # parameters that determine how the given string is printed: strng1,
        #   bullet, first_line_indent_lev, next_lines_indent
        #   (1) if default values are used, strng is printed as-is, otherwise
        #       text_wrap(...) is used that doesn't print newlines
        #   (2) if strng1 is used, then bullet, first_line_indent_lev,
        #        next_lines_indent are ignored

        strng_max_len_plus1 = 11
        if strng1:
            remaining_space = strng_max_len_plus1 - len(strng)
            strng = strng + (remaining_space * " ") + strng1
            init_space = ""
            next_line_space = strng_max_len_plus1 * " "
        else:
            init_space = 3 * first_line_indent_lev * " "
            init_space = init_space + "** " if bullet else init_space
            next_line_space = (len(init_space) + next_lines_indent) * " "

        for out in write_to:
            with out.open('a') as dialogs_stat_file:
                with redirect_stdout(sys.stdout if out ==
                                     self.stdout else dialogs_stat_file):
                    if init_space or next_line_space:
                        print(
                            textwrap.fill(strng,
                                          width=80,
                                          initial_indent=init_space,
                                          subsequent_indent=next_line_space))
                    else:
                        print(strng)
