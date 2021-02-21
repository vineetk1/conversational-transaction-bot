'''
Vineet Kumar, sioom.ai
'''

from collections import Counter
import sys
from contextlib import redirect_stdout
import textwrap
import os


class DialogsInfoOut(object):
    def __init__(self):
        self.max_num_turns_in_dlg = 0
        self.count = Counter()
        # clear files if they exists; create files if not already created
        with open('failed_dialogs_info.txt', 'w'):
            pass
        with open('passed_dialogs_info.txt', 'w'):
            pass
        self.write_out('Abbrevations\n------------',
                       write_to=["passF", "failF"])
        strng = ('Line # (L) of the line in hmn and bot test files;'
                 'Pass (P); Fail (F); Turn (Tr); Source Sequence (S);'
                 ' Target Sequence (T); Hypothesis Sequence: (H0) has '
                 'highest probability, (H1) has next hightest '
                 'probability, etc.')
        self.write_out(strng,
                       write_to=["passF", "failF"],
                       next_lines_manual_indent=False)

    def dlg_info(self, passed: bool, num_consec_turns_passed: int,
                 num_turns_in_dlg: int):
        self.dlg_passed = passed
        self.max_num_turns_in_dlg = max(num_turns_in_dlg,
                                        self.max_num_turns_in_dlg)
        self.count[f'num_dlgs {passed}'] += 1
        self.count[f'num_turns_in_dlg {passed} {num_turns_in_dlg}'] += 1
        self.count[f'num_consec_turns_passed {num_consec_turns_passed}'] += 1

    def turn_info(self, turn_num_in_dlgs: int, passed: bool, untrunc: bool,
                  input: str, actual_output: str, predicted_output: str):
        self.count[f'num_turns {passed} {untrunc}'] += 1
        self.count[f'turn_num_in_dlgs {passed} {turn_num_in_dlgs}'] += 1
        self.write_out(
            f"Tu{turn_num_in_dlgs}-Tc{'(F)' if untrunc else '(T)'}-I:    {input}",
            write_to=['passF' if self.dlg_passed else 'failF'],
            first_line_indent_lev=0,
            next_lines_manual_indent=True,
            next_lines_indent=16)
        self.write_out(
            f"Tu{turn_num_in_dlgs}-Tc{'(F)' if untrunc else '(T)'}-O:    {actual_output}",
            write_to=['passF' if self.dlg_passed else 'failF'],
            first_line_indent_lev=0,
            next_lines_manual_indent=True,
            next_lines_indent=16)
        self.write_out(
            f"Tu{turn_num_in_dlgs}-Tc{'(F)' if untrunc else '(T)'}-PO-{'P' if passed else 'F'}:    {predicted_output}",
            write_to=['passF' if self.dlg_passed else 'failF'],
            first_line_indent_lev=0,
            next_lines_manual_indent=True,
            next_lines_indent=16)

    def print_statistics(self):
        self.write_out('Abbrevations\n------------',
                       write_to=["passF", "failF"])
        strng = '\nStatistics on the test\n----------------------\n'
        self.write_out(strng=strng,
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=False,
                       next_lines_manual_indent=False)

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
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       next_lines_manual_indent=False)

        # Percent of turns with truncated inputs
        num_turns_trunc = (self.count['num_turns False False']
                           if 'num_turns False False' in self.count else
                           0) + (self.count['num_turns True False'] if
                                 'num_turns True False' in self.count else 0)
        strng = (
            f'Percent of turns with truncated inputs = ({num_turns_trunc}/'
            f'{num_turns} x 100) = {(num_turns_trunc/num_turns * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_manual_indent=False)

        # Percent of turns that passed
        num_turns_pass = (self.count['num_turns True True']
                          if 'num_turns True True' in self.count else
                          0) + (self.count['num_turns True False']
                                if 'num_turns True False' in self.count else 0)
        strng = (
            f'Percent of turns that passed = ({num_turns_pass}/{num_turns} x '
            f'100) = {(num_turns_pass/num_turns * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_manual_indent=False)

        # Percent of truncated turns that passed
        if num_turns_trunc:
            num_turns_pass_trunc = self.count['num_turns True False'] \
                if 'num_turns True False' in self.count else 0
            strng = (f'Percent of turns that passed with truncated inputs = '
                     f'({num_turns_pass_trunc}/{num_turns_trunc} x 100) = '
                     f'{(num_turns_pass_trunc / num_turns_trunc * 100):.2f}%')
            self.write_out(strng=strng,
                           write_to=['passF', 'failF', 'stdout'],
                           bullet=True,
                           first_line_indent_lev=2,
                           next_lines_manual_indent=False)

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
                           write_to=['passF', 'failF', 'stdout'],
                           bullet=True,
                           first_line_indent_lev=2,
                           next_lines_manual_indent=False)

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
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_manual_indent=False)

        # Number of dialogs
        num_dlgs = (self.count['num_dlgs True'] if 'num_dlgs True' in self.
                    count else 0) + (self.count['num_dlgs False']
                                     if 'num_dlgs False' in self.count else 0)
        self.write_out(strng=f'Number of dialogs = {num_dlgs}',
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       next_lines_manual_indent=False)

        # Percent of dialogs that passed
        num_dlgs_passed = self.count[
            'num_dlgs True'] if 'num_dlgs True' in self.count else 0
        strng = (f'Number of dialogs that passed= {num_dlgs_passed}/{num_dlgs}'
                 f' x 100 = {(num_dlgs_passed/num_dlgs * 100):.2f}%')
        self.write_out(strng=strng,
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=1,
                       next_lines_manual_indent=False)

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
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=2,
                       next_lines_manual_indent=False)

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
                       write_to=['passF', 'failF', 'stdout'],
                       bullet=True,
                       first_line_indent_lev=2,
                       next_lines_manual_indent=False)

    def write_out(self,
                  strng,
                  write_to=["stdout"],
                  bullet=False,
                  first_line_indent_lev=0,
                  next_lines_manual_indent=True,
                  next_lines_indent=0):
        init_space = 3 * first_line_indent_lev * " "
        init_space = init_space + "** " if bullet else init_space
        if next_lines_manual_indent:
            next_line_space = next_lines_indent * " "
        else:
            next_line_space = (len(init_space) + 1) * " "

        for out in write_to:
            if out == "passF":
                out = 'passed_dialogs_info.txt'
            elif out == "failF":
                out = 'failed_dialogs_info.txt'
            elif out == "stdout":
                pass
            else:
                assert False, f'Illegal argument \'{out}\' in method write_out'
            with open(os.devnull if out == 'stdout' else out, 'a') \
                    as dialogs_stat_file:
                with redirect_stdout(sys.stdout if out ==
                                     'stdout' else dialogs_stat_file):
                    if init_space or next_line_space:
                        print(
                            textwrap.fill(strng,
                                          width=80,
                                          initial_indent=init_space,
                                          subsequent_indent=next_line_space))
                    else:
                        print(strng)
