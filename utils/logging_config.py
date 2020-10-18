'''
Vineet Kumar, sioom.ai
'''

import logging

logger = logging.getLogger()  # root logger
logger.setLevel(logging.DEBUG)  # DEBUG INFO WARN ERROR/EXCEPTION CRITICAL
formatter = logging.Formatter(
    '%(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s')
# '%(levelname)-6s %(filename)s:%(lineno)s: %(message)s')
# '%(levelname)s:%(lineno)s: %(message)s')
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
console.setFormatter(formatter)
logger.addHandler(console)
file = logging.FileHandler('ctb_bot.log')
file.setLevel(logging.DEBUG)
file.setFormatter(formatter)
logger.addHandler(file)
