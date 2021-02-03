'''
Vineet Kumar, sioom.ai
'''

# DEBUG INFO WARN ERROR/EXCEPTION CRITICAL

from logging.config import dictConfig

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'consoleFormatter': {
            'format':
            '%(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s',
        },
        'fileFormatter': {
            'format':
            '[%(asctime)s] %(levelname)-6s %(filename)s:%(lineno)s:%(funcName)s(): %(message)s',
        },
    },
    'handlers': {
        'file': {
            'filename': 'ctb_logs',
            'level': 'ERROR',
            'class': 'logging.FileHandler',
            'formatter': 'fileFormatter',
        },
        'console': {
            'level': 'INFO',
            'class': 'logging.StreamHandler',
            'formatter': 'consoleFormatter',
        },
    },
    'loggers': {
        '':
        {  # root logger
            'handlers': ['file', 'console'],
            'level': 'DEBUG',
        },
    },
}
dictConfig(LOG_CONFIG)
