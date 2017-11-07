import sys

# logging
LOGGING = {
    'version': 1,
    'disable_existing_loggers': True,
    'formatters': {
        'message_only': {
            'format': '%(asctime)s: %(message)s',
            'datefmt': '%d-%m-%Y %H:%M:%S',
        },
        'basic': {
            'format': '%(asctime)s:%(levelname)s: %(message)s',
        },
        'verbose': {
            'format': '%(asctime)s:%(levelname)s:%(name)s.%(funcName)s: %(message)s',
        },
        'verbose_with_pid': {
            'format': '%(asctime)s:%(levelname)s:%(name)s.%(funcName)s:%(process)s: %(message)s',
        },
    },
    'handlers': {
        'basic': {
            'class': 'logging.StreamHandler',
            'formatter': 'verbose',
            'stream': sys.stdout,
        },
        'debug_stdout': {
            'class': 'logging.StreamHandler',
            'level': 'DEBUG',
            'formatter': 'verbose',
            'stream': sys.stdout,
        }
    },
    'loggers': {
        'root': {
            'handlers': ['basic'],
        },
        'server': {
            'handlers': ['debug_stdout'],
            'level': 'INFO',
            'propagate': True,
        }
    }
}

UG_FILE_PATH = None
USER_FEATURE_FILE_PATH = None

BG_FILE_PATH = None
BOOKING_FEATURE_FILE_PATH = None

PROPERTY_FILE_PATH = None
PROPERTY_FEATURE_FILE_PATH = None

UG_BG_RECS_MATRIX_PATH = None
