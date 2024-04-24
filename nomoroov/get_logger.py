import os
import logging


LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=LOG_LEVEL)


def get_logger(filename):
    return logging.getLogger(os.path.basename(filename))
