import logging


def setup(log_file: str):
    """Sets up the logging.

    Args:
        log_file: The log file, e.g., '/your/path/logging.log'.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(ch)

    # create log file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
