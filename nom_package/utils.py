import logging
import time
import sys

_logging_handler = None
def setup_logging(log_level="info", stream=sys.stdout, log_file=None):
    """
    Turn on logging with specific configuration
    Parameters
    ----------
    log_level : 'info', 'debug', 'warning', 'error'
        the logging level to set; logging below this level is ignored.
    stream : sys.stdout or sys.stderr
    log_file : filename path where the logger has to be written
    """

    levels = {
            "info" : logging.INFO,
            "debug" : logging.DEBUG,
            "warning" : logging.WARNING,
            "error" : logging.ERROR
            }

    logger = logging.getLogger();
    t0 = time.time()

    class Formatter(logging.Formatter):
        def format(self, record):
            self._style._fmt = '[%09.2f]' % (time.time() - t0) + ' %(asctime)s %(name)-20s %(levelname)-8s %(message)s'
            return super(Formatter,self).format(record)
    fmt = Formatter(datefmt='%y-%m-%d %H:%M ')

    global _logging_handler
    if _logging_handler is None:
        _logging_handler = logging.StreamHandler(stream=stream)
        logger.addHandler(_logging_handler)

    _logging_handler.setFormatter(fmt)
    logger.setLevel(levels[log_level])

    # SAVE LOG INTO A LOG FILE
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)


import collections.abc

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        print(key, value)
        if isinstance(value, collections.abc.Mapping) and value:
            returned = deep_update(source.get(key, {}), value)
            source[key] = returned
        else:
            source[key] = overrides[key]
    return source
