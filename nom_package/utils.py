# coding: utf-8
# Author : Edmond Chaussidon (CEA)

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

#------------------------------------------------------------------------------#
# dictionary upadte at different level

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

#------------------------------------------------------------------------------#
# Linear regression with iminuit
# Iminuit is only needed if you want to launch use_Kfold=False otherwise the linear regression is performed with sklearn

import numpy as np

class LeastSquares:
    def __init__(self, model, regulator, x, y, cov_inv):
        self.model = model
        self.regulator = regulator
        self.x = np.array(x)
        self.y = np.array(y)
        self.cov_inv = np.array(cov_inv)
        self.func_code = make_func_code(describe(self.model)[1:])

    def __call__(self, *par):
        ym = self.model(self.x, *par)
        chi2 = (self.y - ym).T.dot(self.cov_inv).dot(self.y - ym) + self.regulator*(np.nanmean(ym) - 1)**2
        return chi2


def regression_least_square(model, regulator, data_x, data_y, data_y_cov_inv, nbr_params, use_minos=False, print_covariance=False, print_param=True, return_errors=False, **dict_ini):
    from iminuit import Minuit, describe
    from iminuit.util import make_func_code

    chisq = LeastSquares(model, regulator, data_x, data_y, data_y_cov_inv)
    m = Minuit(chisq, forced_parameters=[f"a{i}" for i in range(0, nbr_params)], **dict_ini)
    # make the regression:
    m.migrad()
    if print_param:
        print(m.params)
    if use_minos:
        print(m.minos())
    if print_covariance:
        print(repr(m.covariance))
    if return_errors:
        return [m.values[f"a{i}"] for i in range(0, nbr_params)], [m.errors[f"a{i}"] for i in range(0, nbr_params)]
    else:
        return [m.values[f"a{i}"] for i in range(0, nbr_params)]

#------------------------------------------------------------------------------#
