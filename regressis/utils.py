# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import sys
import time
import logging


logger = logging.getLogger("utils")

_logging_handler = None


def setup_logging(log_level="info", stream=sys.stdout, log_file=None):
    """
    Turn on logging with specific configuration?

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


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


#------------------------------------------------------------------------------#
def zone_name_to_column_name(zone_name):
    """
    Convert zone_name into corresponding name in footprint dataframe.
    """
    translator = {'North':'ISNORTH', 'South':'ISSOUTHWITHOUTDES', 'Des':'ISDES',
                  'South_mid':'ISSOUTHMID', 'South_pole':'ISSOUTHPOLE',
                  'Des_mid':'ISDESMID', 'South_all':'ISSOUTH'}
    if zone_name in translator.keys():
        return translator[zone_name]
    else:
        logger.error(f'{zone_name} is an UNEXPECTED REGION...')
        sys.exit()

#------------------------------------------------------------------------------#
# dictionary upadte at different level

def deep_update(source, overrides):
    """
    Update a nested dictionary or similar mapping.
    Modify ``source`` in place.
    """
    for key, value in overrides.items():
        if hasattr(value, 'items'):
            source.setdefault(key, {})
            deep_update(source[key], value)
        else:
            source[key] = value

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

import healpy as hp

def hp_in_box(nside, radecbox, inclusive=True, fact=4):
    """
    Determine which HEALPixels touch an RA, Dec box.
    Taken from https://github.com/desihub/desitarget/blob/master/py/desitarget/geomask.py.

    Parameters
    ----------
    nside : :class:`int`
        (NESTED) HEALPixel nside.
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax]
        forming the edges of a box in RA/Dec (degrees).
    inclusive : :class:`bool`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`.
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`.
    Returns
    -------
    :class:`list`
        HEALPixels at the passed `nside` that touch the RA/Dec box.

    Notes
    -----
        - Uses `healpy.query_polygon()` to retrieve the RA geodesics
          and then :func:`hp_in_dec_range()` to limit by Dec.
        - When the RA range exceeds 180o, `healpy.query_polygon()`
          defines the range as that with the smallest area (i.e the box
          can wrap-around in RA). To avoid any ambiguity, this function
          will only limit by the passed Decs in such cases.
        - Only strictly correct for Decs from -90+1e-3(o) to 90-1e3(o).
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM area enclosed isn't well-defined if RA covers more than 180o.
    if np.abs(ramax-ramin) <= 180.:
        # ADM retrieve RA range. The 1e-3 prevents edge effects near poles.
        npole, spole = 90-1e-3, -90+1e-3
        # ADM convert RA/Dec to co-latitude and longitude in radians.
        rapairs = np.array([ramin, ramin, ramax, ramax])
        decpairs = np.array([spole, npole, npole, spole])
        thetapairs, phipairs = np.radians(90.-decpairs), np.radians(rapairs)

        # ADM convert to Cartesian vectors remembering to transpose
        # ADM to pass the array to query_polygon in the correct order.
        vecs = hp.dir2vec(thetapairs, phipairs).T

        # ADM determine the pixels that touch the RA range.
        pixra = hp.query_polygon(nside, vecs,
                                 inclusive=inclusive, fact=fact, nest=True)
    else:
        logger.warning(f'Max RA ({ramax}) and Min RA ({ramin}) separated by > 180o...')
        logger.warning('...will only limit to passed Declinations')
        pixra = np.arange(hp.nside2npix(nside))

    # ADM convert Dec to co-latitude in radians.
    # ADM remember that, min/max swap because of the -ve sign.
    # ADM determine the pixels that touch the box.
    pixring = hp.query_strip(nside, np.radians(90.-decmax), np.radians(90.-decmin),
                             inclusive=inclusive, nest=False)
    pixdec = hp.ring2nest(nside, pixring) # not yet implemented

    # ADM return the pixels in the box.
    pixnum = list(set(pixra).intersection(set(pixdec)))

    return pixnum
