#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import logging
import traceback

import numpy as np
import pandas as pd
import healpy as hp
import fitsio


logger = logging.getLogger("Utils")

_logging_handler = None


def exception_handler(exc_type, exc_value, exc_traceback):
    """Print exception with a logger."""
    # Do not print traceback if the exception has been handled and logged
    _logger_name = 'Exception'
    log = logging.getLogger(_logger_name)
    line = '='*100
    #log.critical(line[len(_logger_name) + 5:] + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    log.critical('\n' + line + '\n' + ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback)) + line)
    if exc_type is KeyboardInterrupt:
        log.critical('Interrupted by the user.')
    else:
        log.critical('An error occured.')


def setup_logging(log_level="info", stream=sys.stdout, log_file=None):
    """
    Turn on logging with specific configuration.

    Parameters
    ----------
    log_level : 'info', 'debug', 'warning', 'error'
        Logging level, message below this level are not logged.
    stream : sys.stdout or sys.stderr
        Where to stream.
    log_file : str, default=None
        If not ``None`` stream to file name.
    """
    levels = {"info" : logging.INFO,
              "debug" : logging.DEBUG,
              "warning" : logging.WARNING,
              "error" : logging.ERROR}

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
    logger.setLevel(levels[log_level.lower()])

    # SAVE LOG INTO A LOG FILE
    if log_file is not None:
        fh = logging.FileHandler(log_file)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    sys.excepthook = exception_handler


def setup_mplstyle():
    """Load the default regressis style for matplotlib"""
    # On NERSC, you may need to load tex with `module load texlive`
    from matplotlib import pyplot as plt
    plt.style.use(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'regressis.mplstyle'))


def mkdir(dirname):
    """Try to create ``dirnm`` and catch :class:`OSError`."""
    try:
        os.makedirs(dirname) # MPI...
    except OSError:
        return


def unique_list(li):
    """Remove duplicates while preserving order."""
    toret = []
    for el in li:
        if el not in toret: toret.append(el)
    return toret


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


def to_tex(string):
    """Replace '_' by ' ' in a string to use latex format in matplotlib."""
    return string.replace('_', ' ')


def read_fits_to_pandas(filename, ext=1, columns=None):
    """
    Read a .fits file and convert it into a :class:`pandas.DataFrame`.
    Warning: it does not work if a column contains a list or an array.

    Parameters
    ----------
    filename : str
        Path where the .fits file is saved.
    ext : int or str
        Extension to read.
    columns : list of str
        List of columns to read. Useful to avoid to use too much memory.

    Returns :
    ---------
    data_frame : pandas.DataFrame
        Data frame containing data in the fits file.
    """
    logger.info(f'Read ext: {ext} from {filename}')
    file = fitsio.FITS(filename)[ext]
    if columns is None: file = file[columns]
    return pd.DataFrame(file.read().byteswap().newbyteorder())


def build_healpix_map(nside, ra, dec, precomputed_pix=None, sel=None, weights=None, in_deg2=False):
    """
    Build healpix map from input ra, dec.

    Parameters
    ----------
    nside : int
        Healpix resolution of the output.
    ra : array like
        Array containing Right Ascension in degree.
    dec : array like
        Array containing Declination in degree. Same size as ``ra``.
    precomputed_pix : array like, default=None
        Array containing precomputed healpix pixel values for each (ra, dec) given.
        Avoid the time consuming computation: hp.ang2pix. Same size as ``ra``.
        Note if precomputed_pix is passed, ra, dec info can be whatever they will be not used.
    sel : boolean array like, default=None
        Mask array to select only considered objects from ra, dec catalog. Same size as ``ra``.
    weights : array like, default=None
        Optional weights.
    in_deg2 : bool, default=False
        If ``True``, divide the output by the pixel area.

    Returns
    -------
    map: array
        Density map of objetcs from (ra, dec) in a healpix map at nside in nested order.
    """
    if sel is None:
        sel = np.ones(precomputed_pix.size if (precomputed_pix is not None) else ra.size, dtype='?')
    pix = precomputed_pix[sel] if (precomputed_pix is not None) else hp.ang2pix(nside, ra[sel], dec[sel], nest=True, lonlat=True)
    map = np.bincount(pix, weights=weights, minlength=hp.nside2npix(nside)) / 1.0
    if in_deg2:
        map = map / hp.nside2pixarea(nside, degrees=True)
    return map


def mean_on_healpix_map(map, depth_neighbours=1):
    """
    Build the average of a healpix map with a specific width.
    It is similar to a convolution with a gaussian like kernel of size depth_neighbours.

    Parameters
    ----------
    map : array
        Full healpix map assumed nested.
    depth_neighbours : int
        Width of the kernel.

    Returns
    -------
    mean_map : array
        Full averaged healpix map.
    """
    def get_all_neighbours(nside, i, depth_neighbours=1):
        # get the pixel number of the neighbours of i at required width given by depth_neighbours
        pixel_list = hp.get_all_neighbours(nside, i, nest=True)
        pixel_tmp = pixel_list
        depth_neighbours -= 1
        while depth_neighbours != 0 :
            pixel_tmp = hp.get_all_neighbours(nside, pixel_tmp, nest=True)
            pixel_tmp = np.reshape(pixel_tmp, pixel_tmp.size)
            pixel_list = np.append(pixel_list, pixel_tmp)
            depth_neighbours -= 1
        return pixel_list

    nside = hp.npix2nside(map.size)
    mean_map = np.zeros(map.size)
    for i in range(map.size):
        neighbour_pixels = get_all_neighbours(nside, i, depth_neighbours)
        mean_map[i] = np.nansum(map[neighbour_pixels], axis=0)/neighbour_pixels.size
    return mean_map


def hp_in_box(nside, radecbox, inclusive=True, fact=4):
    """
    Determine which HEALPixels touch an RA, Dec box.
    Taken from https://github.com/desihub/desitarget/blob/master/py/desitarget/geomask.py.

    Parameters
    ----------
    nside : int
        (NESTED) HEALPixel nside.
    radecbox : list
        4-entry list of coordinates [ramin, ramax, decmin, decmax]
        forming the edges of a box in RA/Dec (degrees).
    inclusive : bool, optional, default=True
        See documentation for :meth:`healpy.query_polygon`.
    fact : int, default=4
        See documentation for :meth:`healpy.query_polygon`.

    Returns
    -------
    pixels : list
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
