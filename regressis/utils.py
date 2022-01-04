#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
import logging

import numpy as np
import healpy as hp

logger = logging.getLogger("utils")

_logging_handler = None


def setup_logging(log_level="info", stream=sys.stdout, log_file=None):
    """
    Turn on logging with specific configuration.

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
def build_healpix_map(nside, ra, dec, in_deg2=False):
    """
    Build healpix map from ra, dec input.

    Parameters
    ----------
    nside: int
        Healpix resolution of the output.
    ra: array like
        Array containg Right Ascension in degree.
    dec: array like
        Array containg Declination in degree. Same size than ra.
    in_deg2: bool, default=False
        If true, divide the output by the pixel areal.

    Returns
    -------
    map: array
        Density map of objetcs from (ra, dec) in a healpix map at nside in nested order.
    """
    map = np.zeros(hp.nside2npix(nside))
    pixels = hp.ang2pix(nside, ra, dec, nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    map[pix] = counts
    if in_deg2:
        map /= hp.nside2pixarea(nside, degrees=True)
    return map


def mean_on_healpix_map(map, depth_neighbours=1):
    """
    Build the average of a healpix map with a specific width.
    It is similar than a convolution in a 2d matrix with a gaussian like kernel of size depth_neighbours.

    Parameters
    ----------
    map: array
        Full healpix map supposed nested.
    depth_neightbours: int
        Width of the average.
    Returns
    -------
    mean_map: array
        Full healpix map convolved with a gaussian like kernel.
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
