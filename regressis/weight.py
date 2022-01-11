#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np
import healpy as hp

from . import utils


logger_photo = logging.getLogger('PhotoWeight')


class PhotoWeight(object):
    """Container of photometric weight with callable function to apply it to a (R.A., Dec.) catalog"""

    def __init__(self, sys_weight_map="/global/cfs/cdirs/desi/users/edmondc/Imaging_weight/MAIN/MAIN_LRG_imaging_weight_256.npy",
                 regions=None, mean_density_region=None):
        """
        Initialize :class:`PhotoWeight`.

        Parameters
        ----------
        sys_weight_map : float array or str
            Photometric weight in a healpix map format with nested scheme or path to .npy file containing the healpix map
        regions : list of str
            List of regions in which the systematic mitigation procedure was applied. The normalized target density
            is computed and the regression is applied independantly in each regions.
        mean_density_region : dict
            Dictionary containing the mean density over region of the considered data for each region in regions
        """

        if isinstance(sys_weight_map, str):
            sys_weight_map = np.load(sys_weight_map)

        self.map = sys_weight_map
        self.nside = hp.npix2nside(sys_weight_map.size)

        self.regions = regions
        self.mean_density_region = mean_density_region

    def __call__(self, ra, dec):
        """
        Build the photometric weight from a healpix map to a (R.A., Dec.) catalog.

        Parameters
        ----------
        ra : float array
            Array containing the R.A. values
        dec : float array
            Array containing the Dec. values. Same size than ra.

        Returns
        -------
        w : float array
            Photometric weight for each (Ra, Dec) values.
        """
        pix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
        return self.map[pix]

    def __str__(self):
        """Return comprehensible print for PhotoWeight class"""
        return f"PhotoWeight at nside:{self.nside} generated on {self.regions}"

    def __setstate__(self, state):
        """Set the class state dictionary. Here all the attributs are picklable, set self.__dict__ is enough."""
        self.__dict__ = state

    def __getstate__(self):
        """Return this class state dictionary. Here all the attributs are picklable, return self.__dict__ is enough."""
        return self.__dict__

    @classmethod
    def load(cls, filename):
        """Load class from disk. Instantiate and initalise class with state dictionary. """
        state = np.load(filename, allow_pickle=True).item()
        logger_photo.info(f"Load PhotoWeight class from {filename}")
        new = cls.__new__(cls)
        new.__setstate__(state)
        return new

    def save(self, filename):
        """Save class to disk. Remark: np.load(dictionary) will call pickle.dumps(). Here we just avoid to load this module."""
        utils.mkdir(os.path.dirname(filename))
        np.save(filename, self.__getstate__())
        logger_photo.info(f"Save PhotoWeight class in {filename}")


# class SpectroPhotoWeight(Base):
    # """Container of photometric weight with callable function to apply it to a (R.A., Dec.) catalog"""
    #
    # logger = logging.getLogger('SpectroPhotoWeight')
    #
    # def __init__(self, sys_weight_map=None):
    #     """
    #     Initialize :class:`SpectroPhotoWeight`.
    #
    #     Parameters
    #     ----------
    #     sys_weight_map: float array or str
    #         Photometric weight in a healpix map format with nested scheme or path to .npy file containing the healpix map
    #     """
    #
    #     if isinstance(sys_weight_map, str):
    #         sys_weight_map = np.load(sys_weight_map)
    #
    #     self.map = sys_weight_map
    #     self.nside = hp.npix2nside(sys_weight_map.size)
