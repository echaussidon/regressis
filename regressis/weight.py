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
                 regions=None, mask_region=None, mean_density_region=None):
        """
        Initialize :class:`PhotoWeight`.

        Parameters
        ----------
        sys_weight_map : float array or str
            Photometric weight in a healpix map format with nested scheme or path to .npy file containing the healpix map
        regions : list of str
            List of regions in which the systematic mitigation procedure was applied. The normalized target density
            is computed and the regression is applied independantly in each regions.
        mask_region : dict
            Dictionary containing the corresponding mask for each region in regions. Mask is a healpix map at self.nside in nested scheme.
            The mask can be collected with :class:`Footprint`. mask_region = {region:Footprint(region) for region in regions}
        mean_density_region : dict
            Dictionary containing the mean density over region of the considered data for each region in regions. NOT expected in deg2.
        """

        if isinstance(sys_weight_map, str):
            sys_weight_map = np.load(sys_weight_map)

        self.map = sys_weight_map
        self.nside = hp.npix2nside(sys_weight_map.size)

        self.regions = regions
        self.mask_region = mask_region
        self.mean_density_region = mean_density_region

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

    def __call__(self, ra, dec, normalize_map=False):
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
        if normalize_map:
            logger_photo.info(f'Map is normalized on {self.regions} before generating the weights')
            w_map = self.map.copy()
            for region in self.regions:
                w_map[self.mask_region[region]] /=  np.mean(self.map[self.mask_region[region]])
        else:
            w_map = self.map
        pix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
        return w_map[pix]

    def fraction_to_remove_per_pixel(self, ratio_mock_reality):
        """
        Build the fraction of objects in each pixel to remove to build the contamination with minimal value to zeros.

        Parameters
        ----------
        ratio_mock_reality: dict
            For each region in regions contain the ratio between the mock and the expected density.

        Returns
        -------
        frac_to_remove : array like
            Healpix map in nested scheme with self.nside. Contain the percentage of how many targets will be removed during mocks contamination.
        """
        frac_to_remove = np.zeros(self.map.size) * np.nan
        for region in self.regions:
            frac_to_remove[self.mask_region[region]] = 1 - 1 / (self.map[self.mask_region[region]] * ratio_mock_reality[region])
        frac_to_remove[frac_to_remove < 0] = 0.
        return frac_to_remove


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
