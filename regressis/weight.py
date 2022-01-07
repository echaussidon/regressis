#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np
import healpy as hp


class PhotoWeight(object):

    """Container of photometric weight with callable function to apply it to a (R.A., Dec.) catalog"""

    logger = logging.getLogger('PhotoWeight')

    def __init__(self, sys_weight_map="/global/cfs/cdirs/desi/users/edmondc/Imaging_weight/MAIN/MAIN_LRG_imaging_weight_256.npy"):
        """
        Initialize :class:`PhotoWeight`.

        Parameters
        ----------
        sys_weight_map: float array or str
            Photometric weight in a healpix map format with nested scheme or path to .npy file containing the healpix map
        """

        if isinstance(sys_weight_map, str):
            sys_weight_map = np.load(sys_weight_map)

        self.map = sys_weight_map
        self.nside = hp.npix2nside(sys_weight_map.size)

    def __call__(self, ra, dec):
        pix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
        return self.map[pix]


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
    #
    # def __call__(self, ra, dec):
    #     pix = hp.ang2pix(self.nside, ra, dec, nest=True, lonlat=True)
    #     return self.map[pix]
