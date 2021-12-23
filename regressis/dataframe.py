#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging

import numpy as np
import healpy as hp
import fitsio
import pandas as pd

from matplotlib import pyplot as plt

from . import utils
from .utils import hp_in_box


logger = logging.getLogger('DataFrame')

# To avoid error from pandas method into the logger -> pandas use NUMEXPR Package
if 'OMP_NUM_THREADS' in os.environ.keys():
  os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ['OMP_NUM_THREADS'])
  os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ['OMP_NUM_THREADS'])
else:
  os.environ.setdefault('NUMEXPR_MAX_THREADS', '1')
  os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')


class PhotometricDataFrame(object):
    """
    Build the dataframe needed to compute the weights due to photometry systematic effects
    """
    def __init__(self, version, tracer, footprint, suffix_tracer='',
                 data_dir=None, output_dir=None,
                 use_median=False, use_new_norm=False, region=None):
        """
        Initialize :class:`DataFrame`

        Parameters
        ----------
        version: str
            Which version you want to use as SV3 or MAIN (for SV3 / MAIN targets) or DA02 / Y1 / ect ...
            Usefull only to load default map saved in data_dir and for the output name of the directory or filename.
        tracer: str
            Which tracer you want to use. Usefull only to load default map saved in data_dir and for
            the output name of the directory or filename.
        footprint: class:`Footprint`
            Contain all the footprint informations needed to extract the specific regions from an healpix map.
        suffix_tracer: str
            Additional suffix for tracer. Usefull only to load default map saved in data_dir and for
            the output name of the directory or filename.
        data_dir: str
            Path where the default map that we want to use are saved. Not needed if you pass as argument the path
            of pixmap / tarets density / fracarea ect... or directly the map as an array.
        output_dir: str
            Path where figures / all the outputs will be saved. If none, nothing is saved
        use_median: bool
            Use median instead of mean to compute the normalized target density.
        use_new_norm: bool
            Use specific area far of the galatic plane and Sgr. Stream (to avoid stellar contaminant) to compute
            the mean target density. Usefull only for :attr:`tracer`=='QSO'.
        region: list of str
            List of region in which we want to apply the systematic mitigation procedure. The normalized target density
            is computed and the regression is applied independantly in each region. If none use the default region given in footprint.
        """
        self.version = version
        self.tracer = tracer
        self.suffix_tracer = suffix_tracer

        self.footprint = footprint
        self.nside = footprint.nside
        self.pixels = np.arange(hp.nside2npix(self.nside))

        # info to normalize the target density
        self.use_median = use_median
        self.use_new_norm = use_new_norm

        # which region we want to use --> if None use default region defined in footprint
        self.region = region
        if self.region is None:
            self.region = self.footprint.default_region
            logger.info(f'Using default regions {self.region}')

        logger.info(f"version: {self.version} -- tracer: {self.tracer} -- region: {self.region}")

        self.data_dir = data_dir # where maps are saved -> usefull only if you do not specified the path of the files in set_features / set_targets ...

        if output_dir is not None:  # if None --> nothing is save and no directory is built
            self.output_dir = os.path.join(output_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}')
            self.output_dataframe_dir = os.path.join(self.output_dir, 'dataframe')
            utils.mkdir(self.output_dir)
            utils.mkdir(self.output_dataframe_dir)
            logger.info(f"Plots are saved in {self.output_dataframe_dir}")
        else:
            self.output_dir = self.output_dataframe_dir = None


    def set_features(self, pixmap=None, sgr_stream=None, sel_columns=None, use_sgr_stream=True):
        """
        Set photometric templates info either from a pixweight array (already loaded) or read it from .fits file
        All the maps should be Healpix maps with :attr:`nside` in nested order.

        Parameters
        ----------
        pixmap: float array or str
            Array containg the photometric template at :attr:`nside` or the path to load the photometric template

        sgr_stream: float array or str
            Array containing the Sgr. Stream feature at the compatible :attr:`nside` or the path to load the Sgr. Stream feature

        sel_columns: list of str
            List containing which photometric features need to be extracted from the pixmap

        use_sgr_stream: bool
            Include or not the Sgr. Stream map --> the feature is really relevant for the QSO TS.
        """
        path_pixweight, path_sgr_stream = None, None

        if sel_columns is None:
            sel_columns = ['STARDENS', 'EBV',
                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                           'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']

        if isinstance(pixmap, str):
            path_pixweight = pixmap
        elif pixmap is None:
            path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.nside}.fits')

        if path_pixweight is not None:
            logger.info(f"Read {path_pixweight}.")
            feature_pixmap = pd.DataFrame(fitsio.FITS(path_pixweight)[1][sel_columns].read().byteswap().newbyteorder())
        else:
            feature_pixmap = pixmap[sel_columns]

        if use_sgr_stream:
            if isinstance(sgr_stream, str):
                path_sgr_stream = sgr_stream
            elif sgr_stream is None:
                path_sgr_stream = os.path.join(self.data_dir, f'sagittarius_stream_{self.nside}.npy')

            if not path_sgr_stream is None:
                # Load Sgr. Stream map
                logger.info(f"Read {path_sgr_stream}")
                stream_map = pd.DataFrame(np.load(path_sgr_stream), columns=['STREAM'])
            else:
                stream_map = pd.DataFrame(sgr_stream, columns=['STREAM'])
            self.features = pd.concat([stream_map, feature_pixmap], axis=1)
        else:
            self.features = feature_pixmap
        logger.info(f"Sanity check: number of NaNs in features: {self.features.isnull().sum().sum()}")


    def set_targets(self, targets=None, fracarea=None):
        """
        Set targets and fracarea map at the correct :attr:`nside`.
        All the maps should be Healpix maps with :attr:`nside` in nested order.

        Parameters
        ----------
        targets: float array or str
            Array containing the healpix map of the considered object density
            or
            path containing the targets
        fracarea: float array or str
            Array containing the associated observed fraction area of a pixel of a healpix map
            or
            path containg the fracarea
        """
        path_targets, path_fracarea = None, None

        if isinstance(targets, str):
            path_targets = targets
        elif targets is None:
            path_targets = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.npy')

        if not path_targets is None:
            logger.info(f"Read {path_targets}")
            targets = np.load(path_targets)
        self.targets = targets

        if isinstance(fracarea, str):
            path_fracarea = fracarea
        elif fracarea is None:
            path_fracarea = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_fracarea_{self.nside}.npy')

        if not path_fracarea is None:
            if os.path.isfile(path_fracarea):
                logger.info(f"Read {path_fracarea}")
                fracarea = np.load(path_fracarea)
            else:
                # Read fracarea_12290 from pixweight file
                logger.info("Do not find corresponding fracarea map --> use FRACAREA_12290 as default fracarea")
                path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.nside}.fits')
                logger.info(f"Read {path_pixweight}")
                fracarea = fitsio.FITS(path_pixweight)[1]['FRACAREA_12290'].read()
        self.fracarea = fracarea


    def build(self, selection_on_fracarea=False):
        """
        Build the normalized target density in the considered zone and choose the pixel to use during the training (clean and remove 'bad' pixels)

        Parameters
        ----------
        selection_on_fracarea: bool
            if True remove queue distribution of the fracarea --> not mandatory since it can be already done when building the target density map (with more specificity) especially for DA02 ect..
        """
        # use only pixels which are observed for the training
        # self.footprint can be an approximation of the true area where observations were conducted
        # use always fracarea > 0 to use observed pixels
        considered_footprint = (self.fracarea > 0) & self.footprint('Footprint')
        keep_to_train = considered_footprint.copy()

        if selection_on_fracarea:
            if self.nside == 512:
                min_fracarea, max_fracarea = 0.5, 1.5
            else:
                min_fracarea, max_fracarea = 0.9, 1.1
            keep_to_train &= (self.fracarea > min_fracarea) & (self.fracarea < max_fracarea)

        logger.info(f"The considered footprint represents {(considered_footprint).sum() / self.footprint('Footprint').sum():2.2%} of the DR9 footprint")
        logger.info(f"They are {(~keep_to_train[considered_footprint]).sum()} pixels which will be not used for the training i.e. {(~keep_to_train[considered_footprint]).sum()/(considered_footprint).sum():2.2%} ot the considered footprint")

        # build normalized targets
        normalized_targets, mean_targets_density = np.zeros(self.targets.size) * np.nan, dict()
        for zone_name in self.region:
            pix_zone = self.footprint(zone_name)
            pix_to_use = pix_zone & keep_to_train

            if self.use_new_norm:
                #compute normalization on subpart of the footprint (for instance which is expected to be free from stellar contamination)
                pix_to_use_norm = pix_to_use & self.footprint.get_keep_to_norm(zone_name)
            else:
                pix_to_use_norm = pix_to_use

            # compute the mean only on pixel with "correct" behaviour
            if not self.use_median:
                mean_targets_density_estimators = np.mean(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])
            else:
                mean_targets_density_estimators = np.median(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])

            # compute normalized_targets every where but we don't care we only use keep_to_train == 1 during the training
            normalized_targets[pix_zone] = self.targets[pix_zone] / (self.fracarea[pix_zone]*mean_targets_density_estimators)
            mean_targets_density[zone_name] = mean_targets_density_estimators
            logger.info(f"  ** {zone_name}: {mean_targets_density_estimators:2.2f} -- {normalized_targets[pix_to_use_norm].mean():1.4f} -- {normalized_targets[pix_to_use].mean():1.4f}")

        # some plots for sanity check
        if self.output_dataframe_dir is not None:
            plt.figure(figsize=(8,6))
            plt.hist(self.targets[considered_footprint], range=(0.1,100), bins=100)
            plt.savefig(os.path.join(self.output_dataframe_dir, f"test_remove_targets_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

            plt.figure(figsize=(8,6))
            plt.hist(self.fracarea[considered_footprint], range=(0.5, 1.4), bins=100)
            plt.savefig(os.path.join(self.output_dataframe_dir, f"test_remove_fracarea_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

            tmp = np.zeros(hp.nside2npix(self.nside))
            tmp[self.pixels[keep_to_train == False]] = 1
            plt.figure(figsize=(8,6))
            hp.mollview(tmp, rot=120, nest=True, title='strange pixel', cmap='jet')
            plt.savefig(os.path.join(self.output_dataframe_dir, f"strange_pixel_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))

            plt.figure(figsize=(8,6))
            plt.hist(normalized_targets[keep_to_train], range=(0.1,5), bins=100)
            plt.savefig(os.path.join(self.output_dataframe_dir, f"normalized_targets_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))

        self.density = normalized_targets
        self.mean_density_region = mean_targets_density
        self.keep_to_train = keep_to_train


# class SpectroscopyDataFrame(object):
#     """
#     Build the dataframe needed to compute the combined weights for photometry and spectroscopy systematic effects
#     """
#     def __init__(self, version, tracer, footprint, suffix_tracer='', data_dir=None, output_dir=None,
#                  Nside=None, use_median=False, use_new_norm=False, mask_lmc=False,
#                  clear_south=True, cut_desi=False, region=None):
#         """
#         Initialize :class:`DataFrame`
#
#         Parameters
#         ----------
#
#         mettre les valeurs par default a la place de kwargs
#
#         """
#         print("aaaa")
