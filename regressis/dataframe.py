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
from .plot import plot_moll


logger = logging.getLogger('DataFrame')

# To avoid error from pandas method into the logger -> pandas use NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))
os.environ.setdefault('NUMEXPR_NUM_THREADS', os.environ.get('OMP_NUM_THREADS', '1'))


class PhotometricDataFrame(object):

    """Container gathering target density and photometric templates."""

    def __init__(self, version, tracer, footprint, suffix_tracer='',
                 data_dir=None, output_dir=None,
                 use_median=False, use_new_norm=False, regions=None):
        """
        Initialize :class:`PhotometricDataFrame`.

        Parameters
        ----------
        version : str
            Which version you want to use: SV3 or MAIN (for SV3 / MAIN targets) or DA02 / Y1 / etc.
            Useful only to load default map saved in ``data_dir`` and for the output name of the directory or file name.
        tracer : str
            Which tracer you want to use. Useful only to load default map saved in data_dir and for
            the output name of the directory or file name.
        footprint : class ``Footprint``
            The footprint information specifying regions in an Healpix format.
        suffix_tracer : str, default=''
            Additional suffix for tracer. Useful only to load default map saved in ``data_dir`` and for
            the output name of the directory or file name.
        data_dir : str, default=None
            Path where the default maps that we want to use are saved. Not needed if you pass as argument the path
            of pixmap / targets density / fracarea ect. or directly the map as an array.
        output_dir : str, default=None
            Path where figures / all the outputs will be saved. If none, nothing is saved on disk.
        use_median : bool, default=False
            Use median instead of mean to compute the normalized target density.
        use_new_norm : bool, default=False
            Use specific zone far of the galatic plane and Sgr. Stream (to avoid stellar contaminant) to compute
            the mean target density. Useful only when ``tracer`` is 'QSO'.
        regions : list of str, default=None
            List of regions in which we want to apply the systematic mitigation procedure. The normalized target density
            is computed and the regression is applied independantly in each regions. If none use the default regions(s) given in footprint.
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

        # which regions we want to use --> if None use default regions defined in footprint
        self.regions = regions
        if self.regions is None:
            self.regions = self.footprint.default_regions
            logger.info(f'Using default regions {self.regions}')

        logger.info(f"version: {self.version} -- tracer: {self.tracer} -- regions: {self.regions}")

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
        pixmap : float array or str, default=None
            Array containg the photometric templates or the path to .fits file containing the photometric templates.
        sgr_stream : float array or str, default=None
            Array containing the Sgr. Stream feature or the path to .npy file containing the Sgr. Stream feature.
        sel_columns : list of str, default=None
            List containing which photometric features must be extracted from the pixmap.
        use_sgr_stream : bool, default=True
            Include or not the Sgr. Stream map --> the feature is very relevant for the QSO TS.
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
            logger.info(f"Read {path_pixweight}")
            feature_pixmap = pd.DataFrame(fitsio.FITS(path_pixweight)[1][sel_columns].read().byteswap().newbyteorder())
        else:
            feature_pixmap = pixmap[sel_columns]

        if use_sgr_stream:
            if isinstance(sgr_stream, str):
                path_sgr_stream = sgr_stream
            elif sgr_stream is None:
                path_sgr_stream = os.path.join(self.data_dir, f'sagittarius_stream_{self.nside}.npy')

            if path_sgr_stream is not None:
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
        Set targets and fracarea maps.
        All the maps should be Healpix maps with :attr:`nside` in nested order.

        Parameters
        ----------
        targets : float array or str, default=None
            Array containing the healpix map of the considered object density or path to .npy file containing the targets.
        fracarea : float array or str, default=None
            Array containing the associated observed fraction area of a pixel of a healpix map
            or path to .npy file containg the fracarea.
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

    def build(self, cut_fracarea=False, fracarea_limits=None):
        """
        Build the normalized target density in the considered regions and choose the pixel to use during the training (clean and remove 'bad' pixels).

        Parameters
        ----------
        cut_fracarea : bool, default=False
            If ``True`` remove queue distribution of the fracarea. This is not mandatory since it can be already done
            when building the target density map with more specificity, e.g. for DA02.
            Max fracarea can be strictly > 1 due to Poisson noise.

        fracarea_limits : tuple, list, default=None
            If a tuple or list, min and max limits for fracarea.
        """
        # use only pixels which are observed for the training
        # self.footprint can be an approximation of the true area where observations were conducted
        # use always fracarea > 0 to use observed pixels
        # remove also pixel with 0 targets --> it should be already removed with fracarea > 0 in targets case
        # but not always with real desi data which have low fracarea...
        considered_footprint = (self.fracarea > 0) & (self.targets > 0) & self.footprint('footprint')
        keep_to_train = considered_footprint.copy()

        if cut_fracarea:
            if isinstance(fracarea_limits, (tuple, list)):
                min_fracarea, max_fracarea = fracarea_limits
            elif self.nside >= 512: # can be cirvumvent increasing the number of randoms...
                min_fracarea, max_fracarea = 0.85, 1.15
            else:
                min_fracarea, max_fracarea = 0.9, 1.1
            keep_to_train &= (self.fracarea > min_fracarea) & (self.fracarea < max_fracarea)

        # file to load DR9 footprint is roughly what we expect to be DR9. At the border, it is expected to have pixel with fracarea == 0 and which are in DR9 Footprint
        # {(considered_footprint).sum() / self.footprint('footprint').sum():2.2%} > 99.9 % is similar than 100 %.
        logger.info(f"The considered footprint represents {(considered_footprint).sum() / self.footprint('footprint').sum():2.2%} of the DR9 footprint")
        logger.info(f"They are {(~keep_to_train[considered_footprint]).sum()} pixels which will be not used for the training i.e. {(~keep_to_train[considered_footprint]).sum()/(considered_footprint).sum():2.2%} of the considered footprint")

        # build normalized targets
        normalized_targets, mean_targets_density = np.zeros(self.targets.size) * np.nan, dict()
        for region_name in self.regions:
            pix_region = self.footprint(region_name)
            pix_to_use = pix_region & keep_to_train

            if self.use_new_norm:
                #compute normalization on subpart of the footprint (for instance which is expected to be free from stellar contamination)
                pix_to_use_norm = pix_to_use & self.footprint.get_normalization_zone(region_name)
            else:
                pix_to_use_norm = pix_to_use

            # compute the mean only on pixel with "correct" behaviour
            if not self.use_median:
                mean_targets_density_estimators = np.mean(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])
            else:
                mean_targets_density_estimators = np.median(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])

            # compute normalized_targets every where
            # We will only use keep_to_train == 1 during the training (where fracarea > 0)
            # Can avoid the warning raised: RuntimeWarning: invalid value encountered in true_divide
            with np.errstate(divide='ignore',invalid='ignore'):
                normalized_targets[pix_region] = self.targets[pix_region] / (self.fracarea[pix_region]*mean_targets_density_estimators)
            mean_targets_density[region_name] = mean_targets_density_estimators
            logger.info(f"  ** {region_name}: {mean_targets_density_estimators:2.2f} -- {normalized_targets[pix_to_use_norm].mean():1.4f} -- {normalized_targets[pix_to_use].mean():1.4f}")

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
            plot_moll(tmp, show=False, label='strange pixel', filename=os.path.join(self.output_dataframe_dir, f"strange_pixel_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"), galactic_plane=True, ecliptic_plane=True)

            plt.figure(figsize=(8,6))
            plt.hist(normalized_targets[keep_to_train], range=(0.1,5), bins=100)
            plt.savefig(os.path.join(self.output_dataframe_dir, f"normalized_targets_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

        self.density = normalized_targets
        self.mean_density_region = mean_targets_density
        self.keep_to_train = keep_to_train


# class SpectroscopyDataFrame(object):
#     """
#     Build the dataframe needed to compute the combined weights for photometry and spectroscopy systematic effects
#     """
#     def __init__(self, version, tracer, footprint, suffix_tracer='', data_dir=None, output_dir=None,
#                  Nside=None, use_median=False, use_new_norm=False, mask_lmc=False,
#                  clear_south=True, cut_desi=False, regions=None):
#         """
#         Initialize :class:`DataFrame`
#
#         Parameters
#         ----------
#         pass
