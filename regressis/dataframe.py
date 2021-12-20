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

from .desi_footprint import DR9_footprint
from .utils import hp_in_box, zone_name_to_column_name


logger = logging.getLogger('DataFrame')

# to avoid error from pandas method into the logger -> pandas use NUMEXPR Package
os.environ.setdefault('NUMEXPR_MAX_THREADS', '8')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '8')


class PhotometricDataFrame(object):
    """
    Build the dataframe needed to compute the weights due to photometry / spectroscopy systematic effects
    """
    def __init__(self, version, tracer, data_dir=None, output_dir=None, suffixe_tracer='',
                 Nside=None, use_median=False, use_new_norm=False, remove_LMC=False,
                 clear_south=True, mask_around_des=False, cut_DESI=False, region=None):
        """
        Initialize :class:`DataFrame`

        Parameters
        ----------

        mettre les valeurs par default a la place de kwargs

        """
        self.version = version
        self.tracer = tracer
        self.suffixe_tracer = suffixe_tracer

        self.data_dir = data_dir #where maps are saved -> usefull only if you do not specified the path of the files in set_features / set_targets ...

        self.output_dir = output_dir #if None --> nothing is save and no directory is built

        self.Nside = Nside
        self.pixels = np.arange(0, hp.nside2npix(self.Nside))

        # info to normalize the target density
        self.use_median = use_median
        self.use_new_norm = use_new_norm

        # footprint info
        self.DR9 = DR9_footprint(self.Nside, remove_LMC=remove_LMC, clear_south=clear_south, mask_around_des=mask_around_des)
        self.cut_DESI = cut_DESI

        # which region we want to use --> if None use default
        self.region = region
        if self.region is None:
            logger.info('Use default region')
            self.region = ['North', 'South', 'Des']

        logger.info(f"Version: {self.version} -- Tracer: {self.tracer} -- REGION: {self.region}")

        if not self.output_dir is None:
            self.output = os.path.join(output_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}')
            if os.path.isdir(self.output):
                logger.info('Ouput folder already exists')
            else:
                logger.info(f'Create output directory: {self.output}')
                os.mkdir(self.output)
                os.mkdir(os.path.join(self.output, 'Build_dataFrame'))
            logger.info(f"Plots are saved in {self.output}")
        else:
            self.output = None


    def set_features(self, pixmap=None, sgr_stream=None, sel_columns=None, use_sgr_stream=True):
        """
        Set photometric templates and footprint info either from a pixweight array (already loaded) or read it from .fits file
        All the map should be HEALPIX MAPS with NSIDE in NESTED order

        Parameters
        ----------
        pixmap: float array dtype
            Array containg the photometric template at the correct Nside (ie) self.Nside

        sgr_stream: float array
            Array containing the Sgr. Stream feature at the compatible Nside

        path_pixweight: str
            todo

        path_sgr_stream: str
            todo

        use_sgr_stream: bool
            Include or not the Sgr. Stream map --> the feature is really relevant for the QSO TS.

        """

        path_pixweight, path_sgr_stream = None, None

        # Build DR9 Legacy Imaging footprint
        footprint = self.DR9.load_footprint()
        if self.cut_DESI: #restricted to DESI footprint
            logger.info('Restrict footrpint to DESI footprint')
            footprint[hp_in_box(self.Nside, [0, 360, -90, -30])] = False

        # extract the different region from DR9_footprint
        north, south, des = self.DR9.load_photometry()
        _, south_mid, south_pole = self.DR9.load_elg_region()
        des_mid = des & ~south_pole
        ngc, sgc = self.DR9.load_ngc_sgc()
        self.footprint = pd.DataFrame({'FOOTPRINT': footprint, 'ISNORTH':north, 'ISSOUTH':south, 'ISDES':des, 'ISSOUTHWITHOUTDES':south&~des,
                                       'ISNGC':ngc, 'ISSGC':sgc, 'ISSOUTHMID':south_mid, 'ISSOUTHPOLE':south_pole, 'ISDESMID':des_mid})

        if sel_columns is None:
            sel_columns = ['STARDENS', 'EBV',
                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                           'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']

        if isinstance(pixmap, str):
            path_pixweight = pixmap
        elif pixmap is None:
            path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.Nside}.fits')

        if not path_pixweight is None:
            logger.info(f"Read {path_pixweight}")
            feature_pixmap = pd.DataFrame(fitsio.FITS(path_pixweight)[1][sel_columns].read().byteswap().newbyteorder())
        else:
            feature_pixmap = pixmap[sel_columns]

        if use_sgr_stream:
            if isinstance(sgr_stream, str):
                path_sgr_stream = sgr_stream
            elif sgr_stream is None:
                path_sgr_stream = os.path.join(self.data_dir, f'sagittarius_stream_{self.Nside}.npy')

            if not path_sgr_stream is None:
                # Load Sgr. Stream map
                logger.info(f"Read {path_sgr_stream}")
                stream_map = pd.DataFrame(np.load(path_sgr_stream), columns=['STREAM'])
            else:
                stream_map = pd.DataFrame(sgr_stream, columns=['STREAM'])
            self.features = pd.concat([stream_map, feature_pixmap], axis=1)
        else:
            self.features = feature_pixmap
        logger.info(f"Sanity check: Number of Nan in features: {self.features.isnull().sum().sum()}")


    def set_targets(self, targets=None, fracarea=None):
        """
        Set targets and fracarea map at the correct Nside (ie) self.Nside
        All the map should be HEALPIX MAPS with NSIDE in NESTED order

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
            path_targets = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy')

        if not path_targets is None:
            logger.info(f"Read {path_targets}")
            targets = np.load(path_targets)
        self.targets = targets

        if isinstance(fracarea, str):
            path_fracarea = fracarea
        elif fracarea is None:
            path_fracarea = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_fracarea_{self.Nside}.npy')

        if not path_fracarea is None:
            if os.path.isfile(path_fracarea):
                logger.info(f"Read {path_fracarea}")
                fracarea = np.load(path_fracarea)
            else:
                # Read fracarea_12290 from pixweight file
                logger.info("Do not find corresponding fracarea map --> USE FRACAREA_12290 AS DEFAULT FRACAREA")
                path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.Nside}.fits')
                logger.info(f"Read {path_pixweight}")
                fracarea = fitsio.FITS(path_pixweight)[1]['FRACAREA_12290'].read()
        self.fracarea = fracarea


    def build_for_regressor(self, selection_on_fracarea=False):
        """
        Build the normalized target density in the considered zone and Choose the pixel to use during the training (clean and remove 'bad' pixels)

        Parameters
        ----------

        selection_on_fracarea: bool
            if True remove queue distribution of the fracarea --> not mandatory since it can be already done when building the target density map (with more specificity) especially for DA02 ect..

        """

        # use only pixels which are observed for the training
        # self.footprint can be an approximation of the true area where observations were conducted
        # use always fracarea > 0 to use observed pixels
        considered_footprint = (self.fracarea > 0) & self.footprint['FOOTPRINT'].values
        keep_to_train = considered_footprint.copy()

        if selection_on_fracarea:
            if self.Nside == 512:
                min_fracarea, max_fracarea = 0.5, 1.5
            else:
                min_fracarea, max_fracarea = 0.9, 1.1
            keep_to_train &= (self.fracarea > min_fracarea) & (self.fracarea < max_fracarea)

        logger.info(f"The considered footprint represents {(considered_footprint).sum() / (self.footprint['FOOTPRINT']).sum():2.2%} of the DR9 footprint")
        logger.info(f"They are {(~keep_to_train[considered_footprint]).sum()} pixels which will be not used for the training (ie) {(~keep_to_train[considered_footprint]).sum()/(considered_footprint).sum():2.2%} ot the considered footprint")

        # build normalized targets
        normalized_targets, mean_targets_density = np.zeros(self.targets.size) * np.NaN, dict()
        for zone_name in self.region:
            pix_zone = self.footprint[zone_name_to_column_name(zone_name)].values
            pix_to_use = pix_zone & keep_to_train

            # only conserve pixel in the correct radec box
            if self.use_new_norm:
                #compute normalization on subpart of the footprint which is not contaminated for the north and the south !
                keep_to_norm = np.zeros(hp.nside2npix(self.Nside))
                if zone_name == 'North':
                    keep_to_norm[hp_in_box(self.Nside, [120, 240, 32.2, 40], inclusive=True)] = 1
                elif zone_name == 'South':
                    keep_to_norm[hp_in_box(self.Nside, [120, 240, 24, 32.2], inclusive=True)] = 1
                else:
                    keep_to_norm = np.ones(hp.nside2npix(self.Nside))
                pix_to_use_norm = pix_to_use & keep_to_norm
            else:
                pix_to_use_norm = pix_to_use

            # compute the mean only on pixel with "correct" behaviour
            if not self.use_median:
                mean_targets_density_estimators = np.mean(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])
            else:
                mean_targets_density_estimators = np.median(self.targets[pix_to_use_norm] / self.fracarea[pix_to_use_norm])

            #compute normalized_targets every where but we don't care we only use keep_to_train == 1 during the training
            normalized_targets[pix_zone] = self.targets[pix_zone] / (self.fracarea[pix_zone]*mean_targets_density_estimators)
            mean_targets_density[zone_name] = mean_targets_density_estimators
            logger.info(f"  ** {zone_name}: {mean_targets_density_estimators:2.2f} -- {normalized_targets[pix_to_use_norm].mean():1.4f} -- {normalized_targets[pix_to_use].mean():1.4f}")

        # some plots for sanity check
        if not self.output is None:
            plt.figure(figsize=(8,6))
            plt.hist(self.targets[considered_footprint], range=(0.1,100), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
            plt.close()

            plt.figure(figsize=(8,6))
            plt.hist(self.fracarea[considered_footprint], range=(0.5, 1.4), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_fracarea_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
            plt.close()

            tmp = np.zeros(hp.nside2npix(self.Nside))
            tmp[self.pixels[keep_to_train == False]] = 1
            plt.figure(figsize=(8,6))
            hp.mollview(tmp, rot=120, nest=True, title='strange pixel', cmap='jet')
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"strange_pixel_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

            plt.figure(figsize=(8,6))
            plt.hist(normalized_targets[keep_to_train], range=(0.1,5), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"normalized_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

        self.density = normalized_targets
        self.mean_density_region = mean_targets_density
        self.keep_to_train = keep_to_train


# class SpectroscopyDataFrame(object):
#     """
#     Build the dataframe needed to compute the combined weights for photometry and spectroscopy systematic effects
#     """
#     def __init__(self, version, tracer, data_dir, output_dir=None, suffixe_tracer='',
#                  Nside=None, use_median=False, use_new_norm=False, remove_LMC=False,
#                  clear_south=True, cut_DESI=False, region=None):
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
