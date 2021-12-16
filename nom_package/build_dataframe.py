# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import logging
logger = logging.getLogger("build_dataframe")

import sys, os
# to avoid error from pandas method into the logger -> pandas use NUMEXPR Package
os.environ['NUMEXPR_MAX_THREADS'] = '8'
os.environ['NUMEXPR_NUM_THREADS'] = '8'

import numpy as np
import healpy as hp
import fitsio
import pandas as pd

import matplotlib.pyplot as plt

from desi_footprint import DR9_footprint
from utils import hp_in_box, zone_name_to_column_name


class PhotometricDataFrame(object):
    """
    Build the dataframe needed to compute the weights due to photometry / spectroscopy systematic effects
    """
    def __init__(self, version, tracer, data_dir=None, output_dir=None, suffixe_tracer='',
                 Nside=None, use_median=False, use_new_norm=False, remove_LMC=False,
                 clear_south=True, cut_DESI=False, region=None):
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
        self.remove_LMC = remove_LMC
        self.clear_south = clear_south
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


    def set_features(self, pixmap=None, sgr_stream=None, path_pixweight=None, path_sgr_stream=None, use_sgr_stream=True):
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

        # Load DR9 Legacy Imaging footprint
        DR9 = DR9_footprint(self.Nside, remove_LMC=self.remove_LMC, clear_south=self.clear_south)
        footprint = DR9.load_footprint()
        if self.cut_DESI: #restricted to DESI footprint
            logger.info('Restrict footrpint to DESI footprint')
            footprint[hp_in_box(self.Nside, [0, 360, -90, -30])] = False

        # extract the different region from DR9_footprint
        north, south, des = DR9.load_photometry(remove_around_des=True)
        _, south_mid, south_pole = DR9.load_elg_region()
        des_mid = des & ~south_pole
        ngc, sgc = DR9.load_ngc_sgc()
        self.footprint = pd.DataFrame({'FOOTPRINT': footprint, 'ISNORTH':north, 'ISSOUTH':south, 'ISDES':des, 'ISSOUTHWITHOUTDES':south&~des,
                                       'ISNGC':ngc, 'ISSGC':sgc, 'ISSOUTHMID':south_mid, 'ISSOUTHPOLE':south_pole, 'ISDESMID':des_mid})

        sel_columns = ['STARDENS', 'EBV',
                       'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                       'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                       'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']

        if not pixmap is None:
            feature_pixmap = pixmap[sel_columns]
        else:
            if path_pixweight is None:
                path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.Nside}.fits')
            # Load pixmap
            logger.info(f"Read {path_pixweight}")
            feature_pixmap = pd.DataFrame(fitsio.FITS(path_pixweight)[1][sel_columns].read().byteswap().newbyteorder())

        if use_sgr_stream:
            if not sgr_stream is None:
                stream_map = pd.DataFrame(sgr_stream, columns=['STREAM'])
            else:
                if path_sgr_stream is None:
                    path_sgr_stream = os.path.join(self.data_dir, f'sagittarius_stream_{self.Nside}.npy')
                # Load Sgr. Stream map
                logger.info(f"Read {path_sgr_stream}")
                stream_map = pd.DataFrame(np.load(path_sgr_stream), columns=['STREAM'])
            self.features = pd.concat([stream_map, feature_pixmap], axis=1)
        else:
            self.features = feature_pixmaps
        logger.info(f"Sanity check: Number of Nan in features: {self.features.isnull().sum().sum()}")


    def set_targets(self, targets=None, fracarea=None, path_targets=None, path_fracarea=None):
        """
        Set targets and fracarea map at the correct Nside (ie) self.Nside
        All the map should be HEALPIX MAPS with NSIDE in NESTED order

        Parameters
        ----------
        targets: float array
            Array containing the healpix map of the considered object density
        fracarea: float array
            Array containing the associated observed fraction area of a pixel of a healpix map
        path_targets: str
            if targets is not given, load this file as targets
        path_fracarea: str
            if fracarea is not given, load this file as fracarea
        """

        if targets is None:
            if path_targets is None:
                path_targets = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.npy')
            logger.info(f"Read {path_targets}")
            targets = np.load(path_targets)
        self.targets = targets

        if fracarea is None:
            if path_fracarea is None:
                path_fracarea = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffixe_tracer}_fracarea_{self.Nside}.npy')

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
        considered_footprint = (self.fracarea > 0) & self.footprint['FOOTPRINT']
        keep_to_train = considered_footprint.copy()

        if selection_on_fracarea:
            if self.Nside == 512:
                min_fracarea, max_fracarea = 0.5, 1.5
            else:
                min_fracarea, max_fracarea = 0.9, 1.1
            keep_to_train &= (self.fracarea > min_fracarea) & (self.fracarea < max_fracarea)

        # remove the queue of the target histogram
        keep_to_train &= (self.targets > np.quantile(self.targets[considered_footprint], 0.1)) & (self.targets < np.quantile(self.targets[considered_footprint], 0.9))

        logger.info(f"The considered footprint represents {(considered_footprint).sum() / (self.footprint['FOOTPRINT']).sum():2.2%} of the DR9 footprint")
        logger.info(f"They are {(~keep_to_train[considered_footprint]).sum()} pixels not considered for the training (ie) {(~keep_to_train[considered_footprint]).sum()/(considered_footprint).sum():2.2%} ot the considered footprint")

        # build normalized targets
        normalized_targets = np.zeros(self.targets.size) * np.NaN
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

            logger.info(f"  ** INFO for {zone_name}: {mean_targets_density_estimators:2.2f} -- {normalized_targets[pix_to_use_norm].mean():1.4f} -- {normalized_targets[pix_to_use].mean():1.4f}")

        # some plots for sanity check
        if not self.output is None:
            plt.figure(figsize=(8,6))
            plt.hist(self.targets, range=(0.1,100), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
            plt.close()

            plt.figure(figsize=(8,6))
            plt.hist(self.fracarea, range=(0.5, 1.4), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"test_remove_fracarea_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))
            plt.close()

            tmp = np.zeros(hp.nside2npix(self.Nside))
            tmp[self.pixels[keep_to_train == False]] = 1
            plt.figure(figsize=(8,6))
            hp.mollview(tmp, rot=120, nest=True, title='strange pixel', cmap='jet')
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"strange_pixel_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

            plt.figure(figsize=(8,6))
            plt.hist(normalized_targets[keep_to_train == 1], range=(0.1,5), bins=100)
            plt.savefig(os.path.join(self.output, 'Build_dataFrame', f"normalized_targets_{self.version}_{self.tracer}{self.suffixe_tracer}_{self.Nside}.png"))

        self.density = normalized_targets
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
