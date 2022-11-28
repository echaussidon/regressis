#!/usr/bin/env python
# coding: utf-8

import sys
import os
import time
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

        self.data_dir = data_dir  # where maps are saved -> usefull only if you do not specified the path of the files in set_features / set_targets ...

        if output_dir is not None:  # if None --> nothing is save and no directory is built
            self.output_dir = os.path.join(output_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}')
            self.output_dataframe_dir = os.path.join(self.output_dir, 'dataframe')
            utils.mkdir(self.output_dir)
            utils.mkdir(self.output_dataframe_dir)
            logger.info(f"Plots are saved in {self.output_dataframe_dir}")
        else:
            self.output_dir = self.output_dataframe_dir = None

    def set_features(self, pixmap=None, sel_columns=None, pixmap_external=None, sel_columns_external=None, use_sgr_stream=True, sgr_stream=None, features_toplot=None):
        """
        Set photometric templates info either from a pixweight array (already loaded) or read it from .fits file
        All the maps should be Healpix maps with :attr:`nside` in nested order.

        Parameters
        ----------
        pixmap : float array or str, default=None
            Array containing the photometric templates or the path to .fits file containing the photometric templates.
        sel_columns : list of str, default=None
            List containing which photometric features must be extracted from the pixmap.
        pixmap_external: float array or str, default=None
            Array containing additional templates in the same format than pixmap (same nside and order) or the path to .fits file containing the templates.
        sel_columns_external: list of str, default=None
            List containing which templates must be extracted from the pixmap_external.
        use_sgr_stream : bool, default=True
            Include or not the Sgr. Stream map --> the feature is very relevant for the QSO TS.
        sgr_stream : float array or str, default=None
            Array containing the Sgr. Stream feature or the path to .npy file containing the Sgr. Stream feature.
        features_toplot : list of str
            list of features to plot in the systematic plots. If None, the plot is done for each feature in self.features
        """
        path_pixweight, path_pixweight_external, path_sgr_stream = None, None, None

        # default columns for the legacy imaging templates
        if sel_columns is None:
            sel_columns = ['STARDENS', 'EBV',
                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                           'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z']
        # default columns for the external templates
        if sel_columns_external is None:
            sel_columns_external = ['KAPPAPLANK', 'HALPHA', 'EBVext',
                                    'CALIBG', 'CALIBR', 'CALIBZ',
                                    'EBVreconMEANF6', 'EBVreconMEANF15']

        if isinstance(pixmap, str):
            path_pixweight = pixmap
        elif pixmap is None:
            path_pixweight = os.path.join(self.data_dir, f'pixweight-dr9-{self.nside}.fits')
        if path_pixweight is not None:
            logger.info(f"Read {path_pixweight}")
            feature_pixmap = pd.DataFrame(fitsio.FITS(path_pixweight)[1][sel_columns].read().byteswap().newbyteorder())[sel_columns]
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
                sgr_stream = np.load(path_sgr_stream)
            feature_pixmap.insert(2, 'STREAM', sgr_stream)

        if isinstance(pixmap_external, str):
            path_pixweight_external = pixmap_external
        if path_pixweight_external is not None:
            logger.info(f"Read {path_pixweight_external}")
            feature_pixmap = pd.concat([feature_pixmap,
                                        pd.DataFrame(fitsio.FITS(path_pixweight_external)[1][sel_columns_external].read().byteswap().newbyteorder())],
                                       axis=1)

        self.features = feature_pixmap
        self.features_toplot = features_toplot
        if features_toplot is None:
            self.features_toplot = feature_pixmap.columns

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

        if path_targets is not None:
            logger.info(f"Read {path_targets}")
            targets = np.load(path_targets)
        self.targets = targets

        if isinstance(fracarea, str):
            path_fracarea = fracarea
        elif fracarea is None:
            path_fracarea = os.path.join(self.data_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_fracarea_{self.nside}.npy')

        if path_fracarea is not None:
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
            when building the target density map (and the corresponding fracarea) with more specificity, e.g. for DA02.
            Max fracarea can be strictly > 1 due to Poisson noise.

        fracarea_limits : tuple, list, default=None
            If a tuple or list, min and max limits for fracarea.
        """
        # use only pixels which are observed for the training
        # self.footprint can be an approximation of the true area where observations were conducted
        # use always fracarea > 0 to use observed pixels
        # remove also pixel with 0 targets --> it should be already removed with fracarea > 0 in targets case
        # but not always with real desi data which have low fracarea at the beginning...
        considered_footprint = (self.fracarea > 0) & (self.targets > 0) & self.footprint('footprint')
        keep_to_train = considered_footprint.copy()

        if cut_fracarea:
            if isinstance(fracarea_limits, (tuple, list)):
                min_fracarea, max_fracarea = fracarea_limits
            elif self.nside >= 512:  # can be cirvumvent increasing the number of randoms...
                min_fracarea, max_fracarea = 0.85, 1.15
            else:
                min_fracarea, max_fracarea = 0.9, 1.1
            keep_to_train &= (self.fracarea > min_fracarea) & (self.fracarea < max_fracarea)

        # file to load DR9 footprint is roughly what we expect to be DR9. At the border, it is expected to have pixels with fracarea == 0 and which are in DR9 Footprint
        # {(considered_footprint).sum() / self.footprint('footprint').sum():2.2%} > 99.9 % is similar than 100 %.
        logger.info(f"The considered footprint represents {(considered_footprint).sum() / self.footprint('footprint').sum():2.2%} of the DR9 footprint")
        logger.info(f"They are {(~keep_to_train[considered_footprint]).sum()} pixels which will be not used for the training i.e. {(~keep_to_train[considered_footprint]).sum()/(considered_footprint).sum():2.2%} of the considered footprint")

        # build normalized targets
        normalized_targets, mean_targets_density = np.zeros(self.targets.size) * np.nan, dict()
        for region_name in self.regions:
            pix_region = self.footprint(region_name)
            pix_to_use = pix_region & keep_to_train

            if self.use_new_norm:
                # compute normalization on subpart of the footprint (for instance which is expected to be free from stellar contamination)
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
            # To avoid the warning raised: RuntimeWarning: invalid value encountered in true_divide
            with np.errstate(divide='ignore', invalid='ignore'):
                normalized_targets[pix_region] = self.targets[pix_region] / (self.fracarea[pix_region] * mean_targets_density_estimators)
            mean_targets_density[region_name] = mean_targets_density_estimators
            logger.info(f"  ** {region_name}: {mean_targets_density_estimators:2.2f} -- {normalized_targets[pix_to_use_norm].mean():1.4f} -- {normalized_targets[pix_to_use].mean():1.4f}")

        # some plots for sanity check
        if self.output_dataframe_dir is not None:
            plt.figure(figsize=(8, 6))
            plt.hist(self.targets[considered_footprint], range=(0.1, 100), bins=100, label='without any selection')
            plt.xlabel(f'nbr of objects per healpix at nside={self.nside}')
            plt.ylabel('nbr of pixels')
            plt.legend()
            plt.savefig(os.path.join(self.output_dataframe_dir, f"test_remove_targets_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

            plt.figure(figsize=(8, 6))
            plt.hist(self.fracarea[considered_footprint], range=(0.1, 1.2), bins=100, label='without any selection')
            plt.xlabel('fracarea')
            plt.ylabel('nbr of pixels')
            plt.legend()
            plt.savefig(os.path.join(self.output_dataframe_dir, f"test_remove_fracarea_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

            tmp = np.zeros(hp.nside2npix(self.nside))
            tmp[self.pixels[keep_to_train == 0]] = 1
            plot_moll(tmp, show=False, label='strange pixel', filename=os.path.join(self.output_dataframe_dir, f"strange_pixel_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"), galactic_plane=True, ecliptic_plane=True)

            plt.figure(figsize=(8, 6))
            for region_name in self.regions:
                pix_region = self.footprint(region_name)
                plt.hist(normalized_targets[pix_region & keep_to_train], range=(0.1, 5), bins=100, label=f'keep to train in {region_name}')
            plt.xlabel('normalized target counts')
            plt.ylabel('nbr of pixels')
            plt.legend()
            plt.savefig(os.path.join(self.output_dataframe_dir, f"normalized_targets_{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}.png"))
            plt.close()

        self.density = normalized_targets
        self.mean_density_region = mean_targets_density
        self.keep_to_train = keep_to_train


def _format_kmeans_params(params, regions=None):
    """
    Format kmeans hyperparameters, i.e. return a dictionary of dictionary of parameters for each input region.

    Parameters
    ----------
    params : dict, default=None
        Parameters.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for MiniBtachKmeans or Kmeans implemenation.
    """

    params = params or {}

    if regions is None: regions = ['North', 'South', 'Des', 'South_all', 'South_mid', 'South_mid_ngc',
                                   'South_mid_sgc', 'South_pole', 'South_mid_no_des', 'Des_mid']
    if not any(region in params for region in regions):
        params = {region: params.copy() for region in regions}
    return params


def _get_kmean_params(params=None, n_clusters=100, seed=123, regions=None):
    """
    Return pre-defined hyperparameters for the :class:`MiniBatchKmeans` for each region.
    Can be updated with ``params``.

    Parameters
    ----------
    params : dict, default=None
        Override parameters, e.g. {'North': {'n_estimators': 20}}.
    n_clusters : int, default=6
        Number of clusters to be computed
    seed : int, default=123
        Fix the random state for reproducibility.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for the Kmeans.
    """

    default = _format_kmeans_params({'n_clusters': n_clusters, 'random_state': seed,
                                     'reassignment_ratio': 1e-3, 'n_init': 5, 'max_no_improvement': 30},
                                    regions=regions)

    utils.deep_update(default, _format_kmeans_params(params or {}, regions=regions))

    return default


class KmeansDataFrame(object):

    """TODO: Container gathering target density and photometric templates."""

    def __init__(self, version, tracer, footprint, suffix_tracer='',
                 data_dir=None, output_dir=None, regions=None, kmeans_features=None):
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
        regions : list of str, default=None
            List of regions in which we want to apply the systematic mitigation procedure. The normalized target density
            is computed and the regression is applied independantly in each regions. If none use the default regions(s) given in footprint.
        """
        self.version = version
        self.tracer = tracer
        self.suffix_tracer = suffix_tracer

        self.footprint = footprint
        self.nside = footprint.nside

        # which regions we want to use --> if None use default regions defined in footprint
        self.regions = regions
        if self.regions is None:
            self.regions = self.footprint.default_regions
            logger.info(f'Using default regions {self.regions}')

        logger.info(f"version: {self.version} -- tracer: {self.tracer} -- regions: {self.regions}")

        self.data_dir = data_dir  # where maps are saved -> usefull only if you do not specified the path of the files in set_features / set_targets ...

        if output_dir is not None:  # if None --> nothing is save and no directory is built
            self.output_dir = os.path.join(output_dir, f'{self.version}_{self.tracer}{self.suffix_tracer}_{self.nside}')
            self.output_dataframe_dir = os.path.join(self.output_dir, 'KMeansDataframe')
            utils.mkdir(self.output_dir)
            utils.mkdir(self.output_dataframe_dir)
            logger.info(f"Plots are saved in {self.output_dataframe_dir}")
        else:
            self.output_dir = self.output_dataframe_dir = None

        # the columns used during the kmeans
        self.kmeans_features = kmeans_features

    def set_data(self, data, kmeans_features, features_toplot=None):
        """
            TODO.
        """

        if self.kmeans_features is None:
            self.kmeans_features = kmeans_features
        else:
            if self.kmeans_features != kmeans_features:
                logger.error('Features used in Kmeans have to be the same and in the same order.')
                sys.exit()
        if features_toplot is None:
            self.features_toplot = self.kmeans_features
        else:
            self.features_toplot = features_toplot

        logger.info(f'Use {data.shape[0]} data:')
        self.data = data
        # standardize the data in order to use euclidian metric
        # only save columns which are normalized
        self.data_norm = data[self.kmeans_features].copy()

        for region in self.regions:
            is_in_region = self.footprint(region)[self.data['HPX']]
            self.data_norm.loc[is_in_region] = (self.data[self.kmeans_features][is_in_region] - self.data[self.kmeans_features][is_in_region].mean()) / self.data[self.kmeans_features][is_in_region].std()
            logger.info(f"    * Standardization: mean of mean and std in {region}: {self.data_norm[is_in_region].mean().mean():2.4f} -- {self.data_norm[is_in_region].std().mean():2.2f}")

        # plot for sanity check
        if self.output_dataframe_dir is not None:
            to_plot = hp.ud_grade(utils.build_healpix_map(self.nside, self.data['RA'], self.data['DEC'], precomputed_pix=self.data['HPX'], in_deg2=True), 64, order_in='NESTED')
            to_plot[~(to_plot > 0)] = np.nan
            mean = np.nanmean(to_plot)
            to_plot /= mean
            plot_moll(to_plot, galactic_plane=True, min=0.5, max=1.5, figsize=(7, 5), label='nbr / mean',
                      show=False, filename=os.path.join(self.output_dataframe_dir, f"density_data_{self.version}_{self.tracer}{self.suffix_tracer}.png"))

    def set_randoms(self, randoms, kmeans_features):
        """
            TODO.
        """

        if self.kmeans_features is None:
            self.kmeans_features = kmeans_features
        else:
            if self.kmeans_features != kmeans_features:
                logger.error('Features used in Kmeans have to be the same and in the same order.')
                sys.exit()

        logger.info(f'Use {randoms.shape[0]} randoms:')
        self.randoms = randoms

        # standardize the data in order to use euclidian metric
        # only save columns which are normalized
        self.randoms_norm = randoms[self.kmeans_features].copy()
        for region in self.regions:
            is_in_region = self.footprint(region)[self.randoms['HPX']]
            self.randoms_norm.loc[is_in_region] = (self.randoms[self.kmeans_features][is_in_region] - self.randoms[self.kmeans_features][is_in_region].mean()) / self.randoms[self.kmeans_features][is_in_region].std()
            logger.info(f"    * Standardization: mean of mean and std in {region}: {self.randoms_norm[is_in_region].mean().mean():2.4f} -- {self.randoms_norm[is_in_region].std().mean():2.2f}")

        # plot for sanity check
        if self.output_dataframe_dir is not None:
            to_plot = hp.ud_grade(utils.build_healpix_map(self.nside, self.randoms['RA'], self.randoms['DEC'], precomputed_pix=self.randoms['HPX'], in_deg2=True), 64, order_in='NESTED')
            to_plot[~(to_plot > 0)] = np.nan
            mean = np.nanmean(to_plot)
            to_plot /= mean
            plot_moll(to_plot, galactic_plane=True, min=0.5, max=1.5, figsize=(7, 5), label='nbr / mean',
                      show=False, filename=os.path.join(self.output_dataframe_dir, f"density_randoms_{self.version}_{self.tracer}{self.suffix_tracer}.png"))

    def build(self, split_kmeans=False, fit_with_randoms=True, kmeans_params=None, kmeans_params_2=None, plot=False):
        """
            Compute the Kmeans only once per region. This can be very slow with a lot of clusters...

            In order to decrease the computation time, we perform the Kmeans twice (ie) build a first set a cluster and then re-apply
            the Kmeans in each cluster to build smaller clusters more numerous.

        """
        from sklearn.cluster import MiniBatchKMeans as KMeans

        # load kmeans_params
        kmeans_params = _get_kmean_params(kmeans_params or {}, regions=self.regions)
        if split_kmeans: kmeans_params_2 = _get_kmean_params(kmeans_params_2 or {}, regions=self.regions)

        # add final attributes:
        self.features, self.density, self.keep_to_train, self.pixels, self.fracarea = pd.DataFrame([]), np.array([]), np.array([], dtype='bool'), np.array([], dtype='int'), np.array([])

        # add columns to the dataframe at the begining to work in each region independantly
        self.data['LABELS'], self.randoms['LABELS'] = np.nan * np.zeros(self.data.shape[0]), np.nan * np.zeros(self.randoms.shape[0])

        # save the number of clusters
        self.n_clusters = {region: 0 for region in self.regions}

        for region in self.regions:
            # which data and randoms are in the current region:
            data_in_region, randoms_in_region = self.footprint(region)[self.data['HPX']], self.footprint(region)[self.randoms['HPX']]

            # Build Kmeans and fit it on the randoms:
            logger.info(f'Fit Kmeans with randoms and predict the label in the dataset on region: {region}:')
            logger.info(f'    * Use MiniBatchKmeans with: {kmeans_params[region]}')
            kmeans = KMeans(**kmeans_params[region])
            """# TODO --> typicallu to remove consatntn feature -> normalisation is broken but whatever the featueres is useless
            ADD ALSO DATA BECAUSE DATA ARE LESS numerous --> constant values are so more probable when increase the number of clusters ..."""
            good_kmeans_features = [feature for feature in self.kmeans_features if (self.randoms[feature][randoms_in_region].std() != 0) & (self.data[feature][data_in_region].std() != 0)]

            if fit_with_randoms:
                t0 = time.time()
                kmeans.fit(self.randoms_norm[good_kmeans_features][randoms_in_region])
                self.randoms.loc[randoms_in_region, 'LABELS'] = kmeans.labels_
                logger.info(f"    * Fit done in {time.time() - t0:2.2f}s. with {randoms_in_region.sum()} randoms, {kmeans_params[region]['n_clusters']} clusters and {len(self.kmeans_features)} features.")

                # Predict the cluster numbers on the data:
                t0 = time.time()
                self.data.loc[data_in_region, 'LABELS'] = kmeans.predict(self.data_norm[good_kmeans_features][data_in_region])
                logger.info(f"    * Predict done in {time.time() - t0:2.2f}s. with {data_in_region.sum()} data, {kmeans_params[region]['n_clusters']} clusters and {len(self.kmeans_features)} features.")
            else:
                t0 = time.time()
                kmeans.fit(self.data_norm[good_kmeans_features][data_in_region])
                self.data.loc[data_in_region, 'LABELS'] = kmeans.labels_
                logger.info(f"    * Fit done in {time.time() - t0:2.2f}s. with {data_in_region.sum()} data, {kmeans_params[region]['n_clusters']} clusters and {len(self.kmeans_features)} features.")

                # Predict the cluster numbers on the data:
                t0 = time.time()
                self.randoms.loc[randoms_in_region, 'LABELS'] = kmeans.predict(self.randoms_norm[good_kmeans_features][randoms_in_region])
                logger.info(f"    * Predict done in {time.time() - t0:2.2f}s. with {randoms_in_region.sum()} randoms, {kmeans_params[region]['n_clusters']} clusters and {len(self.kmeans_features)} features.")

            # save number of clusters
            self.n_clusters[region] = kmeans_params[region]['n_clusters']

            if split_kmeans:
                logger.info(f"    * Fit one Kmeans on each of the {kmeans_params[region]['n_clusters']} clusters with: {kmeans_params_2[region]}")

                # build final labels for data and randoms:
                data_labels, randoms_labels = np.nan * np.zeros_like(self.data['LABELS'].values), np.nan * np.zeros_like(self.randoms['LABELS'].values)

                # run one kmeans on each cluster determined by the first kmeans:
                t0 = time.time()
                for i in range(kmeans_params[region]['n_clusters']):
                    # select which randoms / data are in the current cluster and in the correct region:
                    in_clust_randoms, in_clust_data = randoms_in_region & (self.randoms['LABELS'].values == i), data_in_region & (self.data['LABELS'].values == i)

                    # Since we want to recompute a Kmeans, we need to standardize the local dataframe once on each cluster !
                    # Otherwise, features used during the first kmeans will have less impact --> this can have strong impact during the mitigation!
                    self.randoms_norm.loc[in_clust_randoms] = (self.randoms[self.kmeans_features][in_clust_randoms] - self.randoms[self.kmeans_features][in_clust_randoms].mean()) / self.randoms[self.kmeans_features][in_clust_randoms].std()
                    self.data_norm.loc[in_clust_data] = (self.data[self.kmeans_features][in_clust_data] - self.data[self.kmeans_features][in_clust_data].mean()) / self.data[self.kmeans_features][in_clust_data].std()

                    # Build Kmeans and fit it on the randoms and predict:
                    kmeans = KMeans(**kmeans_params_2[region])
                    good_kmeans_features = [feature for feature in self.kmeans_features if (self.randoms[feature][in_clust_randoms].std() != 0) & (self.data[feature][in_clust_data].std() != 0)]

                    if fit_with_randoms:
                        kmeans.fit(self.randoms_norm[good_kmeans_features][in_clust_randoms])
                        randoms_labels[in_clust_randoms] = i * kmeans_params_2[region]['n_clusters'] + kmeans.labels_
                        data_labels[in_clust_data] = i * kmeans_params_2[region]['n_clusters'] + kmeans.predict(self.data_norm[good_kmeans_features][in_clust_data])
                    else:
                        kmeans.fit(self.data_norm[good_kmeans_features][in_clust_data])
                        data_labels[in_clust_data] = i * kmeans_params_2[region]['n_clusters'] + kmeans.labels_
                        randoms_labels[in_clust_randoms] = i * kmeans_params_2[region]['n_clusters'] + kmeans.predict(self.randoms_norm[good_kmeans_features][in_clust_randoms])

                # save number of clusters
                self.n_clusters[region] *= kmeans_params_2[region]['n_clusters']

                # save the new labels:
                self.data.loc[data_in_region, 'LABELS'], self.randoms.loc[randoms_in_region, 'LABELS'] = data_labels[data_in_region], randoms_labels[randoms_in_region]
                logger.info(f"    * Fit and predict {kmeans_params[region]['n_clusters']} Kmeans done in {time.time() - t0:2.2f}s.")

            # build features and density:
            t0 = time.time()
            data_group, randoms_group = self.data[data_in_region].groupby('LABELS'), self.randoms[randoms_in_region].groupby('LABELS')

            # Sometime no data are in the cluster --> problem with groupby:
            # count the number of data and randoms in each clsuter. Take care sometimes a cluster do not have any data/randoms --> need to fill it with 0 by hands.
            randoms_counts = randoms_group.size()
            # randoms_counts.reindex(np.arange(0, int(kmeans_params[region]['n_clusters'] * kmeans_params_2[region]['n_clusters']), 1), fill_value=0)
            data_counts = data_group.size().reindex(randoms_counts.index, fill_value=0)
            density = data_counts / randoms_counts
            # keep only pixels with data inside
            logger.warning('        ** Keep only pixels in feature space with data inside (Correct or not ?? -> TO TEST) and density < 2.2 ')
            keep_to_train = (density > 0) & (density < 2.2)
            density /= np.mean(density[keep_to_train])

            print(density)
            print(np.mean(density[keep_to_train]))
            print(density.mean())
            print((density == 0).sum())

            # use randoms to find the mean value inside each cluster
            self.features = pd.concat([self.features, randoms_group.mean()], ignore_index=True)
            # normalized density (around 1)
            self.density = np.concatenate([self.density, density])
            # pixels used for the training
            self.keep_to_train = np.concatenate([self.keep_to_train, keep_to_train])
            # give a caduc pixel value which is in the correct region
            self.pixels = np.concatenate([self.pixels, int(self.data['HPX'][data_in_region].values[0]) * np.ones(density.size, dtype='int')])
            # fracarea, in order to plot the systematic maps !
            self.fracarea = np.concatenate([self.fracarea, np.mean(randoms_counts) / randoms_counts])
            logger.info(f"    * Build features done in {time.time() - t0:2.2f}s.")

        plt.figure()
        for region in self.regions:
            clust_in_region = self.footprint(region)[self.pixels]
            plt.hist(density[clust_in_region], bins=100, range=(0, 5), label=f'{region}: {np.mean(density[keep_to_train & clust_in_region])}:2.2f')
        plt.legend()
        # plt.savefig(os.path.join(self.output_dataframe_dir, 'test_density.png'))
        plt.savefig('test_density.png')
        plt.close()

        # plot for sanity check
        if (self.output_dataframe_dir is not None) & plot:

            plt.figure()
            for region in self.regions:
                clust_in_region = self.footprint(region)[self.pixels]
                plt.hist(density[clust_in_region], bins=100, range=(0, 5), label=f'{region}')
            plt.legend()
            plt.savefig(os.path.join(self.output_dataframe_dir, 'test_density.png'))
            plt.close()

            plt.figure(figsize=(10, 4))
            for region in self.regions:
                data_in_region, randoms_in_region = self.footprint(region)[self.data['HPX']], self.footprint(region)[self.randoms['HPX']]
                plt.subplot(121)
                plt.hist(self.data['LABELS'][data_in_region].values, bins=int(np.max(self.data['LABELS'][data_in_region]) + 1), alpha=0.8, density=1, label=region)
                plt.subplot(122)
                plt.hist(self.randoms['LABELS'][randoms_in_region].values, bins=int(np.max(self.randoms['LABELS'][randoms_in_region]) + 1), alpha=0.8, density=1, label=region)
            plt.subplot(121)
            plt.xlabel('cluster id')
            plt.ylabel('number of objects')
            plt.legend(title='data:')
            plt.subplot(122)
            plt.xlabel('cluster id')
            plt.ylabel('number of objects')
            plt.legend(title='randoms:')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dataframe_dir, f"nbr_objects_per_clusters_{self.version}_{self.tracer}{self.suffix_tracer}.png"))
            plt.close()

            # Check if we find the same value in the cluster data and randoms data:
            dd_norm, rr_norm = self.data_norm.copy(), self.randoms_norm.copy()
            dd_norm['LABELS'], rr_norm['LABELS'] = self.data['LABELS'], self.randoms['LABELS']
            dd_group_norm = dd_norm[self.footprint(self.regions[0])[self.data['HPX']]].groupby('LABELS')
            rr_group_norm = rr_norm[self.footprint(self.regions[0])[self.randoms['HPX']]].groupby('LABELS')
            mean_dd_norm, mean_rr_norm = dd_group_norm.mean(), rr_group_norm.mean()
            std_dd_norm, std_rr_norm = dd_group_norm.std(), rr_group_norm.std()
            sel = np.isin(mean_rr_norm.index, mean_dd_norm.index)
            subsamp = int(sel.sum() / 500)
            with np.errstate(invalid='ignore'):
                plt.figure(figsize=(20, 10))
                for i, name in enumerate(self.kmeans_features):
                    plt.subplot(2, 6, i + 1)
                    plt.errorbar(mean_rr_norm[sel][name][::subsamp], mean_dd_norm[name][::subsamp], xerr=std_rr_norm[sel][name][::subsamp], yerr=std_dd_norm[name][::subsamp], fmt="o", zorder=0)
                    plt.xlabel(f'randoms {name}')
                    plt.ylabel(f'data {name}')
                    plt.xlim([-2, 2])
                    plt.ylim([-2, 2])
                    plt.plot(np.linspace(-2, 2, 100), np.linspace(-2, 2, 100), ls='--', c='k', zorder=10)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dataframe_dir, f"mean_features_{self.version}_{self.tracer}{self.suffix_tracer}_{self.regions[0]}.png"))
                plt.close()
