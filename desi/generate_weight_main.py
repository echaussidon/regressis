#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import logging

from regressis import PhotometricDataFrame, Regression, DR9Footprint, setup_logging
from regressis.utils import mkdir, setup_mplstyle


logger = logging.getLogger('MAIN')


def _compute_weight(version, tracer, footprint, suffix_tracer, seed, dataframe_params, max_plot_cart, feature_names=None):
    """
    Compute weight for a given tracer with a given parametrization.

    Parameters
    ----------
    version : str
        Which version you want to use: SV3 or MAIN (for SV3 / MAIN targets) or DA02 / Y1 / etc.
        Useful only to load default map saved in ``data_dir`` and for the output name of the directory or file name.
    tracer : str
        Which tracer you want to use. Useful only to load default map saved in data_dir and for
        the output name of the directory or file name.
    footprint : Footprint
        The footprint information specifying regions in an Healpix format.
    suffix_tracer : str, default=''
        Additional suffix for tracer. Useful only to load default map saved in ``data_dir`` and for
        the output name of the directory or file name.
    seed : int, default=123
        Fix the random state of RF and NN for reproducibility.
    dataframe_params : dict
        Dictionary with additional parameters to initialize :class:`PhotometricDataFrame`.
    max_plot_cart : float, default=400
        Maximum density in the plot of object density in the sky.
    feature_names: list of str
        Names of features used during the regression. If ``None``, use default one.
    """
    dataframe = PhotometricDataFrame(version, tracer, footprint, suffix_tracer, **dataframe_params)
    dataframe.set_features()
    dataframe.set_targets()
    dataframe.build(cut_fracarea=True)
    regression = Regression(dataframe, regressor='RF', n_jobs=40, use_kfold=True, feature_names=feature_names, compute_permutation_importance=True, overwrite=True, seed=seed, save_regressor=False)
    regression.get_weight_map(save=True)
    regression.plot_maps_and_systematics(max_plot_cart=max_plot_cart)


def _bgs_any_weight(seed):
    """Compute weight with standard parametrization for BGS in MAIN."""
    logger.info("Compute weight for BGS at nside = 256")

    version, tracer, suffix_tracer, nside = 'MAIN', 'BGS_ANY', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1800

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart)


def _bgs_faint_weight(seed):
    """Compute weight with standard parametrization for BGS FAINT in MAIN."""
    logger.info("Compute weight for BGS_FAINT at nside = 256")

    version, tracer, suffix_tracer, nside = 'MAIN', 'BGS_FAINT', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart)


def _bgs_bright_weight(seed):
    """Compute weight with standard parametrization for BGS BRIGHT in MAIN."""
    logger.info("Compute weight for BGS_BRIGHT at nside = 256")

    version, tracer, suffix_tracer, nside = 'MAIN', 'BGS_BRIGHT', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart)


def _lrg_weight(seed):
    """Compute weight with standard parametrization for LRG in MAIN."""
    logger.info("Compute weight for LRG at nside = 256")

    version, tracer, suffix_tracer, nside = 'MAIN', 'LRG', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart)


def _elg_weight(seed, add_stream=False):
    """Compute weight with standard parametrization for ELG in MAIN. If add_stream=True then add STREAM during the regression."""
    logger.info(f"Compute weight for ELG at nside = 512 with Sgr. Stream? {add_stream}")

    version, tracer, suffix_tracer, nside = 'MAIN', 'ELG', '', 512
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    if add_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        suffix_tracer = '_with_stream'
        dataframe_params['use_new_norm'] = True
    else:
        feature_names = None
    max_plot_cart = 3500

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart, feature_names)


def _elg_vlo_weight(seed, add_stream=False):
    """Compute weight with standard parametrization for ELG VLO in MAIN. If add_stream=True then add STREAM during the regression."""
    logger.info(f"Compute weight for ELG_VLO at nside = 256 with Sgr. Stream map? {add_stream}")

    version, tracer, suffix_tracer, nside = 'MAIN', 'ELG_VLO', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = False
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1500

    if add_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        suffix_tracer = '_with_stream'
        dataframe_params['use_new_norm'] = True
    else:
        feature_names = None

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart, feature_names)


def _elg_lop_weight(seed, add_stream=False):
    """Compute weight with standard parametrization for ELG LOP in MAIN. If add_stream=True then add STREAM during the regression."""
    logger.info("Compute weight for ELG at nside = 512 with Sgr. Stream map")

    version, tracer, suffix_tracer, nside = 'MAIN', 'ELG_LOP', '', 512
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = True
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 2500

    if add_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        suffix_tracer = '_with_stream'
        dataframe_params['use_new_norm'] = True
    else:
        feature_names = None

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart, feature_names)


def _qso_weight(seed):
    """Compute weight with standard parametrization for QSO in MAIN."""
    logger.info("Compute weight for QSO at nside = 256 with Sgr. Stream map")

    version, tracer, suffix_tracer, nside = 'MAIN', 'QSO', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=True, clear_south=True, mask_around_des=True, cut_desi=False)

    dataframe_params = dict()
    dataframe_params['data_dir'] = '../data'
    dataframe_params['output_dir'] = '../res/MAIN'
    dataframe_params['use_median'] = False
    dataframe_params['use_new_norm'] = True
    dataframe_params['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 400

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, dataframe_params, max_plot_cart)


if __name__ == '__main__':

    setup_logging(log_file='MAIN.log')
    setup_mplstyle()

    mkdir('../res/MAIN')

    #_bgs_any_weight(130)
    #_bgs_faint_weight(133)
    #_bgs_bright_weight(136)
    #_lrg_weight(140)
    #_elg_weight(150)
    # _elg_weight(155, add_stream=True)
    _elg_vlo_weight(160)
    # _elg_vlo_weight(165, add_stream=True)
    _elg_lop_weight(170)
    _elg_lop_weight(175, add_stream=True)
    _qso_weight(180)

    print("\nMOVE the MAIN.log file into the output directory ../res/MAIN\n")
    shutil.move('MAIN.log', '../res/MAIN/MAIN.log')
