#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import logging

from regressis import PhotometricDataFrame, Regression, DR9Footprint, setup_logging
from regressis.utils import mkdir, setup_mplstyle


logger = logging.getLogger('DA02')


def _compute_weight(version, tracer, footprint, suffix_tracer, suffix_regressor, cut_fracarea, seed, dataframe_params, max_plot_cart, feature_names=None):
    """

    Compute weight for a given tracer with a given parametrization

    Parameters
    ----------
    version: str
        Which version you want to use as SV3 or MAIN (for SV3 / MAIN targets) or DA02 / Y1 / etc. ...
        Useful only to load default map saved in data_dir and for the output name of the directory or filename.
    tracer: str
        Which tracer you want to use. Usefull only to load default map saved in data_dir and for
        the output name of the directory or file name.
    footprint: Footprint
        Contain all the footprint informations needed to extract the specific regions from an healpix map.
    suffix_tracer: str
        Additional suffix for tracer. Usefull only to load default map saved in data_dir and for
        the output name of the directory or filename.
    suffix_regressor : str
        Additional suffix to build regressor output directory. Useful to test on the same data different hyperparameters.
    cut_fracarea: bool
        If True create the dataframe with a selection on fracarea. In DA02, fracarea is already selected (set as nan where we don't want to use it) in the corresponding fracarea file.
    seed: int
        Fix the seed in ML algorithm for reproductibility
    param: dict
        dictionary with additional parameters to initialize :class:`PhotometricDataFrame`
    max_plot_cart: float
        Maximum value when plot map with plot_moll
    feature_names: list of str
        If not None use this list of feature during the regression otherwise use the default one.
    """
    dataframe = PhotometricDataFrame(version, tracer, footprint, suffix_tracer, **dataframe_params)
    dataframe.set_features()
    dataframe.set_targets()
    dataframe.build(cut_fracarea=cut_fracarea)
    regression = Regression(dataframe, regressor='RF', suffix_regressor=suffix_regressor, n_jobs=40, use_kfold=True, feature_names=feature_names, compute_permutation_importance=True, overwrite=True, seed=seed, save_regressor=False)
    regression.get_weight_map(save=True)
    regression.plot_maps_and_systematics(max_plot_cart=max_plot_cart, cut_fracarea=cut_fracarea)


def _bgs_any_weight(seed):
    """
        Compute weight with standard parametrization for BGS_ANY in DA02.
    """
    logger.info("Compute weight for BGS_ANY at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'BGS_ANY', '', 128
    suffix_regressor = ''
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 2000

    cut_fracarea = False

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, suffix_regressor, cut_fracarea, seed, param, max_plot_cart)


def _lrg_weight(seed):
    """
        Compute weight with standard parametrization for LRG in DA02.
    """
    logger.info("Compute weight for LRG at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'LRG', '', 128
    suffix_regressor = ''
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    cut_fracarea = False

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, suffix_regressor, cut_fracarea, seed, param, max_plot_cart)


def _elg_weight(seed):
    """
        Compute weight with standard parametrization for ELG in DA02.
    """
    logger.info(f"Compute weight for ELG at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'ELG', '', 512
    suffix_regressor = ''
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, cut_desi=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 3500

    cut_fracarea = False

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, suffix_regressor, cut_fracarea, seed, param, max_plot_cart)


def _qso_weight(seed, use_stream=True, use_stardens=True):
    """
        Compute weight with standard parametrization for QSO in DA02. If use_stream / use_stardens is False --> do not use STREAM / STARDENS as feature during the regression.
    """
    logger.info(f"Compute weight for QSO at Nside=256 with Sgr. Stream? {use_stream} with stardens? {use_stardens}")

    version, tracer, suffix_tracer, nside = 'DA02', 'QSO', '', 128
    suffix_regressor = ''
    dr9_footprint = DR9Footprint(nside, mask_lmc=True, clear_south=True, mask_around_des=True, cut_desi=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['regions'] = ['North', 'South', 'Des']
    max_plot_cart = 400

    cut_fracarea = False

    feature_names = ['EBV', 'STARDENS', 'STREAM',
                     'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                     'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    if not use_stardens:
        feature_names.remove('STARDENS')
        suffix_regressor += '_without_stardens'

    if not use_stream:
        feature_names.remove('STREAM')
        suffix_regressor += '_without_stream'

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, suffix_regressor, cut_fracarea, seed, param, max_plot_cart, feature_names)


if __name__ == '__main__':

    setup_logging(log_file='DA02.log')
    setup_mplstyle()

    mkdir('../res/DA02')

    _bgs_any_weight(210)
    _lrg_weight(220)
    _elg_weight(240)
    _qso_weight(250)
    _qso_weight(250, use_stream=False, use_stardens=False)

    print("\nMOVE the DA02.log file into the output directory ../res/DA02\n")
    shutil.move('DA02.log', '../res/DA02/DA02.log')
