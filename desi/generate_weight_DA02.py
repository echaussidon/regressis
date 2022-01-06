#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import logging
import time

from regressis import PhotometricDataFrame, Regressor, DR9Footprint, setup_logging
from regressis.utils import mkdir, load_regressis_style


logger = logging.getLogger('DA02')


def _compute_weight(version, tracer, footprint, suffix_tracer, seed, param, max_plot_cart, feature_names=None):
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
    dataframe.build(cut_fracarea=True)
    regression = Regression(dataframe, regressor='RF', n_jobs=40, use_kfold=True, feature_names=feature_names, compute_permutation_importance=True, overwrite=True, seed=seed, save_regressor=False)
    regression.get_weight_map(save=True, savedir=dataframe_params['output_dir'])
    regression.plot_maps_and_systematics(max_plot_cart=max_plot_cart)


def _bgs_any_weight(seed):
    """
        Compute weight with standard parametrization for BGS_ANY in DA02.
    """
    start = time.time()
    logger.info("Compute weight for BGS_ANY at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'BGS_ANY', '', 128
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 2000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _lrg_weight(seed):
    """
        Compute weight with standard parametrization for LRG in DA02.
    """
    start = time.time()
    logger.info("Compute weight for LRG at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'LRG', '', 128
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _elg_weight(seed):
    """
        Compute weight with standard parametrization for ELG in DA02.
    """
    start = time.time()
    logger.info(f"Compute weight for ELG at Nside=128")

    version, tracer, suffix_tracer, nside = 'DA02', 'ELG', '', 512
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']

    max_plot_cart = 3500

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _qso_weight(seed, use_stream=True, use_stardens=True):
    """
        Compute weight with standard parametrization for QSO in DA02. If use_stream / use_stardens is False --> do not use STREAM / STARDENS as feature during the regression.
    """
    start = time.time()
    logger.info(f"Compute weight for QSO at Nside=256 with Sgr. Stream? {use_stream} with stardens? {use_stardens}")

    version, tracer, suffix_tracer, nside = 'DA02', 'QSO', '', 128
    dr9_footprint = DR9Footprint(nside, mask_lmc=True, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/DA02'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 400

    feature_names = ['EBV',
                     'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                     'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    if use_stream:
        feature_names.append('STREAM')
        suffix_tracer = '_with_stream'
    if use_stradens:
        feature_names.append('STARDENS')
    else:
        suffix_tracer += '_without_stardens'

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart, feature_names)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


if __name__ == '__main__':

    setup_logging(log_file='DA02.log')
    load_regressis_style()

    mkdir('../res/DA02')

    _bgs_any_weight(210)
    _lrg_weight(220)
    _elg_weight(240)
    _qso_weight(250)
    _qso_weight(250)
    _qso_weight(250)


    print("\nMOVE the DA02.log file into the output directory ../res/DA02\n")
    shutil.move('DA02.log', '../res/DA02/DA02.log')
