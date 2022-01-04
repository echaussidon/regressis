#!/usr/bin/env python
# coding: utf-8

import os
import shutil
import logging
import time

from regressis import PhotometricDataFrame, Regressor, DR9Footprint, setup_logging
from regressis.utils import mkdir


logger = logging.getLogger('SV3')


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
    dataframe = PhotometricDataFrame(version, tracer, dr9_footprint, suffix_tracer, **param)
    dataframe.set_features()
    print(" ")
    dataframe.set_targets()
    print(" ")
    dataframe.build(selection_on_fracarea=True)
    print(" ")
    regressor = Regressor(dataframe, engine='LINEAR', use_kfold=False, feature_names=feature_names, compute_permutation_importance=True, overwrite_regression=True, n_jobs=6, seed=seed, save_regressor=False)
    print(" ")
    regressor.make_regression()
    print(" ")
    w_sys = regressor.build_w_sys_map(savemap=True, savedir=param['output_dir'])
    print(" ")
    regressor.plot_maps_and_systematics(max_plot_cart=max_plot_cart)
    print(" ")


def _lrg_weight(seed):
    """
        Compute weight with standard parametrization for LRG in SV3.
    """
    start = time.time()
    logger.info("Compute weight for LRG at Nside=256")

    version, tracer, suffix_tracer, nside = 'SV3', 'LRG', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/SV3'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 1000

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _elg_weight(seed, add_stream=False):
    """
        Compute weight with standard parametrization for ELG in SV3. If add_stream=True then add STREAM during the regression.
    """
    start = time.time()
    logger.info(f"Compute weight for ELG at Nside=512 with Sgr. Stream? {add_stream}")

    version, tracer, suffix_tracer, nside = 'SV3', 'ELG', '', 512
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/SV3'
    param['use_median'] = False
    param['use_new_norm'] = False
    param['region'] = ['North', 'South', 'Des']
    if add_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        suffix_tracer = '_with_stream'
        param['use_new_norm'] = True
    else:
        feature_names = None
    max_plot_cart = 3500

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart, feature_names)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _elg_hip_weight(seed, add_stream=False):
    """
        Compute weight with standard parametrization for ELG HIP in SV3. If add_stream=True then add STREAM during the regression.
    """
    start = time.time()
    logger.info("Compute weight for ELG at Nside=512 with Sgr. Stream map")

    version, tracer, suffix_tracer, nside = 'SV3', 'ELG_HIP', '', 512
    dr9_footprint = DR9Footprint(nside, mask_lmc=False, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/SV3'
    param['use_median'] = False
    param['use_new_norm'] = True
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 2500

    if add_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
        suffix_tracer = '_with_stream'
        param['use_new_norm'] = True
    else:
        feature_names = None

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart, feature_names)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


def _qso_weight(seed):
    """
        Compute weight with standard parametrization for QSO in SV3.
    """
    start = time.time()
    logger.info("Compute weight for QSO at Nside=256 with Sgr. Stream map")

    version, tracer, suffix_tracer, nside = 'SV3', 'QSO', '', 256
    dr9_footprint = DR9Footprint(nside, mask_lmc=True, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = '../data'
    param['output_dir'] = '../res/SV3'
    param['use_median'] = False
    param['use_new_norm'] = True
    param['region'] = ['North', 'South', 'Des']
    max_plot_cart = 400

    _compute_weight(version, tracer, dr9_footprint, suffix_tracer, seed, param, max_plot_cart)

    logger.info(f"Done in {time.time() - start:2.2f}\n")


if __name__ == '__main__':

    setup_logging(log_file='SV3.log')

    mkdir('../res/SV3')

    _lrg_weight(40)
    _elg_weight(50)
    #_elg_weight(55, add_stream=True)
    _elg_hip_weight(60)
    #_elg_hip_weight(65, add_stream=True)
    _qso_weight(70)

    print("\nMOVE the SV3.log file into the output directory ../res/SV3\n")
    shutil.move('SV3.log', '../res/SV3/SV3.log')
