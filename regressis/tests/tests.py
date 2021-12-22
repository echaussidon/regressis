#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np

from regressis import PhotometricDataFrame, Regressor, setup_logging

logger = logging.getLogger('Tests')


def test_case_qso():

    data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_case_qso') # where the pixmap + sgr + QSO target maps are

    #output_dir = os.path.join(basedir, 'res')
    output_dir = None #do not save figure

    print(" ")
    version, tracer, suffix_tracer = 'SV3', 'QSO', ''

    param_targets = dict()
    param_targets['nside'] = 256
    param_targets['use_median'] = False
    param_targets['use_new_norm'] = False
    param_targets['mask_lmc'] = True
    param_targets['clear_south'] = True
    param_targets['mask_around_des'] = True

    # param_targets['region'] = ['North', 'South', 'Des'] # value by default
    # region available = ['North', 'South', 'South_pole', 'Des_mid'], ['North', 'South_mid', 'South_pole'], ['North', 'South_all']
    param_targets['region'] = ['Des']

    dataframe = PhotometricDataFrame(version, tracer, data_dir, output_dir, suffix_tracer=suffix_tracer, **param_targets)
    dataframe.set_features()
    print(" ")
    dataframe.set_targets()
    print(" ")
    dataframe.build(selection_on_fracarea=True)
    print(" ")  #sedd 123
    regressor = Regressor(dataframe, engine='RF', compute_permutation_importance=False, overwrite_regression=False, n_jobs=6, seed=124, save_regressor=False, updated_nfold={'Des':2})
    print(" ")
    regressor.make_regression()
    print(" ")
    w_sys = regressor.build_w_sys_map(savemap=False, savedir=data_dir)
    print(" ")
    #regressor.plot_maps_and_systematics(max_plot_cart=400)

    logger.info('Load precompute systematic weights and compare the current computation')
    w_sys_test = np.load(os.path.join(data_dir, 'SV3_QSO_imaging_weight_256.npy'))
    mask = ~np.isnan(w_sys)
    assert np.allclose(w_sys[mask], w_sys_test[mask])
    logger.info('TEST')


if __name__ == '__main__':

    setup_logging()
    test_case_qso()
