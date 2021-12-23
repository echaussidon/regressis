#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np

from regressis import PhotometricDataFrame, Regressor, DR9Footprint, setup_logging


logger = logging.getLogger('Tests')


def test_case_qso():
    print(" ")
    version, tracer, suffix_tracer = 'SV3', 'QSO', ''
    dr9 = DR9Footprint(256, mask_lmc=True, clear_south=True, mask_around_des=True, desi_cut=False)

    param = dict()
    param['data_dir'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_case_qso') # where the pixmap + sgr + QSO target maps are
    #output_dir = os.path.join(basedir, 'res')
    param['output_dir'] = None # do not save figure
    param['use_median'] = False
    param['use_new_norm'] = False
    # region available = ['North', 'South', 'South_pole', 'Des_mid'], ['North', 'South_mid', 'South_pole'], ['North', 'South_all'], ['North', 'South', 'Des'] # value by default
    param['region'] = ['Des']

    dataframe = PhotometricDataFrame(version, tracer, dr9, suffix_tracer, **param)
    dataframe.set_features()
    print(" ")
    dataframe.set_targets()
    print(" ")
    dataframe.build(selection_on_fracarea=True)
    print(" ")
    regressor = Regressor(dataframe, engine='RF', compute_permutation_importance=False, overwrite_regression=False, n_jobs=6, seed=123, save_regressor=False, updated_nfold={'Des':2})
    print(" ")
    regressor.make_regression()
    print(" ")
    w_sys = regressor.build_w_sys_map(savemap=False, savedir=param['data_dir'])
    print(" ")
    #regressor.plot_maps_and_systematics(max_plot_cart=400)

    logger.info('Load precompute systematic weights and compare the current computation')
    w_sys_test = np.load(os.path.join(param['data_dir'], 'SV3_QSO_imaging_weight_256.npy'))
    mask = ~np.isnan(w_sys)
    assert np.allclose(w_sys[mask], w_sys_test[mask]), "The computation of systematic weights in test case gives bad result, please do not change any parameter in tests.py"
    logger.info('Test is complete without any error :) !')


if __name__ == '__main__':

    setup_logging()
    test_case_qso()
