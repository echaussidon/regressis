#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np

from regressis import PhotometricDataFrame, Regression, DR9Footprint, setup_logging


logger = logging.getLogger('Tests')


def test_case_qso():
    version, tracer, suffix_tracer = 'SV3', 'QSO', ''
    dr9_footprint = DR9Footprint(256, mask_lmc=True, clear_south=True, mask_around_des=True, cut_desi=False)

    params = dict()
    params['data_dir'] = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'test_case_qso') # where the pixmap + sgr + QSO target maps are
    #params['output_dir'] = '../../Res'
    params['output_dir'] = None # do not save figure
    params['use_median'] = False
    params['use_new_norm'] = False
    # region available = ['North', 'South', 'South_pole', 'Des_mid'], ['North', 'South_mid', 'South_pole'], ['North', 'South_all'], ['North', 'South', 'Des'] # value by default
    params['regions'] = ['Des']

    dataframe = PhotometricDataFrame(version, tracer, dr9_footprint, suffix_tracer, **params)
    dataframe.set_features()
    dataframe.set_targets()
    dataframe.build(cut_fracarea=True)
    regression = Regression(dataframe, regressor='RF', compute_permutation_importance=False, overwrite=False, n_jobs=6, seed=123, save_regressor=False, nfold_params={'Des':2})
    wsys = regression.get_weight_map(save=False, savedir=params['data_dir'])
    if params.get('output_dir', None) is not None:
        regression.plot_maps_and_systematics(max_plot_cart=400)

    logger.info('Load precomputed systematic weights and compare to the current computation')
    wsys_ref = np.load(os.path.join(params['data_dir'], 'SV3_QSO_imaging_weight_256.npy'))
    mask = ~np.isnan(wsys)
    assert np.allclose(wsys[mask], wsys_ref[mask]), "The computation of systematic weights in test case yields incorrt result, have you changed any parameter in tests.py?"
    logger.info('Test completed without any error')


if __name__ == '__main__':

    setup_logging()
    test_case_qso()
