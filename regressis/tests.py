#!/usr/bin/env python
# coding: utf-8

import logging
logger = logging.getLogger("TESTS")

from dataframe import PhotometricDataFrame
from regressor import Regressor

# Set up logging
from utils import setup_logging
setup_logging()

import os
data_dir = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'tests/test_case_qso') #where the pixmap + sgr + QSO targets map are

#output_dir = os.path.join(basedir, 'res')
output_dir = None #do not save figure

print(" ")
version = 'SV3'
tracer = 'QSO'
suffixe_tracer = '' # Si on veut par exemple avoir QSO_newsel

param_targets = dict()
param_targets['Nside'] = 256
param_targets['use_median'] = False
param_targets['use_new_norm'] = False
param_targets['remove_LMC'] = True
param_targets['clear_south'] = True
param_targets['mask_around_des'] = True
# param_targets['region'] = ['North', 'South', 'Des'] #(default)
# region available = ['North', 'South', 'South_pole', 'Des_mid'], ['North', 'South_mid', 'South_pole'], ['North', 'South_all']

param_targets['region'] = ['Des']

dataframe = PhotometricDataFrame(version, tracer, data_dir, output_dir, suffixe_tracer=suffixe_tracer, **param_targets)
dataframe.set_features()
print(" ")
dataframe.set_targets()
print(" ")
dataframe.build_for_regressor(selection_on_fracarea=True)
print(" ")
regressor = Regressor(dataframe, engine='RF', compute_permutation_importance=True, overwrite_regression=True, n_jobs=6, seed=123, save_regressor=False, updated_nfold={'Des':2})
print(" ")
regressor.make_regression()
print(" ")
w_sys = regressor.build_w_sys_map(savemap=False, savedir=data_dir)
print(" ")
#regressor.plot_maps_and_systematics(max_plot_cart=400)

import numpy as np
logger.info('LOAD PRECOMPUTE WSYSMAP')
w_sys_test = np.load(os.path.join(data_dir, 'SV3_QSO_imaging_weight_256.npy'))

test = (np.abs(w_sys[~np.isnan(w_sys)] - w_sys_test[~np.isnan(w_sys)]) < 1e-15).sum()/w_sys[~np.isnan(w_sys)].size
logger.info(f'fraction of same value is {test:2.2%}')





## implementer d'autre truc comme les valeurs de shapley la permutation importance ? ect ... --> créer des fonctions dans Regressor
