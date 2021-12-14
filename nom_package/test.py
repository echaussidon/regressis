from build_dataframe import DataFrame
from regressor import Regressor

import matplotlib.pyplot as plt

# Set up logging
from utils import setup_logging
setup_logging()

import os
basedir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_dir = os.path.join(basedir, 'Data') #where the pixmap + sgr are
output_dir = os.path.join(basedir, 'Res')

version = 'SV3'
tracer = 'QSO'
suffixe_tracer = '' # Si on veut par exemple avoir QSO_newsel


param_targets = dict()
param_targets['Nside'] = 256

param_targets['use_median'] = False
param_targets['use_new_norm'] = False

param_targets['remove_LMC'] = True
param_targets['clear_south'] = True

# param_targets['region'] = ['North', 'South', 'Des'] #(default)
# region available = ['North', 'South', 'South_pole', 'Des_mid'], ['North', 'South_mid', 'South_pole'], ['North', 'South_all']

param_targets['region'] = ['Des']

dataframe = DataFrame(version, tracer, data_dir, output_dir, suffixe_tracer=suffixe_tracer, **param_targets)

dataframe.load_feature_from_healpix()
dataframe.load_target_from_healpix(data_dir=data_dir, load_fracarea=False)

dataframe.build_dataframe_photometry()

regressor = Regressor(dataframe, engine='RF', overwrite_regression=True, n_jobs=4)

regressor.make_regression(save_w_sys_map=True)


## il faut fixer la seeeed

## AJOUTER LES ERREURS DANS .fit(X, Y, Y_err) --< ultra ilportante
## mettre aussi linear dans le k-fold correctement p(avec scikit leanr et garder entrainment complet avec minuit)


## remarque pour els tiles on creera quand meme une colonne hxpxixel juste pour pouvoir faire le kfold et faire la speration pour la photometr --> prendre valuer au milieu de la petal --> ok (Le Nside n'aura plus d'importznce sauf pour al localisation)

## implementer d'autre truc comme les valeurs de shapley la permutation importance ? ect ... --> créer des fonctions dans Regressor
