from build_dataframe import PhotometricDataFrame
from regressor import Regressor

# Set up logging
from utils import setup_logging
setup_logging()

import os
basedir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
data_dir = os.path.join(basedir, 'Data') #where the pixmap + sgr are
output_dir = os.path.join(basedir, 'Res')

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

#param_targets['region'] = ['Des']

dataframe = PhotometricDataFrame(version, tracer, data_dir, output_dir, suffixe_tracer=suffixe_tracer, **param_targets)
dataframe.set_features()
print(" ")
dataframe.set_targets()
print(" ")
dataframe.build_for_regressor(selection_on_fracarea=True)

# regressor = Regressor(dataframe, engine='RF', overwrite_regression=True, n_jobs=6, save_regressor=False, updated_nfold={'Des':2})
# regressor.make_regression()
#
# regressor.save_w_sys_map()
# regressor.plot_maps_and_systematics(max_plot_cart=400)

## il faut fixer la seeeed

## AJOUTER LES ERREURS DANS .fit(X, Y, Y_err) --< ultra ilportante
## mettre aussi linear dans le k-fold correctement p(avec scikit leanr et garder entrainment complet avec minuit)


## remarque pour els tiles on creera quand meme une colonne hxpxixel juste pour pouvoir faire le kfold et faire la speration pour la photometr --> prendre valuer au milieu de la petal --> ok (Le Nside n'aura plus d'importznce sauf pour al localisation)

## implementer d'autre truc comme les valeurs de shapley la permutation importance ? ect ... --> créer des fonctions dans Regressor
