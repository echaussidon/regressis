import sys
if sys.path[0][:7] == '/global':
    local = False #we are on Nersc
elif sys.path[0][:4] == '/usr':
    local = True #we are in my mac

import numpy as np
import healpy as hp

import pandas as pd
from astropy.table import Table

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
if local:
    plt.style.use('~/Documents/CEA/Software/desi_ec/ec_style.mplstyle')
else:
    plt.style.use('~/Software/desi_ec/ec_style.mplstyle')

import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import plot_partial_dependence
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split, GroupKFold, cross_val_predict
from sklearn.neural_network import MLPRegressor

import pickle

from linear_regression import regression_least_square

from plot import to_tex

#------------------------------------------------------------------------------#

if local:
    n_jobs = 6 #pour RF
else:
    n_jobs = 30 #pour RF


#il suffit de commenter une feature qu'on ne veut pas dans les 4 tableaux pour qu'elle ne soit pas utiliée :)

def _load_feature_names(suffixe, use_stream=True, use_stars=True):
    if suffixe in ['_QSO', '_QSO_goodz', '_QSO_clust']:
        return _load_feature_names_for_qso(use_stars)
    elif suffixe in ['_ELG', '_ELG_HIP', '_ELG_LOP', '_ELG_VLO', '_ELG_goodz', '_ELGnoQSO_goodz', '_ELGnoQSO', '_ELG_clust']:
        return _load_feature_names_for_elg(use_stream)
    elif suffixe in ['_LRG_LOWDENS', '_LRG', '_LRG_goodz', '_LRG_goodz']:
        return _load_feature_names_for_lrg()
    elif suffixe in ['_BGS_ANY', '_BGS_FAINT', '_BGS_BRIGHT']:
        return _load_feature_names_for_bgs()
    else:
        print("[WARNING] We use the same set of feature than for QSO !!")
        return _load_feature_names_for_qso(use_stars)


def _load_feature_names_for_qso(use_stars=True):
    if use_stars:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                                'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                                'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_for_normalization = ['STARDENS', 'EBV',
                                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_pandas = ['STARDENS', 'EBV', 'STREAM',
                                'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z', 'PSFDEPTH W1', 'PSFDEPTH W2',
                                'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

        feature_names_to_plot = ['Stardens', '$E(B-V)$', 'Sgr.\nStream',
                                 '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$', '$PSF_{Depth}$ $W1$', '$PSF_{Depth}$ $W2$',
                                 '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']
    else:
        feature_names = ['EBV',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_for_normalization = ['EBV',
                                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_pandas = ['EBV',
                                'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z', 'PSFDEPTH W1', 'PSFDEPTH W2',
                                'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

        feature_names_to_plot = ['$E(B-V)$',
                                 '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$', '$PSF_{Depth}$ $W1$', '$PSF_{Depth}$ $W2$',
                                 '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']
    return feature_names, feature_names_for_normalization, feature_names_pandas, feature_names_to_plot

def _load_feature_names_for_elg(use_stream=True):
    if use_stream:
        feature_names = ['STARDENS', 'EBV', 'STREAM',
                                'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                                'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_for_normalization = ['STARDENS', 'EBV',
                                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_pandas = ['STARDENS', 'EBV', 'STREAM',
                                'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z',
                                'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

        feature_names_to_plot = ['Stardens', '$E(B-V)$', 'Sgr.\nStream',
                                 '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$',
                                 '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']
    else:
        feature_names = ['STARDENS', 'EBV',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_for_normalization = ['STARDENS', 'EBV',
                                           'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                                           'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

        feature_names_pandas = ['STARDENS', 'EBV',
                                'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z',
                                'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

        feature_names_to_plot = ['Stardens', '$E(B-V)$',
                                 '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$',
                                 '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']

    return feature_names, feature_names_for_normalization, feature_names_pandas, feature_names_to_plot

def _load_feature_names_for_lrg():
    feature_names = ['STARDENS', 'EBV',
                     'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1',
                     'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    feature_names_for_normalization = ['STARDENS', 'EBV',
                                       'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1',
                                       'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    feature_names_pandas = ['STARDENS', 'EBV',
                            'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z', 'PSFDEPTH W1',
                            'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

    feature _names_to_plot = ['Stardens', '$E(B-V)$',
                             '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$', '$PSF_{Depth}$ $W1$',
                             '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']

    return feature_names, feature_names_for_normalization, feature_names_pandas, feature_names_to_plot

def _load_feature_names_for_bgs():
    feature_names = ['STARDENS', 'EBV',
                            'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                            'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    feature_names_for_normalization = ['STARDENS', 'EBV',
                                       'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                                       'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    feature_names_pandas = ['STARDENS', 'EBV',
                            'PSFDEPTH G', 'PSFDEPTH R', 'PSFDEPTH Z',
                            'PSFSIZE G', 'PSFSIZE R', 'PSFSIZE Z']

    feature_names_to_plot = ['Stardens', '$E(B-V)$',
                             '$PSF_{Depth}$ $g$', '$PSF_{Depth}$ $r$', '$PSF_{Depth}$ $z$',
                             '$PSF_{Size}$ $g$', '$PSF_{Size}$ $r$', '$PSF_{Size}$ $r$']

    return feature_names, feature_names_for_normalization, feature_names_pandas, feature_names_to_plot

#------------------------------------------------------------------------------#

def _load_targets(suffixe, release, version, Nside):
    targets = np.load(f'Data/{version}{suffixe}_targets_{Nside}.npy', allow_pickle=True)
    if suffixe == '_QSO' or suffixe == '_QSO_goodz' or suffixe == '_QSO_clust':
        ax_lim = 0.2
        max_plot_cart = 400
    elif suffixe == '_ELG_HIP':
        max_plot_cart = 2500
        ax_lim = 0.1
    elif suffixe == '_ELG_LOP':
        max_plot_cart = 1500
        ax_lim = 0.1
    elif suffixe == '_ELG_VLO':
        max_plot_cart = 550
        ax_lim = 0.1
    elif suffixe == '_ELG' or suffixe == '_ELG_goodz' or suffixe == '_ELGnoQSO_goodz' or suffixe == '_ELGnoQSO' or suffixe == '_ELG_clust':
        max_plot_cart = 3000
        ax_lim = 0.1
    elif suffixe == '_LRG' or suffixe == '_LRG_goodz' or suffixe == '_LRG_clust':
        max_plot_cart = 900
        ax_lim = 0.1
    elif suffixe == '_LRG_LOWDENS':
        max_plot_cart = 700
        ax_lim = 0.1
    elif suffixe == '_BGS_ANY':
        max_plot_cart = 1800
        ax_lim = 0.1
    elif suffixe == '_BGS_FAINT':
        max_plot_cart = 1000
        ax_lim = 0.1
    elif suffixe == '_BGS_BRIGHT':
        max_plot_cart = 800
        ax_lim = 0.1
    elif suffixe == '_ELG_VLO_newsel':
        max_plot_cart = 400
        ax_lim = 0.1
    else:
        print("[ERROR] Please use a correct suffixe for build_features.py")
        sys.exit()
    return targets, max_plot_cart, ax_lim

#------------------------------------------------------------------------------#
def plot_kfold(Nside, kfold, group, pixels, title='', save=True, savename='Res/kfold.pdf'):
    map = np.zeros(hp.nside2npix(Nside))
    index = []
    i = 1
    for index_train, index_test in kfold.split(pixels, groups=group):
        index += [index_test]
        map[pixels[index_test]] = i
        i = i+1
    map[map == 0] = np.NaN
    #attention au sens de l'axe en RA ! --> la on le prend normal et on le retourne à la fin :)
    plt.figure(1)
    map_to_plot = hp.cartview(map, nest=True, flip='geo', rot=120, fig=1, return_projected_map=True)
    plt.close()

    fig, ax = plt.subplots(figsize=(11,7))
    map_plotted = plt.imshow(map_to_plot, interpolation='nearest', cmap='jet', origin='lower', extent=[-60, 300, -90, 90])
    ax.set_xlim(-60, 300)
    ax.xaxis.set_ticks(np.arange(-60, 330, 30))
    plt.gca().invert_xaxis()
    ax.set_xlabel('R.A. [deg]')
    ax.set_ylim(-90, 90)
    ax.yaxis.set_ticks(np.arange(-90, 120, 30))
    ax.set_ylabel('Dec. [deg]')
    ax.grid(True, alpha=0.8, linestyle=':')
    plt.title(title)

    if save:
        plt.savefig(savename)
    plt.close()

    return index

def accuracy_plot_for_regressor(Y, Y_pred, pixels, keep_to_train, dir_to_save):
    print("Trace la precision de notre regression, on l'applique sur le jeu d'entrainement ...\n")
    fig, ax = plt.subplots(1, 3, figsize=(16,8))
    plt.subplots_adjust(left=0.07, right=0.96, bottom=0.1, top=0.9, wspace=0.3)
    ax[0].scatter(pixels[keep_to_train == 1], Y[keep_to_train == 1], color = 'red', label='Truth')
    ax[0].scatter(pixels[keep_to_train == 1], Y_pred[keep_to_train == 1], color = 'blue', label='Estimation')
    ax[0].legend()
    ax[0].set_xlabel('Pixel Number')
    ax[0].set_ylabel('Normalized Targets Density')

    ax[1].scatter(pixels[keep_to_train == 1], (Y_pred[keep_to_train == 1] - Y[keep_to_train == 1])/Y[keep_to_train == 1])
    mean = np.mean((Y_pred[keep_to_train == 1] - Y[keep_to_train == 1])/Y[keep_to_train == 1])
    std = np.std((Y_pred[keep_to_train == 1] - Y[keep_to_train == 1])/Y[keep_to_train == 1])
    ax[1].axhline(mean, color='black', label="mean : {:.4f}".format(mean))
    ax[1].axhline(mean + std, color='black', linestyle='--', label="std : {:.4f}".format(std))
    ax[1].axhline(mean - std, color='black', linestyle='--')
    ax[1].legend()
    ax[1].set_xlabel('Pixel Number')
    ax[1].set_ylabel('Relative Errors')

    ax[2].hist((Y_pred[keep_to_train == 1] - Y[keep_to_train == 1])/Y[keep_to_train == 1], bins=100, range=(-1,1))
    ax[2].set_xlabel('Relative Errors')
    ax[2].set_ylabel('# Pixels')

    plt.savefig(dir_to_save + "kfold_accuracy.png")

def plot_feature_importances(path, feature_names_pandas, feature_names_pandas_to_plot, zone, nbr_fold): #pas très classe mais ca marche donc ok

    importance_evol = dict()
    importance_mean = dict()
    importance_mean_err = dict()

    rank = [i for i in range(len(feature_names_pandas))]
    for area in zone:
        importance_evol[area] = np.zeros((len(feature_names_pandas), nbr_fold[area]))
        for i in range(nbr_fold[area]):
            d = pd.read_pickle(f"{path}{area}/feature_importance_fold_{i}.pkl")
            d.reset_index(inplace=True)
            d.insert(2, 'rank', rank)
            d = d.sort_values('index', ascending=True)
            importance_evol[area][:, i] = d['feature importance'].values
            name_to_plot_order = d['index'].values

        importance_mean[area] = np.mean(importance_evol[area], axis=1)
        importance_mean_err[area] = np.std(importance_evol[area], axis=1) / np.sqrt(nbr_fold[area] - 1)

    index_order = []
    for name in feature_names_pandas:
        i, = np.where(name_to_plot_order == name)
        index_order += [i[0]]

    Y = np.linspace(0, 5, len(feature_names_pandas))

    plt.figure()

    offset = -0.11
    for area in zone:
        plt.barh(Y + offset, importance_mean[area][index_order], xerr=importance_mean_err[area][index_order], height=0.1, label=to_tex(area), alpha=0.8, align='center', ecolor='black', capsize=5)
        offset += 0.11
    plt.legend()
    plt.yticks(Y, feature_names_pandas_to_plot, size=12)
    plt.xlabel('Feature Importances')
    plt.savefig(path + 'Feature_importances.pdf')
    plt.close()

def make_regressor_kfold(X, Y, keep_to_train, pixels, Nside, feature_names_for_normalization, feature_names_pandas, use_neural_network=False, nbr_fold=5, min_samples_leaf=20, plot_accuracy=True, dir_to_save='Res'):
    if use_neural_network:
        if dir_to_save[-6:-1] == 'North':
            dict_ini_mlp = {'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8), 'max_iter': 6000, \
                            'n_iter_no_change': 100, 'random_state': 5, 'solver': 'adam', 'tol': 1e-5}
        elif dir_to_save[-6:-1] == 'South':
            dict_ini_mlp = {'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8), 'max_iter': 6000, \
                            'n_iter_no_change': 100, 'random_state': 5, 'solver': 'adam', 'tol': 1e-5}
        else:
            dict_ini_mlp = {'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8), 'max_iter': 6000, \
                            'n_iter_no_change': 100, 'random_state': 5, 'solver': 'adam', 'tol': 1e-5}
        print(f"            ** USE NEURAL NETWORK")
        print(f"                        *** Dict ini : {dict_ini_mlp}")
        regressor = MLPRegressor(**dict_ini_mlp)
    else:
        print(f"            ** USE RANDOM FOREST")
        dict_ini_rf = {'n_estimators':200, 'min_samples_leaf':min_samples_leaf, 'max_depth':None, 'max_leaf_nodes':None, 'n_jobs':n_jobs}
        print(f"                        *** Dict ini : {dict_ini_rf}")
        regressor = RandomForestRegressor(**dict_ini_rf)

    kfold = GroupKFold(n_splits=nbr_fold)
    size_group = 1000  * (Nside / 256)**2
    group = [i//size_group for i in range(pixels.size)]

    print("\nTrace les differents k-fold qui vont etre utilises ...")
    print("    * On utilise : {} avec un group_size = {}".format(kfold, size_group))
    index = plot_kfold(Nside, kfold, group, pixels, title='{}-Fold repartition'.format(nbr_fold), save=True, savename=dir_to_save+'kfold_repartition.pdf')

    Y_pred = np.zeros(pixels.size)
    X.reset_index(drop=True, inplace=True)

    print("\nPrediction pour le Fold :")
    start = time.time()
    for i in range(nbr_fold):
        print("     * {}".format(i))
        fold_index = index[i]
        keep_to_train_fold = np.delete(keep_to_train, fold_index)
        print(f"[INFO] There are {np.sum(keep_to_train_fold == 1)} pixels to train fold {i} which contains {np.sum(keep_to_train == 1) - np.sum(keep_to_train_fold == 1)} pixels (kept for the global training)")

        if use_neural_network:
            print("                         *** On normalise et recentre le jeu d'entrainement ...")
            print("[WARNING :] We normalize and center features on the training footprint fot his fold training !")
            print("[WARNING :] Treat only features which are not the Sgr. stream (already normalized) !")
            X_fold = X.copy()
            X_fold[feature_names_for_normalization] = (X[feature_names_for_normalization] - X[feature_names_for_normalization].drop(fold_index)[keep_to_train_fold == 1].mean())/X[feature_names_for_normalization].drop(fold_index)[keep_to_train_fold == 1].std()
            print(f"[TEST :] Mean of Mean and Std on all features : {X_fold[feature_names_for_normalization].mean().mean()} -- {X_fold[feature_names_for_normalization].std().mean()}")
            print(f"[TEST :] Mean of Mean and Std on the fold-training features : {X_fold[feature_names_for_normalization].drop(fold_index)[keep_to_train_fold == 1].mean().mean()} -- {X_fold[feature_names_for_normalization].drop(fold_index)[keep_to_train_fold == 1].std().mean()}\n")
        else:
            print("                         *** On NE normalise et recentre PAS le jeu d'entrainement ...")
            X_fold = X.copy()

        X_train, Y_train = X_fold.drop(fold_index)[keep_to_train_fold == 1], np.delete(Y, fold_index)[keep_to_train_fold == 1]
        regressor.fit(X_train, Y_train)

        if not use_neural_network:
            print("                         *** On sauvegarde importance feature ...")
            df_feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names_pandas, columns=['feature importance']).sort_values('feature importance', ascending=False).to_pickle(dir_to_save+f"feature_importance_fold_{i}.pkl")
            df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names_pandas)
            df_feature_long = pd.melt(df_feature_all, var_name='feature name', value_name='values').to_pickle(dir_to_save+f"feature_importance_all_trees_fold_{i}.pkl")

        Y_pred_fold = np.zeros(fold_index.size)
        Y_pred_fold = regressor.predict(X_fold.iloc[fold_index])
        Y_pred[fold_index] = Y_pred_fold

    print("    * Fait en : {:.3f} s".format(time.time() - start))

    if plot_accuracy:
        accuracy_plot_for_regressor(Y, Y_pred, pixels, keep_to_train, dir_to_save)

    return Y_pred, index

def make_polynomial_regressor(X, Y, keep_to_train, regulator, feature_names_for_normalization):

    def model(x, *par): # Estimateur == modele utilise
        return par[0]*np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

    nbr_features = X.shape[1]
    print(f"[TEST] Number of features used : {nbr_features}")
    nbr_params = nbr_features + 1

    print(f"            ** Taille de l'échantillon (non nan value): {np.sum(Y>0)}")
    print(f"            ** Information sur normalized targets : Mean = {np.nanmean(Y)} and Std = {np.nanstd(Y)}")
    print("[WARNING] We normalize and center features on the training footprint (don't forget to normalized also the non training footprint)")
    X.loc[:, feature_names_for_normalization] = (X[feature_names_for_normalization] - X[feature_names_for_normalization][keep_to_train == 1].mean())/X[feature_names_for_normalization][keep_to_train == 1].std()
    X_train, Y_train = X[keep_to_train == 1], Y[keep_to_train == 1]
    print(f"[TEST] Mean of Mean and Std training features (should be 0, 1): {X_train.mean().mean()} -- {X_train.std().mean()}\n")

    dict_ini = {f'a{i}': 0 if i==0 else 0 for i in range(0, nbr_params)}
    dict_ini.update({f'error_a{i}': 0.001 for i in range(0, nbr_params)})
    dict_ini.update({f'limit_a{i}': (-1, 2) if i==0 else (-3,3) for i in range(0, nbr_params)})
    dict_ini.update({'errordef': 1}) #for leastsquare
    Y_cov_inv = np.diag(1/np.sqrt(Y_train))

    param = regression_least_square(model, regulator, X_train, Y_train, Y_cov_inv, nbr_params, **dict_ini)

    print(f"[TEST] Mean of systematics_correction : {model(X_train, *param).mean()} \n")

    return model(X, *param)
