#!/usr/bin/env python
# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import os
import sys
import time
import logging

import numpy as np
import healpy as hp
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, load

from . import utils
from .utils import deep_update, regression_least_square, zone_name_to_column_name


logger = logging.getLogger("Regressor")


def _load_feature_names(tracer, use_stream=None, use_stars=None):
    """
    Load the default feature set for regression.

    Parameters
    ----------
    tracer : str
        The tracer name e.g. QSO / ELG / LRG / BGS

    use_stream : bool
        Use Sgr. Stream as template; default = False for all and True for QSO.

    use_stars : bool
        Use stardens as template; default = True.

    Returns
    -------
    feature_names: list
        Names of features to regress against.
    """

    feature_names = ['STARDENS', 'EBV', 'STREAM',
                     'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                     'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

    if tracer == 'QSO':
        to_remove = []
        if not (use_stream is None) and not use_stream:
            to_remove.append(['STREAM'])
    elif tracer == 'ELG':
        to_remove = ['PSFDEPTH_W1', 'PSFDEPTH_W2', 'STREAM']
        if not (use_stream is None) and use_stream:
            to_remove.remove(['STREAM'])
    elif tracer == ['LRG']:
        to_remove = ['PSFDEPTH_W2', 'STREAM']
        if not (use_stream is None) and use_stream:
            to_remove.remove(['STREAM'])
    elif tracer == ['BGS']:
        to_remove = ['PSFDEPTH_W1', 'PSFDEPTH_W2', 'STREAM']
        if not (use_stream is None) and use_stream:
            to_remove.remove(['STREAM'])
    else:
        logger.info("We use the same set of feature than for QSO !!")
        to_remove = []
        if not (use_stream is None) and not use_stream:
            to_remove.append(['STREAM'])

    if not (use_stars is None) and not use_stars:
        to_remove.append(['STARDENS'])

    for elmt in to_remove:
        feature_names.remove(elmt)
    logger.info(f"We use the set: {feature_names}")
    return feature_names


def _load_rf_hyperparameters(updated_param=None, n_jobs=6, seed=123):
    """
    Load pre-defined hyperparameters for RF regressor for each specific region available. Can be updated with updated_param.

    Parameters
    ----------
    updated_param: dict
        updated param (e.g) {'North':{n_estimators:20}}

    n_jobs: int
        To parallelize the computation

    seed: int
        Fix the random_state of the Regressor for reproductibility

    Returns
    -------
    param: dict
        dictionary containing for each region a dictionary of the hyper-parameters for the regressor
    """
    param = dict()
    min_samples_leaf = {'North':20, 'South':20, 'Des':20, 'South_all':20, 'South_mid':20, 'South_pole':20, 'South_mid_no_des':20, 'Des_mid':20}
    for key in min_samples_leaf:
        param[key] = {'n_estimators':200, 'min_samples_leaf':min_samples_leaf[key], 'max_depth':None, 'max_leaf_nodes':None, 'n_jobs':n_jobs, 'random_state':seed}
    if updated_param is not None:
        deep_update(param, updated_param)
    return param


def _load_mlp_hyperparameters(updated_param=None, seed=123):
    """
    Load pre-defined hyperparameters for NN regressor for each specific region available. Can be updated with updated_param.

    Parameters
    ----------
    updated_param: dict
        updated param (e.g) {'North':{max_iter:20}}
    seed: int
        Fix the random_state of the Regressor for reproductibility

    Returns
    -------
    param: dict
        dictionary containing for each region a dictionary of the hyper-parameters for the regressor
    """
    param = dict()
    for key in ['North', 'South', 'Des', 'South_all', 'South_mid', 'South_pole', 'South_mid_no_des', 'Des_mid']:
        param[key] = {'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8),
                      'max_iter': 6000, 'n_iter_no_change': 100, 'random_state': 5, 'solver': 'adam', 'tol': 1e-5, 'random_state':seed}
    if updated_param is not None:
        deep_update(param, updated_param)
    return param


def _load_linear_hyperparameters(updated_param=None):
    """
    Load pre-defined hyperparameters for Linear regressor for each specific region available. Can be updated with updated_param.

    Parameters
    ----------
    updated_param: dict
        updated param (e.g) {'North':{n_jobs:2}}

    Returns
    -------
    param: dict
        dictionary containing for each region a dictionary of the hyper-parameters for the regressor
    """
    param = dict()
    for key in ['North', 'South', 'Des', 'South_all', 'South_mid', 'South_pole', 'South_mid_no_des', 'Des_mid']:
        param[key] = {}
    if updated_param is not None:
        deep_update(param, updated_param)
    return param


def _load_nfold(updated_param=None):
    """
    Load pre-defined number of folds for each specific region available. Can be updated with updated_param.

    Parameters
    ----------
    updated_param: dict
        updated param (e.g) {'North':2}

    Returns
    -------
    param: dict
        dictionary containing for each region the number of folds for the Kfold training
    """
    param = {'North':6, 'South':12, 'Des':6, 'South_all':18, 'South_mid':14, 'South_pole':5, 'Des_mid':3}
    if updated_param is not None:
        deep_update(param, updated_param)
    return param


class Regressor(object):
    """
    Implementation of the Systematic Correction based on template fitting regression
    """

    def __init__(self, dataframe, engine, feature_names=None, use_kfold=True,
                 updated_param_rf=None, updated_param_mlp=None, updated_param_linear=None, updated_nfold=None,
                 compute_permutation_importance=True, overwrite_regression=False, save_regressor=False, n_jobs=6, seed=123):
        """
        Initialize :class:`Regressor`.

        Parameters
        ----------
        dataframe: DataFrame class
            dataframe containing all the information to run the regressor. the method build_for_regressor need to be run before calling a Regressor class.
        engine: str
            either RF (Random Forest), NN (Multi layer perceptron), LINEAR  --> all based on scikit-learn implementation
        feature_names: str array
            List of feature name used during the regression. By default load feature name from _load_feature_names
        use_kfold: bool
            If True use Kfold computation. If False do not use Machine learning algortihm and compute Linear regression with iminuit on all the dataset.
        updated_param_rf / updated_param_mlp / updated_param_linear : dict
            Containing specific hyperparameters to update the default values loaded in _load_rf_hyperparameters / _load_mlp_hyperparameters / _load_linear_hyperparameters
        compute_permutation_importance: bool
            Compute and plot the permutation importance for inspection. It has to be compared with giny importance from Random Forest regressor.
        overwrite_regression: bool
            If True overwrite file in the output directory (if it is set in dataframe otherwise nothing happens). If false an error is raised and the ouput directory need to be deleted.
        save_regressor: bool
            If True regressor for each region and each fold is saved. WARNING: it is space consuming. Can be usefull to make some more advanced plots.
        n_jobs: int
            To parallelize the Random Forest computation.
        seed: int
            Fix the random state of RandomForestRegressor and MLPRegressor for reproductibility

        # TODO: why different updated_param_rf=None, updated_param_mlp=None, updated_param_linear=None when you pass only one engine anyway???
        # TODO: e.g. you can just provide kwargs_engine=None
        """

        self.dataframe = dataframe
        self.engine = engine.upper()
        if feature_names is None:
            self.feature_names = _load_feature_names(dataframe.tracer)
        else:
            self.feature_names = feature_names

        # set up the parameter for the considered regressor
        self.use_kfold = use_kfold
        if use_kfold:
            if self.engine == 'RF':
                self.param_regressor = _load_rf_hyperparameters(updated_param_rf, n_jobs, seed)
            elif self.engine == 'NN':
                self.param_regressor = _load_mlp_hyperparameters(updated_param_mlp, seed)
            elif self.engine == 'LINEAR':
                self.param_regressor = _load_linear_hyperparameters(updated_param_linear)
        else:
            # TODO: why??? RF and NN can also be used without kfold??
            # TODO: why dependence in iminuit when there is LinearRegression in sklearn?
            logging.warning("DO NOT USE K-FOLD --> ONLY REGRESSION AVAILABLE IS LINEAR WITH IMINUIT")
            self.engine = 'Linear_without_kfold'
            self.param_regressor = {'regulator':2e6*(self.dataframe.nside/256)**3} # TODO: why regulator???

        # in NN and Linear case, we normalize and standardize the data
        # STREAM is already normalize --> remove it from the list
        if self.engine != 'RF':
            if 'STREAM' in self.feature_names:
                self.feature_names_to_normalize = self.feature_names.copy()
                self.feature_names_to_normalize.remove('STREAM')
            else:
                self.feature_names_to_normalize = self.feature_names.copy()
        else:
            self.feature_names_to_normalize = None

        self.compute_permutation_importance = compute_permutation_importance
        self.save_regressor = save_regressor
        self.nfold = _load_nfold(updated_nfold)

        ## If not self.dataframe.output is None --> save figure and info !!
        if self.dataframe.output_dir is not None:
            # create the corresponding output folder --> put here since self.engine can be update with use_kfold = False
            if os.path.isdir(os.path.join(self.dataframe.output_dir, self.engine)):
                if not overwrite_regression:
                    raise ValueError(f"{os.path.join(self.dataframe.output_dir, self.engine)} already exist and overwrite_regression is set as {overwrite_regression}")
                else:
                    logger.warning(f"OVERWRITE {os.path.join(self.dataframe.output_dir, self.engine)}")
                    logger.warning(f"PLEASE REMOVE THE OUPUT FOLDER TO HAVE CLEAN OUTPUT: rm -rf {os.path.join(self.dataframe.output_dir, self.engine)}")
            else:
                logger.info(f"The output folder {os.path.join(self.dataframe.output_dir, self.engine)} is created")
                utils.mkdir(os.path.join(self.dataframe.output_dir, self.engine))


    def make_regression(self):
        """
        Compute systematic weight with the selected engine method and choosen hyperparameters.
        Some plots for inspection and the kfold indices are saved if an output directory is given in the dataframe.
        """
        # To store the result of the regression (ie) Y_pred
        F = np.zeros(self.dataframe.pixels.size)
        fold_index = dict()

        for zone_name in self.dataframe.region:
            if not self.dataframe.output_dir is None:
                save_info = True
                save_dir = os.path.join(self.dataframe.output_dir, self.engine, zone_name)
                if not os.path.isdir(save_dir):
                    os.mkdir(save_dir)
            else:
                save_info = False
                save_dir = None

            zone = self.dataframe.footprint[zone_name_to_column_name(zone_name)].values ## mask array

            logger.info(f"  ** {zone_name} :")
            X = self.dataframe.features[self.feature_names][zone]
            Y = self.dataframe.density[zone]
            keep_to_train_zone = self.dataframe.keep_to_train[zone]
            pixels_zone = self.dataframe.pixels[zone]

            logger.info(f"    --> Sample size {zone_name}: {keep_to_train_zone.sum()} -- Total Sample Size: {self.dataframe.keep_to_train.sum()} -- Training Fraction: {keep_to_train_zone.sum()/self.dataframe.keep_to_train.sum():.2%}")
            logger.info(f"    --> use Kfold training ? {self.use_kfold}")
            logger.info(f"    --> Engine: {self.engine} with params: {self.param_regressor[zone_name]}")

            if self.use_kfold:
                # TODO: regressor should be set in __init__().
                if self.engine == 'NN':
                    regressor = MLPRegressor(**self.param_regressor[zone_name])
                    normalized_feature, use_sample_weight = True, False
                elif self.engine == 'RF':
                    regressor = RandomForestRegressor(**self.param_regressor[zone_name])
                    normalized_feature, use_sample_weight = False, True
                elif self.engine == 'LINEAR':
                    regressor = LinearRegression(**self.param_regressor[zone_name])
                    normalized_feature, use_sample_weight = True, True

                F[zone], fold_index[zone_name] = Regressor.make_regressor_kfold(regressor, self.nfold[zone_name], X, Y, keep_to_train_zone,
                                                                                use_sample_weight, normalized_feature, self.feature_names_to_normalize,
                                                                                pixels_zone, self.dataframe.nside, feature_names=self.feature_names,
                                                                                compute_permutation_importance=self.compute_permutation_importance,
                                                                                save_regressor=self.save_regressor, save_info=save_info, save_dir=save_dir)
            else:
                # do not use k-fold training --> standard linear regression with iminuit
                F[zone] = Regressor.make_polynomial_regressor(X, Y, keep_to_train_zone, self.feature_names_to_normalize, self.param_regressor)
                fold_index = None

        # save evalutaion and fold_index
        # TODO: PEP8, self.F is not a clear name
        self.F = F # Y_pred for every entry in each zone
        self.fold_index = fold_index # fold_index --> les pixels de chaque fold dans chaque zone ! --> usefull to save if we want to reapply the regressor

        if save_info and (not fold_index is None):
            logger.info(f"    --> Save Kfold index in {os.path.join(self.dataframe.output_dir, self.engine, f'kfold_index.joblib')}")
            dump(self.fold_index, os.path.join(self.dataframe.output_dir, self.engine, f'kfold_index.joblib'))


    @staticmethod
    def build_kfold(kfold, pixels, group):
        """
        Build the folding of pixels with a GroupKfold class of Scikit-learn using a specific grouping given by group.
        A group cannot be splitted during the Kfold generation.

        Parameters:
        ----------
        kfold: GroupKFold class
            Scikit-learn class with split method to build the group Kfold.
        pixels: array like
            List of pixels which have to be splitted in Kfold
        group: array like
            Same size than pixels. It contains the group number of each pixel in pixels.
            A group cannot be splitted with kfold (ie) all pixels in the same group will be in the Fold.

        Returns:
        --------
        index: list of list
            Return a list (index == Fold number) containing the index list of pixels belonging to the fold i

        """
        index = []
        for index_train, index_test in kfold.split(pixels, groups=group):
            index += [index_test]
        return index

    @staticmethod
    def plot_kfold(nside, pixels, index_list, savename, title=''):
        """
        Plot the folding of pixels following the repartition given in index_list.

        Parameters:
        -----------
        nside: int
            Healpix resolution used in pixels
        pixels: array like
            pixels contains the pixel numbers which are split in the Kfold
        index_list: list of list
            list (index == Fold number) containing the index list of pixels belonging to the fold i --> output of *build_kfold*
        savename: str
            Path where the plot will be saved
        title: str
            Title for the figure
        """
        map = np.zeros(hp.nside2npix(nside))
        for i, index in enumerate(index_list) :
            map[pixels[index]] = i + 1 # to avoid 0
        map[map == 0] = np.NaN

        #attention au sens de l'axe en RA ! --> la on le prend normal et on le retourne à la fin :)
        plt.figure(1)
        map_to_plot = hp.cartview(map, nest=True, flip='geo', rot=120, fig=1, return_projected_map=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(10,6))
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
        plt.savefig(savename)
        plt.close()


    @staticmethod
    def make_regressor_kfold(regressor, nfold, X, Y, keep_to_train, use_sample_weight,
                             normalized_feature, feature_names_to_normalize,
                             pixels, nside, feature_names=None,
                             compute_permutation_importance=False,
                             save_regressor=False, save_info=False, save_dir=''):
        """
        Perform the Kfold training/evaluation of (X, Y) with regressor as engine for regression and nfold.
        The training is performed only with X[keep_to_train]. Warning the K-fold split is generated with pixels and not pixels[keep_to_train].
        This choice is done to have always the same splitting whatever the selection used to keep the training data.
        The Kfold is calibrated to create patch of 52 deg^2 each and to covert all the specific region of the footprint.
        Inspect the "kfold_repartition.png" for specific design.
        Note that the Kfold is purely geometrical meaning that if two rows in X have the same pixel value it has to be in the K-fold.

        Parameters:
        -----------
        regressor: Scikit-learn Regressor Class
            regressor used to perform the regression. No parameter will be modified here.
        nfold: int
            Number of fold which will be use in Kfold training.
        X: array like
            Feature dataset for the regression
        Y: array like
            The target values for the regression. Same size than X
        keep_to_train: bool array like
            Which row in X and Y is kept for the training. Same size than X and Y. The regression is then applied on all X.
        use_sample_weight: bool
            If true use 1/np.sqrt(Y_train) as weight during the regression. Only available for RF or LINEAR.
        normalized_feature: bool
            If True normalized and centered feature in feature_names_to_normalize -> mandatory for Linear ou MLP regression
        feature_names_to_normalize: array like
            List of feature names to normalize. 'STREAM' feature is already normalized do not normalize it again !!
        pixels: array like
            Pixels list corresponding of the pixel number of each feature row.
        nside: int
            Healpix resolution used in pixels
        feature_names: array like
            Feature name used during the regression. It is used for plotting information. If save_info is False do not need to pass it.
        compute_permutation_importance: bool
            If True compute and plot the permutation importance. The figure is saved only if save_info is True.
        save_regressor: bool
            If True save each regressor for each fold. Take care it is memory space consumming especially for RandomForestRegressor. Only available if save_info is True.
        save_info: bool
            If True save inspection plots and if it is required in save_regressor, the fitted regressor in save_dir.
        save_dir: str
            Directory path where the files will be saved if required with save_info.

        Returns:
        --------
        Y_pred: array like
            Gives the evaluation of the regression on X with K-folding. It has the same size of X.
            The evalutation is NOT applied ONLY where keep_to_train is True.
        index: list of list
            list (index == Fold number) containing the index list of pixels belonging to the fold i --> output of *build_kfold*
        """

        kfold = GroupKFold(n_splits=nfold)
        # TODO: size_group should be really hard-coded??
        size_group = 1000  * (nside / 256)**2 # define to have ~ 52 deg**2 for each patch (ie) group
        group = [i//size_group for i in range(pixels.size)]
        logger.info(f"    --> We use: {kfold} Kfold with group_size={size_group}")
        index = Regressor.build_kfold(kfold, pixels, group)
        if save_info:
            Regressor.plot_kfold(nside, pixels, index, os.path.join(save_dir, 'kfold_repartition.png'), title=f'{nfold}-Fold repartition')

        Y_pred = np.zeros(pixels.size)
        X.reset_index(drop=True, inplace=True)

        logger.info("    --> Train and eval for the fold :")
        start = time.time()
        for i in range(nfold):
            logger.info(f"        * {i}")
            fold_index = index[i]
            keep_to_train_fold = np.delete(keep_to_train, fold_index)
            logger.info(f"          --> There are {np.sum(keep_to_train_fold == 1)} pixels to train fold {i} which contains {np.sum(keep_to_train == 1) - np.sum(keep_to_train_fold == 1)} pixels (kept for the global training)")

            if normalized_feature:
                logger.info("          --> We normalize and center all features (execpt the STREAM) on the training footprint")
                X_fold = X.copy()
                X_fold[feature_names_to_normalize] = (X[feature_names_to_normalize] - X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean())/X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std()
                logger.info(f"          --> Mean of Mean and Std on all features : {X_fold[feature_names_to_normalize].mean().mean()} -- {X_fold[feature_names_to_normalize].std().mean()}")
                logger.info(f"          --> Mean of Mean and Std on the fold-training features : {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean().mean()} -- {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std().mean()}\n")
            else:
                logger.info("          --> We do NOT normalize feature in the training set --> not NEEDED")
                X_fold = X.copy()

            X_train, Y_train = X_fold.drop(fold_index)[keep_to_train_fold == 1], np.delete(Y, fold_index)[keep_to_train_fold == 1]
            if use_sample_weight:
                logger.info("          --> The training is done with sample_weight=1/np.sqrt(Y_train)")
                regressor.fit(X_train, Y_train, sample_weight=1/np.sqrt(Y_train))
            else:
                regressor.fit(X_train, Y_train)

            Y_pred_fold = np.zeros(fold_index.size)
            Y_pred_fold = regressor.predict(X_fold.iloc[fold_index])
            Y_pred[fold_index] = Y_pred_fold

            if save_info:
                # Save regressor
                if save_regressor:
                    dump(regressor, os.path.join(save_dir, f'regressor_fold_{i}.joblib'))

                #use only reliable pixel (ie) keep_to_train == 1 also in the fold !
                Regressor.plot_efficiency(Y[fold_index], Y_pred_fold, pixels[fold_index], keep_to_train[fold_index], os.path.join(save_dir, f"kfold_efficiency_fold_{i}.png"))

                #for more complex plot as importance feature ect ..--> save regressor and
                if os.path.basename(os.path.dirname(save_dir)) == 'RF':
                    Regressor.plot_importance_feature(regressor, feature_names, os.path.join(save_dir, f"feature_importance_fold_{i}.png"))

                if compute_permutation_importance:
                    Regressor.plot_permutation_importance(regressor, X_fold.iloc[fold_index], Y[fold_index], feature_names, os.path.join(save_dir, f"permutation_importance_fold_{i}.png"))

        logger.info("    --> Done in: {:.3f} s".format(time.time() - start))
        return Y_pred, index


    @staticmethod
    def make_polynomial_regressor(X, Y, keep_to_train, feature_names_to_normalize, param_regressor):
        """
        Perform the Kfold training/evaluation of (X, Y) with regressor as engine for regression and nfold.
        The training is performed only with X[keep_to_train]. Warning the K-fold split is generated with pixels and not pixels[keep_to_train].
        This choice is done to have always the same splitting whatever the selection used to keep the training data.
        The Kfold is calibrated to create patch of 52 deg^2 each and to covert all the specific region of the footprint.
        Inspect the "kfold_repartition.png" for specific design.
        Note that the Kfold is purely geometrical meaning that if two rows in X have the same pixel value it has to be in the K-fold.

        Parameters:
        -----------
        X: array like
            Feature dataset for the regression
        Y: array like
            The target values for the regression. Same size than X
        keep_to_train: bool array like
            Which row in X and Y is kept for the training. Same size than X and Y. The regression is then applied on all X.
        feature_names_to_normalize: array like
            List of feature names to normalize. 'STREAM' feature is already normalized do not normalize it again !!
        param_regressor: dict
            dictionary containing 'regulator' parameter. (It is a L1 regularisation term)

        Returns:
        --------
        Y_pred: array like
            Gives the evaluation of the regression on X with K-folding. It has the same size of X.
            The evalutation is NOT applied ONLY where keep_to_train is True.
        """
        def model(x, *par):
            return par[0]*np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

        nbr_features = X.shape[1]
        logger.info(f"[TEST] Number of features used : {nbr_features}")
        nbr_params = nbr_features + 1
        # TODO: English please!!!
        logger.info(f"            ** Taille de l'échantillon (non nan value): {np.sum(Y>0)}")
        logger.info(f"            ** Information sur normalized targets : Mean = {np.nanmean(Y)} and Std = {np.nanstd(Y)}")
        logger.info("[WARNING] We normalize and center features on the training footprint (don't forget to normalize also the non training footprint)")
        logger.info(feature_names_to_normalize)
        X.loc[:, feature_names_to_normalize] = (X[feature_names_to_normalize] - X[feature_names_to_normalize][keep_to_train == 1].mean())/X[feature_names_to_normalize][keep_to_train == 1].std()
        X_train, Y_train = X[keep_to_train == 1], Y[keep_to_train == 1]
        logger.info(f"[TEST] Mean of Mean and Std training features (should be 0, 1): {X_train[feature_names_to_normalize].mean().mean()} -- {X_train[feature_names_to_normalize].std().mean()}\n")

        dict_ini = {f'a{i}': 0 if i==0 else 0 for i in range(0, nbr_params)}
        dict_ini.update({f'error_a{i}': 0.001 for i in range(0, nbr_params)})
        dict_ini.update({f'limit_a{i}': (-1, 2) if i==0 else (-3,3) for i in range(0, nbr_params)})
        dict_ini.update({'errordef': 1}) #for leastsquare
        Y_cov_inv = np.diag(1/np.sqrt(Y_train))

        param = regression_least_square(model, param_regressor['regulator'], X_train, Y_train, Y_cov_inv, nbr_params, **dict_ini)

        logger.info(f"[TEST] Mean of systematics_correction : {model(X_train, *param).mean()} \n")

        return model(X, *param)


    def build_w_sys_map(self, return_map=True, savemap=False, savedir=None):
        """
        We save the healpix systematic maps

        Parameter:
        ----------
        return_map: bool
            if true return the sysematic weight map
        savemap: bool
            if True, save the map in savedir
        savedir: str
            path where to save the map, if None use default path
        """
        w = np.zeros(hp.nside2npix(self.dataframe.nside))*np.NaN
        w[self.dataframe.pixels[self.F > 0]] = 1.0/self.F[self.F > 0]

        if savemap:
            if savedir is None:
                savedir = os.path.join(self.dataframe.output_dir, self.engine)
            filename_weight_save = os.path.join(savedir, f'{self.dataframe.version}_{self.dataframe.tracer}_imaging_weight_{self.dataframe.nside}.npy')
            logger.info(f"Save photometric weight in a healpix map with {self.dataframe.nside} here: {filename_weight_save}")
            np.save(filename_weight_save, w)

        if return_map:
            return w


    @staticmethod
    def plot_efficiency(Y, Y_pred, pixels, keep_to_train, path_to_save):
        """
        Draw the correction 'efficiency' of the training. The method excepts to reduce the variance in the histogram of the predicted density.

        Parameters:
        -----------
        Y: array like
            The normalized target density.
        Y_pred: array like
            Same size than Y. The perdicted normalized target density with K-fold (Y is not used for the prediction). 1/Y_pred will be the sys weights.
        keep_to_train: bool array like
            Same size than Y. Mask array given which row of Y was used in the training (ie) Y is not an outlier
        path_to_save: str
            Where the figure will be save
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(left=0.07, right=0.96, bottom=0.1, top=0.9, wspace=0.3)
        ax[0].scatter(pixels[keep_to_train == 1], Y[keep_to_train == 1], color = 'red', label='Initial (before regression)')
        ax[0].scatter(pixels[keep_to_train == 1], Y_pred[keep_to_train == 1], color = 'blue', label='Corrected (after regression)')
        ax[0].legend()
        ax[0].set_xlabel('Pixel Number')
        ax[0].set_ylabel('Normalized Targets Density')

        ax[1].hist(Y[keep_to_train == 1], color='blue', bins=50, range=(0.,2.), density=1, label='Initial')
        ax[1].hist(Y_pred[keep_to_train == 1], color='red', histtype='step', bins=50, range=(0.,2.), density=1, label='corrected')
        ax[1].legend()
        ax[1].set_xlabel('Normalized Targets Density')

        plt.tight_layout()
        plt.savefig(path_to_save)
        plt.close()


    @staticmethod
    def plot_importance_feature(regressor, feature_names, path_to_save, max_num_feature=8):
        """
        Plot the giny importance feature for RandomForestRegressor.

        Parameters:
        regressor: RandomForestRegressor
            Regressor already trained
        feature_names: str array like
            List of feature used during the regression
        path_to_save: str
            Where the figure will be saved.
        max_num_feature: int
            Number of feature to plot in the figure.
        """

        import seaborn as sns

        feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names, columns=['feature importance']).sort_values('feature importance', ascending=False)
        feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names)
        feature_all = pd.melt(feature_all, var_name='feature name', value_name='values')

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        sns.swarmplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], alpha=0.7, size=2, color="k")
        sns.boxplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], fliersize=0.6, palette=sns.color_palette("husl", 8), linewidth=0.6, showmeans=False, meanline=True, meanprops=dict(linestyle=':', linewidth=1.5, color='dimgrey'))
        ax.set_xticklabels(feature_importance.index[:max_num_feature], rotation=15, ha='center')
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(path_to_save)


    @staticmethod
    def plot_permutation_importance(regressor, X, Y, feature_names, path_to_save):
        """
        Compute and plot the permutation importance for the regressor (alternative/ complementary metric to giny importance)

        Parameters:
        regressor: Scikit-learn regressor class
            Regressor already trained
        X: array like
            Feature array
        Y: array like
            Same size than X.
        feature_names: str array like
            List of feature used during the regression
        path_to_save: str
            Where the figure will be saved.
        """
        from sklearn.inspection import permutation_importance
        ## The permutation feature importance is defined to be the decrease in a model score
        ## when a single feature value is randomly shuffled. This procedure breaks the relationship
        ## between the feature and the target, thus the drop in the model score is indicative of
        ## how much the model depends on the feature.
        logger.info("          --> Compute permutation importance feature ...")
        permut_importance = permutation_importance(regressor, X, Y, n_repeats=15, random_state=4)

        fig, ax = plt.subplots()
        ax.boxplot(permut_importance.importances.T, vert=False, labels=feature_names)
        ax.set_title(f"Permutation Importance")
        ax.set_ylabel("Features")
        fig.tight_layout()
        plt.savefig(path_to_save)
        plt.close()


    def plot_maps_and_systematics(self, max_plot_cart=400, ax_lim=0.2, adaptative_binning=False, nobjects_by_bins=2000, n_bins=None, cut_fracarea=True, min_fracarea=0.9, max_fracarea=1.1,):
        """
        Make plot to check and validate the regression.
        The result are saved in the corresponding outpur directory.

        Parameters:
        -----------
        max_plot_cart: float
            maximum density used to plot the object density in the sky
        ax_lim: float
            maximum value of density - 1 fluctuation in function of the feature
        adaptative_binning: bool

        nobjects_by_bins: int

        n_bins: int

        cut_fracarea: bool

        min_fracarea, max_fracarea: float
        """

        from plot import plot_moll
        from systematics import plot_systematic_from_map

        dir_output = os.path.join(self.dataframe.output_dir, self.engine, 'Fig')
        if not os.path.isdir(dir_output):
            os.mkdir(dir_output)
        logger.info(f"Save density maps and systematic plots in the output directory: {dir_output}")

        targets = self.dataframe.targets / (hp.nside2pixarea(self.dataframe.nside, degrees=True)*self.dataframe.fracarea)
        targets[~self.dataframe.footprint['FOOTPRINT']] = np.nan

        w = np.zeros(hp.nside2npix(self.dataframe.nside))
        w[self.dataframe.pixels[self.F>0]] = 1.0/self.F[self.F>0]
        targets_without_systematics = targets*w

        plot_moll(hp.ud_grade(targets, 64, order_in='NESTED'), min=0, max=max_plot_cart, show=False, savename=os.path.join(dir_output, 'targerts.pdf'), galactic_plane=True, ecliptic_plane=True)
        plot_moll(hp.ud_grade(targets_without_systematics, 64, order_in='NESTED'), min=0, max=max_plot_cart,  show=False, savename=os.path.join(dir_output, 'targets_without_systematics.pdf'), galactic_plane=True, ecliptic_plane=True)
        map_to_plot = w.copy()
        map_to_plot[map_to_plot == 0] = np.NaN
        map_to_plot = map_to_plot - 1
        plot_moll(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.2, max=0.2, label='weight - 1',  show=False, savename=os.path.join(dir_output, 'systematic_weights.pdf'), galactic_plane=True, ecliptic_plane=True)

        plot_systematic_from_map([targets, targets_without_systematics], ['No correction', 'Systematics correction'], self.dataframe.fracarea, self.dataframe.footprint, self.dataframe.features, dir_output, self.dataframe.region,
                                  ax_lim=ax_lim, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins, n_bins=n_bins,
                                  cut_fracarea=cut_fracarea, min_fracarea=min_fracarea, max_fracarea=max_fracarea)
