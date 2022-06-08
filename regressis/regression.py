#!/usr/bin/env python
# coding: utf-8

import os
import time
import logging

import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump

from . import utils
from .weight import PhotoWeight


logger = logging.getLogger("Regression")


_all_feature_names = ['STARDENS', 'EBV', 'STREAM',
                      'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']

_all_regions = ['North', 'South', 'Des', 'South_all', 'South_mid', 'South_mid_ngc', 'South_mid_sgc', 'South_pole', 'South_mid_no_des', 'Des_mid']


def _get_feature_names(tracer=None, use_stream=None, use_stars=None, feature_names=None):
    """
    Return the default set of features for regression.

    Parameters
    ----------
    tracer : str, default=None
        The tracer name e.g. QSO / ELG / LRG / BGS.
    use_stream : bool, default=None
        Use Sgr. Stream as template; default is ``False`` for all and ``True`` for QSO.
    use_stars : bool, default=None
        Use stardens as template; default is ``True`` for all.
    feature_names : list, default=_all_feature_names
        List of all features to select from.

    Returns
    -------
    feature_names : list
        Names of features to regress against.
    """
    if feature_names is None:
        feature_names = _all_feature_names
    feature_names = feature_names.copy()
    to_remove = []

    if tracer == 'QSO':
        if use_stream is None: use_stream = True
    elif tracer == 'ELG':
        to_remove = ['PSFDEPTH_W1', 'PSFDEPTH_W2']
    elif tracer == 'LRG':
        to_remove = ['PSFDEPTH_W2']
    elif tracer == 'BGS':
        to_remove = ['PSFDEPTH_W1', 'PSFDEPTH_W2']

    if use_stars is None: use_stars = True
    if use_stream is None: use_stream = False

    if not use_stream:
        to_remove.append('STREAM')

    if not use_stars:
        to_remove.append('STARDENS')

    for name in to_remove:
        if name in feature_names: feature_names.remove(name)

    return utils.unique_list(feature_names)


def _format_regressor_params(params, regions=None):
    """
    Format regressor hyperparameters, i.e. return a dictionary of dictionary of parameters for each input region.

    Parameters
    ----------
    params : dict, default=None
        Parameters.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for the regressor.
    """
    params = params or {}
    if regions is None: regions = _all_regions
    if not any(region in params for region in regions):
        params = {region: params for region in regions}
    return params


def _get_rf_params(params=None, n_jobs=6, seed=123, regions=None):
    """
    Return pre-defined hyperparameters for the :class:`RandomForestRegressor` for each region.
    Can be updated with ``params``.

    Parameters
    ----------
    params : dict, default=None
        Override parameters, e.g. {'North': {'n_estimators': 20}}.
    n_jobs : int, default=6
        Number of jobs, to parallelize the computation.
    seed : int, default=123
        Fix the random state for reproducibility.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for the regressor.
    """
    default = _format_regressor_params({'n_estimators': 200, 'min_samples_leaf': 20, 'max_depth': None,
                                        'max_leaf_nodes': None, 'n_jobs': n_jobs, 'random_state': seed},
                                       regions=regions)
    utils.deep_update(default, _format_regressor_params(params or {}, regions=regions))
    return default


def _get_mlp_params(params=None, seed=123, regions=None):
    """
    Return pre-defined hyperparameters for the :class:`MLPRegressor` for each region.
    Can be updated with ``params``.

    Parameters
    ----------
    params : dict, default=None
        Override parameters, e.g. {'North': {'max_iter': 20}}.
    seed : int, default=123
        Fix the random state for reproducibility.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for the regressor.
    """
    default = _format_regressor_params({'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8),
                                        'max_iter': 6000, 'n_iter_no_change': 100, 'solver': 'adam', 'tol': 1e-5, 'random_state': seed},
                                       regions=regions)
    utils.deep_update(default, _format_regressor_params(params or {}, regions=regions))
    return default


def _get_linear_params(params=None, regions=None):
    """
    Return pre-defined hyperparameters for the :class:`LinearRegression` for each region.
    Can be updated with ``params``.

    Parameter
    ---------
    params : dict, default=None
        Override parameters, e.g. {'North': {'n_jobs': 2}}.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    params : dict
        Dictionary containing for each region a dictionary of the hyperparameters for the regressor.
    """
    default = _format_regressor_params(dict(), regions=regions)
    utils.deep_update(default, _format_regressor_params(params or {}, regions=regions))
    return default


def _get_nfold_params(params=None, regions=None):
    """
    Return pre-defined number of folds for each specific region available.
    Can be updated with ``params``.

    Parameters
    ----------
    params : dict, default=None
        Override parameters, e.g. {'North': 2}.
    regions : list, default=None
        List of regions. Defaults to ``_all_regions``.

    Returns
    -------
    param : dict
        Dictionary containing for each region the number of folds for the k-fold training.
    """
    default = _format_regressor_params({'North': 6, 'South': 12, 'Des': 6, 'South_all': 18, 'South_mid': 14, 'South_mid_ngc': 7, 'South_mid_sgc': 7, 'South_pole': 5, 'Des_mid': 3}, regions=regions)
    utils.deep_update(default, _format_regressor_params(params or {}, regions=regions))
    return default


class Regression(object):

    """Implementation of template fitting regression."""

    def __init__(self, dataframe, regressor='RF', suffix_regressor='', feature_names=None, use_kfold=True,
                 regressor_params=None, nfold_params=None, compute_permutation_importance=True,
                 overwrite=False, save_regressor=False, n_jobs=6, seed=123):
        """
        Initialize :class:`Regression`, i.e. train input ``regressor`` on input ``dataframe``.

        Parameters
        ----------
        dataframe : DataFrame
            Data frame containing all the information to fit/train the regressor.
        regressor : str, scikit-learn Regressor
            Either "RF" (random forest), "NN" (multi layer perceptron), "Linear", or scikit-learn Regressor class or instance.
        suffix_regressor: str
            Additional suffix for regressor. Used only for output directory / names. Useful to compare same regressor with different hyperparameters or features.
        feature_names : list of str, default=None
            Names of features used during the regression. By default get feature names from :func:`_get_feature_names`.
        use_kfold : bool, default=True
            If ``True`` use k-fold to fit/predict target density. It is mandatory with RF and NN to avoid strong overfitting.
        regressor_params : dict, default=None
            Dictionary specific hyperparameters to update the default values loaded in _get_rf_params / _get_mlp_params / _get_linear_params.
            Can be provided as a single dictionary, applicable to all regions, or as a dictionary containing parameters for each region of :attr:`dataframe.regions`.
        compute_permutation_importance : bool, default=True
            Compute and plot the permutation importance for inspection. It has to be compared with giny importance from RF regressor.
        overwrite : bool, default=False
            If ``True`` and the output directory :attr:`dataframe.output_dir` is set, overwrite files (if they exist).
            If ``False``, and :attr:`dataframe.output_dir` exists, an error is raised.
        save_regressor : bool, default=False
            If ``True`` regressor for each region and each fold is saved. Warning: it is space consuming. Can be useful to make some more advanced plots.
        n_jobs : int, default=6
            To parallelize the RF training.
        seed : int, default=123
            Fix the random state of RF and NN for reproducibility.
        """
        if getattr(dataframe, 'density', None) is None: dataframe.build()
        self.dataframe = dataframe

        if feature_names is None:
            self.feature_names = _get_feature_names(dataframe.tracer)
        else:
            self.feature_names = utils.unique_list(feature_names)
        logger.info(f"We use the set of features: {self.feature_names}")

        # set up the parameter for the considered regressor
        self.use_kfold = use_kfold
        if not self.use_kfold:
            logger.warning("k-fold training not used")

        self.regressor_params = _format_regressor_params(regressor_params or {}, regions=self.dataframe.regions)
        self.regressor = regressor
        self.suffix_regressor = suffix_regressor

        if isinstance(regressor, str):
            self.regressor_name = regressor
            if self.regressor_name.upper() == 'RF':
                self.regressor = RandomForestRegressor()
                self.regressor_params = _get_rf_params(self.regressor_params, n_jobs, seed, regions=self.dataframe.regions)
                self.normalized_feature, self.use_sample_weight = False, True
            elif self.regressor_name.upper() == 'NN':
                self.regressor = MLPRegressor()
                self.regressor_params = _get_mlp_params(self.regressor_params, seed, regions=self.dataframe.regions)
                self.normalized_feature, self.use_sample_weight = True, False
            elif self.regressor_name.upper() == 'LINEAR':
                self.regressor = LinearRegression()
                self.regressor_params = _get_linear_params(self.regressor_params, regions=self.dataframe.regions)
                self.normalized_feature, self.use_sample_weight = True, True
            else:
                raise ValueError(f'Unknown regressor {regressor}. Choices are ["RF", "NN", "LINEAR"].')
        else:
            self.regressor_name = self.regressor.__class__.__name__

        # in NN and Linear case, we normalize and standardize the data
        # STREAM is already normalized remove it from the list
        if self.regressor_name.upper() != 'RF':
            self.feature_names_to_normalize = self.feature_names.copy()
            if 'STREAM' in self.feature_names_to_normalize: self.feature_names_to_normalize.remove('STREAM')
        else:
            self.feature_names_to_normalize = None

        self.compute_permutation_importance = compute_permutation_importance
        self.save_regressor = save_regressor
        self.nfold = _get_nfold_params(nfold_params, regions=self.dataframe.regions)

        # If not self.dataframe.output is None --> save figure and info !!
        if self.dataframe.output_dir is not None:
            # create the corresponding output folder --> put here since self.regressor_name can be update with use_kfold = False
            if os.path.isdir(os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor)):
                if not overwrite:
                    raise ValueError(f"{os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor)} already exists and overwrite is set as {overwrite}")
                logger.warning(f"Overwriting {os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor)}")
                logger.warning(f"Please remove the output folder to have a clean output: rm -rf {os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor)}")
            else:
                logger.info(f"The output folder {os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor)} is created")
            utils.mkdir(os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor))
        self.run()

    def run(self):
        """
        Predict the density with the provided engine and hyperparameters.
        If :attr:`use_kfold`, k-fold training is performed, based on patches of 52 deg^2.
        Some plots for inspection and the k-fold indices (if relevant) are saved the output directory of :attr:`dataframe` if provided.
        """
        # To store the result of the regression i.e. Y_pred
        Y_pred = np.zeros(self.dataframe.pixels.size)
        fold_index = dict()

        for region in self.dataframe.regions:
            if self.dataframe.output_dir is not None:
                save_info = True
                save_dir = os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor, region)
                utils.mkdir(save_dir)
            else:
                save_info = False
                save_dir = None

            zone = self.dataframe.footprint(region)  # mask array

            logger.info(f"  ** {region} :")
            X = self.dataframe.features[self.feature_names][zone]
            Y = self.dataframe.density[zone]
            keep_to_train_zone = self.dataframe.keep_to_train[zone]
            pixels_zone = self.dataframe.pixels[zone]

            logger.info(f"    --> Sample size {region}: {keep_to_train_zone.sum()} -- Total sample size: {self.dataframe.keep_to_train.sum()} -- Training fraction: {keep_to_train_zone.sum()/self.dataframe.keep_to_train.sum():.2%}")
            logger.info(f"    --> Use k-fold training ? {self.use_kfold}")
            params = self.regressor_params[region]
            logger.info(f"    --> Engine: {self.regressor_name} with params: {params}")
            if params:
                self.regressor.set_params(**params)

            if self.use_kfold:
                size_group = 1000 * (self.dataframe.nside / 256)**2  # define to have ~ 52 deg**2 for each patch (ie) group
                Y_pred[zone], fold_index[region] = self.fit_and_predict_on_kfold(self.regressor, self.nfold[region], size_group, X, Y, pixels_zone, self.dataframe.nside, keep_to_train_zone,
                                                                                 use_sample_weight=self.use_sample_weight, feature_names=self.feature_names, feature_names_to_normalize=self.feature_names_to_normalize,
                                                                                 compute_permutation_importance=self.compute_permutation_importance,
                                                                                 save_regressor=self.save_regressor, save_info=save_info, save_dir=save_dir)
            else:
                Y_pred[zone] = self.fit_and_predict(self.regressor, X[keep_to_train_zone], Y[keep_to_train_zone], X, Y_eval=Y, use_sample_weight=self.use_sample_weight,
                                                    feature_names=self.feature_names, feature_names_to_normalize=self.feature_names_to_normalize,
                                                    compute_permutation_importance=self.compute_permutation_importance,
                                                    save_regressor=self.save_regressor, save_info=save_info, save_dir=save_dir)
                fold_index = None

        # save evalutaion and fold_index
        self.Y_pred = Y_pred  # Y_pred for every entry in each zone
        self.fold_index = fold_index  # fold_index --> useful to save if we want to reapply the regressor

        if save_info and fold_index is not None:
            logger.info(f"    --> Save k-fold index in {os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor, f'kfold_index.joblib')}")
            dump(self.fold_index, os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor, 'kfold_index.joblib'))

    @staticmethod
    def build_kfold(kfold, pixels, groups):
        """
        Build the folding of pixels with scikit-learn's :class:`~sklearn.model_selection.GroupKfold` using a specific grouping given by group.
        All pixels in the same group will be in the same fold.

        Parameters
        ----------
        kfold : GroupKfold
            scikit-learn class with split method to build the group k-fold.
        pixels : array like
            List of pixels which must be splitted in k-fold.
        groups : array like
            Same size as pixels. It contains the group label for each pixel in pixels.
            All pixels in the same group will be in the same fold.

        Returns
        -------
        index : list of list
            For each fold, the index list of pixels belonging to that fold.
        """
        index = []
        for index_train, index_test in kfold.split(pixels, groups=groups):
            index += [index_test]
        return index

    @staticmethod
    def plot_kfold(nside, pixels, index_list, filename, title=''):
        """
        Plot the folding of pixels following the repartition given in index_list.

        Parameters
        ----------
        nside : int
            Healpix resolution used in pixels.
        pixels : array like
            Pixel numbers which are split in the k-fold
        index_list : list of list
            For each fold, the index list of pixels belonging to that fold; output of :meth:`Regression.build_kfold`.
        filename : str
            Path where the plot will be saved.
        title : str
            Title for the figure.
        """
        map = np.zeros(hp.nside2npix(nside))
        for i, index in enumerate(index_list):
            map[pixels[index]] = i + 1  # to avoid 0
        map[map == 0] = np.nan

        # attention au sens de l'axe en RA ! --> la on le prend normal et on le retourne à la fin :)
        plt.figure(1)
        map_to_plot = hp.cartview(map, nest=True, flip='geo', rot=120, fig=1, return_projected_map=True)
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.imshow(map_to_plot, interpolation='nearest', cmap='jet', origin='lower', extent=[-60, 300, -90, 90])
        ax.set_xlim(-60, 300)
        ax.xaxis.set_ticks(np.arange(-60, 330, 30))
        plt.gca().invert_xaxis()
        ax.set_xlabel('R.A. [deg]')
        ax.set_ylim(-90, 90)
        ax.yaxis.set_ticks(np.arange(-90, 120, 30))
        ax.set_ylabel('Dec. [deg]')
        ax.grid(True, alpha=0.8, linestyle=':')
        plt.title(title)
        plt.savefig(filename)
        plt.close()

    @classmethod
    def fit_and_predict(cls, regressor, X_train, Y_train, X_eval, Y_eval=None, use_sample_weight=False,
                        feature_names=None, feature_names_to_normalize=None, compute_permutation_importance=False,
                        save_regressor=False, save_info=False, save_dir='', suffix=''):
        """
        Perform the training of the regressor with (X_train, Y_train) and evaluate it in ``X_eval``.
        ``Y_eval`` is only used to plot efficiency and permutation importance if required.

        Parameters
        ----------
        regressor : scikit-learn Regressor
            Regressor instance used to perform the regression.
            The regressor will be fitted with ``(X_train, Y_train)`` and will be evaluated at ``X_eval``.
        X_train : array like
            Data frame used for the training.
        Y_train : array like
            Target for ``X_train``.
        X_eval : array like
            Data frame used for the evaluation.
        Y_eval : array like, default=None
            Target for ``X_eval``. Only if ``save_info`` is ``True``.
        use_sample_weight : bool, default=False
            If ``True`` use ``1/np.sqrt(Y_train)`` as weight during the regression. Only available for RF or LINEAR.
        feature_names : array like, default=None
            Names of features used during the regression. It is used for plotting information. Only used if ``save_info`` is ``True``.
        feature_names_to_normalize : array like, default=None
            Names of features to normalize ; mandatory for LINEAR or NN regression. 'STREAM' feature is already normalized; do not normalize it again.
        compute_permutation_importance : bool, default=False
            If ``True`` compute and plot the permutation importance. Only if ``save_info`` is ``True``.
        save_regressor : bool, default=False
            If ``True`` save each regressor. Take care it is memory space consumming especially for RF. Only available if ``save_info`` is ``True``.
        save_info : bool, default=False
            If ``True`` save inspection plots and if it is required in ``save_regressor``, the fitted regressor in ``save_dir``.
        save_dir : str, default=''
            Directory path where the files will be saved if required with ``save_info``.
        suffix : str, default=''
            Suffix used to save output.

        Returns
        -------
        Y_pred: array like
            Returns the evaluation of the regression on ``X_eval`` with fitting done on ``(X_train, Y_train)``. It has the same size of ``X_eval``.
        """
        if feature_names_to_normalize:
            logger.info("          --> All features (except the STREAM) are normalized and centered on the training footprint")
            mean, std = X_train[feature_names_to_normalize].mean(), X_train[feature_names_to_normalize].std()
            X_for_training = X_train.copy()
            X_for_training[feature_names_to_normalize] = (X_train[feature_names_to_normalize] - mean) / std
            X_eval[feature_names_to_normalize] = (X_eval[feature_names_to_normalize] - mean) / std
            logger.info(f"          --> Mean of mean and std on the fold-training features: {X_for_training[feature_names_to_normalize].mean().mean():2.4f} -- {X_for_training[feature_names_to_normalize].std().mean():2.2f}")
        else:
            logger.info("          --> Features are not normalized")
            X_for_training = X_train.copy()

        if use_sample_weight:
            logger.info("          --> The training is done with sample_weight = 1/np.sqrt(Y_train)")
            regressor.fit(X_for_training, Y_train, sample_weight=1 / np.sqrt(Y_train))
        else:
            regressor.fit(X_for_training, Y_train)

        Y_pred = regressor.predict(X_eval)

        if save_info:
            # Save regressor
            if save_regressor:
                dump(regressor, os.path.join(save_dir, f'regressor{suffix}.joblib'))

            cls.plot_efficiency(Y_eval, Y_pred, os.path.join(save_dir, f"kfold_efficiency{suffix}.png"))

            # for more complex plot as importance feature ect ..--> save regressor and
            if hasattr(regressor, 'feature_importances_'):
                cls.plot_importance_feature(regressor, feature_names, os.path.join(save_dir, f"feature_importance{suffix}.png"))

            if compute_permutation_importance:
                cls.plot_permutation_importance(regressor, X_eval, Y_eval, feature_names, os.path.join(save_dir, f"permutation_importance{suffix}.png"))

        return Y_pred

    @classmethod
    def fit_and_predict_on_kfold(cls, regressor, nfold, size_group, X, Y, pixels, nside, keep_to_train=None, use_sample_weight=False,
                                 feature_names=None, feature_names_to_normalize=None, compute_permutation_importance=False,
                                 save_regressor=False, save_info=False, save_dir=''):
        """
        Perform the k-fold training/evaluation of ``(X, Y)`` with ``regressor`` as engine.
        The training is performed only with ``X[keep_to_train]``. Warning: the k-fold split is generated with pixels and not ``pixels[keep_to_train]``;
        this is to use the same splitting whatever the selection applied on training data.
        The k-fold is purely geometrical. Inspect the "kfold_repartition.png" for specific design.
        TODO: same pixels (if some appear several times) should be in the same k-fold. There are no such pixel duplicates in the case of photometric systematics,
        but there may be if spectroscopic features are added.

        Parameters
        ----------
        regressor : scikit-learn Regressor
            Regressor instance used to perform the regression.
        nfold : int
            Number of folds to be used in k-fold training.
        size_group : int
            Number of pixels in each group. All pixels in the same group will be in the same fold.
        X : array like
            Features for the regression.
        Y : array like
            The target values for the regression. Same length as ``X``.
        pixels : array like
            Pixels; same length as ``X``.
        nside : int
            Healpix resolution used in pixels.
        keep_to_train : bool array like, default=None
            Rows ``X`` and ``Y`` to keep for the training. Same length as X and Y. The evaluation is then applied on all ``X``.
            Defaults to all rows kept for the training.
        use_sample_weight : bool, default=False
            If ``True`` use ``1/np.sqrt(Y_train)`` as weight during the regression. Only available for RF or LINEAR.
        feature_names : array like, default=None
            Names of features used during the regression. It is used for plotting information. Only used if ``save_info`` is ``True``.
        feature_names_to_normalize : array like, default=None
            Names of features to normalize; mandatory for LINEAR or NN regression. 'STREAM' feature is already normalized; do not normalize it again.
        compute_permutation_importance : bool, default=False
            If ``True`` compute and plot the permutation importance. Only if ``save_info`` is ``True``.
        save_regressor : bool, default=False
            If ``True`` save each regressor for each fold. Take care it is memory space consumming especially for RF. Only available if ``save_info`` is ``True``.
        save_info : bool, default=False
            If ``True`` save inspection plots and if it is required in ``save_regressor``, the fitted regressor in ``save_dir``.
        save_dir : str, default=''
            Directory path where the files will be saved if required with ``save_info``.
        suffix : str, default=''
            Suffix used to save output.

        Returns
        -------
        Y_pred : array like
            Evaluation of the model on ``X`` with k-folding. It has the same shape as ``Y``.
            Evaluation is not restricted to ``keep_to_train``.
        index : list of list
            For each fold, the index list of pixels belonging to that fold; output of :meth:`Regression.build_kfold`.
        """
        kfold = GroupKFold(n_splits=nfold)
        groups = [i // size_group for i in range(pixels.size)]
        logger.info(f"    --> We use: {kfold} k-fold with group_size = {size_group}")
        index = cls.build_kfold(kfold, pixels, groups)
        if save_info:
            cls.plot_kfold(nside, pixels, index, os.path.join(save_dir, 'kfold_repartition.png'), title=f'{nfold}-Fold repartition')

        Y_pred = np.zeros(pixels.size)
        X.reset_index(drop=True, inplace=True)
        if keep_to_train is None: keep_to_train = np.ones(len(Y), dtype='?')

        logger.info("    --> Train and eval for the fold:")
        start = time.time()
        for i in range(nfold):
            logger.info(f"        * {i}")
            fold_index = index[i]
            # select row for the training i.e. remove fold index
            X_fold, Y_fold, keep_to_train_fold = X.drop(fold_index), np.delete(Y, fold_index), np.delete(keep_to_train, fold_index)
            # select row for the evaluation
            X_eval, Y_eval = X.iloc[fold_index], Y[fold_index]
            logger.info(f"          --> There are {np.sum(keep_to_train_fold == 1)} pixels to train fold {i} which contains {np.sum(keep_to_train == 1) - np.sum(keep_to_train_fold == 1)} pixels (kept for the global training)")

            Y_pred[fold_index] = cls.fit_and_predict(regressor, X_fold[keep_to_train_fold], Y_fold[keep_to_train_fold], X_eval, Y_eval=Y_eval,
                                                     use_sample_weight=use_sample_weight, feature_names=feature_names, feature_names_to_normalize=feature_names_to_normalize,
                                                     compute_permutation_importance=compute_permutation_importance, save_regressor=save_regressor, save_info=save_info, save_dir=save_dir, suffix=f'_fold_{i}')

        logger.info("    --> Done in: {:.3f} s".format(time.time() - start))
        return Y_pred, index

    def get_weight(self, save=False, savedir=None):
        """
        Save the healpix systematic maps.

        Parameters
        ----------
        save_map : bool, default=False
            Whether to save the map in ``savedir``.

        savedir : str, default=None
            Directory where to save the map; if ``None`` use default path.

        Returns
        -------
        w : ``PhotoWeight`` class
            Weight class with callable function to apply it into a real catalogue.
        """
        w = np.zeros(hp.nside2npix(self.dataframe.nside)) * np.nan
        w[self.dataframe.pixels[self.Y_pred > 0]] = 1.0 / self.Y_pred[self.Y_pred > 0]
        weight = PhotoWeight(sys_weight_map=w, regions=self.dataframe.regions,
                             mask_region={region: self.dataframe.footprint(region) for region in self.dataframe.regions},
                             mean_density_region=self.dataframe.mean_density_region)

        if save:
            if savedir is None:
                savedir = os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor)
            filename_save = os.path.join(savedir, f'{self.dataframe.version}_{self.dataframe.tracer}{self.dataframe.suffix_tracer}_imaging_weight_{self.dataframe.nside}.npy')
            weight.save(filename_save)

        return weight

    @staticmethod
    def plot_efficiency(Y, Y_pred, filename):
        """
        Plot the 'efficiency' of the prediction.
        Compares the histogram of the initial and predicted target densities.

        Parameters
        ----------
        Y : array like
            The normalized target density.
        Y_pred : array like
            Same size as ``Y``. The predicted normalized target density.
        filename : str
            Where the figure will be saved.
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(left=0.07, right=0.96, bottom=0.1, top=0.9, wspace=0.3)
        ax[0].scatter(np.arange(Y.size), Y, color='red', label='Initial (before regression)')
        ax[0].scatter(np.arange(Y_pred.size), Y_pred, color='blue', label='Predicted (after regression)')
        ax[0].legend()
        ax[0].set_xlabel('Pixel Number')
        ax[0].set_ylabel('Normalized Target Density')

        ax[1].hist(Y, color='blue', bins=50, range=(0., 2.), density=1, label='Initial')
        ax[1].hist(Y_pred, color='red', histtype='step', bins=50, range=(0., 2.), density=1, label='Predicted')
        ax[1].legend()
        ax[1].set_xlabel('Normalized Target Density')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_importance_feature(regressor, feature_names, filename, max_num_feature=8):
        """
        Plot the giny importance feature for RF.

        Parameters
        ----------
        regressor : RandomForestRegressor
            Regressor instance already fitted.
        feature_names : str array like
            List of features used during the regression.
        filename : str
            Where the figure will be saved.
        max_num_feature : int, default=8
            Number of features to plot in the figure.
        """
        import pandas as pd
        import seaborn as sns

        feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names, columns=['feature importance']).sort_values('feature importance', ascending=False)
        feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names)
        feature_all = pd.melt(feature_all, var_name='feature name', value_name='values')

        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        sns.swarmplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], alpha=0.7, size=2, color="k")
        sns.boxplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], fliersize=0.6, palette=sns.color_palette("husl", 8), linewidth=0.6, showmeans=False, meanline=True, meanprops=dict(linestyle=':', linewidth=1.5, color='dimgrey'))
        ax.set_xticklabels([utils.to_tex(name) for name in feature_importance.index[:max_num_feature]], rotation=15, ha='center')
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    @staticmethod
    def plot_permutation_importance(regressor, X, Y, feature_names, filename):
        """
        Compute and plot the permutation importance for the regressor (alternative / complementary metric to giny importance).

        Parameters
        ----------
        regressor: Scikit-learn regressor class
            Regressor instance already fitted.
        X : array like
            Features.
        Y : array like
            Same length as X.
        feature_names : str array like
            List of feature used during the regression.
        filename : str
            Where the figure will be saved.
        """
        from sklearn.inspection import permutation_importance
        # The permutation feature importance is defined to be the decrease in a model score
        # when a single feature value is randomly shuffled. This procedure breaks the relationship
        # between the feature and the target, thus the drop in the model score is indicative of
        # how much the model depends on the feature.

        if np.isfinite(Y).sum() == 0:
            logger.info("          --> No data to compute permutation feature...")
        else:
            logger.info("          --> Compute permutation importance feature...")
            # Warning: If you do not use keep_to_train, you can have some weird value in Y (mostly at the border of the footprint)
            permut_importance = permutation_importance(regressor, X[np.isfinite(Y)], Y[np.isfinite(Y)], n_repeats=20, random_state=4)

            fig, ax = plt.subplots()
            ax.boxplot(permut_importance.importances.T, vert=False, labels=[utils.to_tex(name) for name in feature_names])
            ax.set_title("Permutation Importance")
            ax.set_ylabel("Features")
            fig.tight_layout()
            plt.savefig(filename)
            plt.close()

    def plot_maps_and_systematics(self, show=False, save=True, max_plot_cart=400, ax_lim=0.2,
                                  adaptative_binning=False, nobj_per_bin=2000, n_bins=None,
                                  cut_fracarea=True, limits_fracarea=(0.9, 1.1),
                                  save_table=False, save_table_suffix=''):
        """
        Make plot to check and validate the regression.
        The results are saved in the corresponding output directory.

        Parameters
        ----------
        show: bool
            If True display the figure.
        save: bool
            If True save the figure in os.path.join(self.dataframe.output_dir, self.regressor_name+self.suffix_regressor, 'Fig').
            Directory is created if it does not exist !
        max_plot_cart : float, default=400
            Maximum density in the plot of object density in the sky.
        ax_lim : float, default=0.2
            Maximum value of relative density fluctuations in the systematic plots.
        adaptative_binning : bool, default=False
            If ``True``, use a binning with same number of objects in each bin, which can be useful to regularise the errors in the histogram.
        nobj_per_bin : int, default=2000
            Relevant only if ``adaptative_binning`` is ``True``. Fix the number of objects in each bin.
        n_bins : int, default=None
            Only relevant if ``adaptative_binning`` is ``False``. Fix the number of bins used in systematic plots.
            If ``None``, use pre-defined parameters set in `systematics.py`.
        cut_fracarea : bool, default=False
            If ``True`` remove queue distribution of the fracarea.
        fracarea_limits : tuple, list, default=None
            If a tuple or list, min and max limits for fracarea.
        save_table : bool
            If true, save in .ecsv format the lines plotted as requiered by the DESI collaboration during the publication process.
        save_table_suffix : str
            If save_table is True, the line will be saved under `f'{save_table_suffix}{sysname}_{label}.ecsv'`
        """
        from .plot import plot_moll
        from .systematics import plot_systematic_from_map

        if save:
            dir_output = os.path.join(self.dataframe.output_dir, self.regressor_name + self.suffix_regressor, 'Fig')
            if not os.path.isdir(dir_output):
                os.mkdir(dir_output)
            logger.info(f"Save density maps and systematic plots in the output directory: {dir_output}")
        else:
            dir_output = None

        with np.errstate(divide='ignore', invalid='ignore'):  # to avoid warning when divide by np.NaN or 0 --> gives np.NaN, ok !
            targets = self.dataframe.targets / (hp.nside2pixarea(self.dataframe.nside, degrees=True) * self.dataframe.fracarea)
        targets[~self.dataframe.footprint('Footprint')] = np.nan

        w = np.zeros(hp.nside2npix(self.dataframe.nside))
        w[self.dataframe.pixels[self.Y_pred > 0]] = 1.0 / self.Y_pred[self.Y_pred > 0]
        targets_without_systematics = targets * w

        with np.errstate(divide='ignore', invalid='ignore'):
            filename = None
            if save: filename = os.path.join(dir_output, 'targets.pdf')
            plot_moll(hp.ud_grade(targets, 64, order_in='NESTED'), min=0, max=max_plot_cart, title='density', show=show, filename=filename, galactic_plane=True, ecliptic_plane=True)
            if save: filename = os.path.join(dir_output, 'targets_without_systematics.pdf')
            plot_moll(hp.ud_grade(targets_without_systematics, 64, order_in='NESTED'), min=0, max=max_plot_cart, title='weighted density', show=show, filename=filename, galactic_plane=True, ecliptic_plane=True)
            map_to_plot = w.copy()
            map_to_plot[map_to_plot == 0] = np.nan
            map_to_plot -= 1
            if save: filename = os.path.join(dir_output, 'systematic_weights.pdf')
            plot_moll(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.2, max=0.2, label='weight - 1', show=show, filename=filename, galactic_plane=True, ecliptic_plane=True)

        plot_systematic_from_map([targets, targets_without_systematics], ['No correction', 'Systematics correction'],
                                 self.dataframe.fracarea, self.dataframe.footprint, self.dataframe.features, self.dataframe.regions,
                                 ax_lim=ax_lim, adaptative_binning=adaptative_binning, nobj_per_bin=nobj_per_bin, n_bins=n_bins,
                                 cut_fracarea=cut_fracarea, limits_fracarea=limits_fracarea,
                                 save_table=save_table, save_table_suffix=save_table_suffix, show=show, save=save, savedir=dir_output)
