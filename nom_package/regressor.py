# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import logging
logger = logging.getLogger("regressor")

import os, sys, time

import numpy as np
import healpy as hp
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
#plt.style.use('~/Software/desi_ec/ec_style.mplstyle')

from utils import deep_update, regression_least_square, zone_name_to_column_name

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, load


def _load_feature_names(tracer, use_stream=None, use_stars=None):
    """
    Load the default feature set to mitigate the systematic effects

    Parameters:
    ----------
    tracer: str
        the tracer name e.g. QSO / ELG / LRG / BGS
    use_stream: bool
        Use Sgr. Stream as template f--> default = False for all and True for QSO
    use_stars: bool
        Use stardens as template --> default = True
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


def _load_rf_hyperparameters(updated_param=None, n_jobs=6):
    param = dict()
    min_samples_leaf = {'North':20, 'South':20, 'Des':20, 'South_all':20, 'South_mid':20, 'South_pole':20, 'South_mid_no_des':20, 'Des_mid':20}
    for key in min_samples_leaf:
        param[key] = {'n_estimators':200, 'min_samples_leaf':min_samples_leaf[key], 'max_depth':None, 'max_leaf_nodes':None, 'n_jobs':n_jobs}
    if not updated_param is None:
        param = deep_update(param, updated_param)
    return param


def _load_mlp_hyperparameters(updated_param=None):
    param = {'activation': 'logistic', 'batch_size': 1000, 'hidden_layer_sizes': (10, 8),
             'max_iter': 6000, 'n_iter_no_change': 100, 'random_state': 5, 'solver': 'adam', 'tol': 1e-5}
    if not updated_param is None:
        param = deep_update(param, updated_param)
    return param


def _load_linear_hyperparameters(updated_param=None):
    param = dict()
    if not updated_param is None:
        param = deep_update(param, updated_param)
    return param


def _load_nfold(updated_param=None):
    param = {'North':6, 'South':12, 'Des':6, 'South_all':18, 'South_mid':14, 'South_pole':5, 'Des_mid':3}
    if not updated_param is None:
        param = deep_update(param, updated_param)
    return param


class Regressor(object):
    """
    Implementation of the Systematic Correction based on template fitting regression
    """

    def __init__(self, dataframe, engine, feature_names=None, use_Kfold=True,
                 updated_param_rf=None, updated_param_mlp=None, updated_param_linear=None, updated_nfold=None,
                 overwrite_regression=False, save_regressor=False, n_jobs=6):
        """
        Initialize :class:`Regressor`

        Parameters
        ----------

        """
        self.dataframe = dataframe
        self.engine = engine
        if feature_names is None:
            self.feature_names = _load_feature_names(dataframe.tracer)
        else:
            self.feature_names = feature_names

        # set up the parameter for the considered regressor
        if use_Kfold:
            self.use_Kfold = use_Kfold
            if self.engine == 'RF':
                self.param_regressor = _load_rf_hyperparameters(updated_param_rf, n_jobs)
            elif self.engine == 'NN':
                self.param_regressor = _load_mlp_hyperparameters(updated_param_mlp)
            elif self.engine == 'LINEAR':
                self.param_regressor = _load_linear_hyperparameters(updated_param_linear)
        else:
            logging.warning("DO NOT USE K-FOLD --> ONLY REGRESSION AVAILABLE IS LINEAR WITH IMINUIT")
            self.use_Kfold = use_Kfold
            self.engine = 'Linear'
            self.param_regressor = {'regulator':2e6*(self.dataframe.Nside/256)**3}

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

        self.save_regressor = save_regressor
        self.nfold = _load_nfold(updated_nfold)

        ## IF not self.dataframe.output is None --> save figure and info !!
        if not self.dataframe.output is None:
            # create the corresponding output folder --> put here since self.engine can be update with use_Kfold = False
            if os.path.isdir(os.path.join(self.dataframe.output, self.engine)):
                if not overwrite_regression:
                    logger.error(f"{os.path.join(self.dataframe.output, self.engine)} already exist and overwrite_regression is set as {overwrite_regression}")
                    sys.exit()
                else:
                    logger.warning(f"OVERWRITE {os.path.join(self.dataframe.output, self.engine)}")
                    logger.warning(f"PLEASE REMOVE THE OUPUT FOLDER TO HAVE CLEAN OUTPUT: rm -rf {os.path.join(self.dataframe.output, self.engine)}")
            else:
                logger.info(f"The output folder {os.path.join(self.dataframe.output, self.engine)} is created")
                os.mkdir(os.path.join(self.dataframe.output, self.engine))

    def make_regression(self):
        """
        Compute systematic weight with the selected engine method and choosen hyperparameters.
        TO DO
        """
        # To store the result of the regression (ie) Y_pred
        F = np.zeros(self.dataframe.pixels.size)
        fold_index = dict()

        for zone_name in self.dataframe.region:
            if not self.dataframe.output is None:
                save_info = True
                save_dir = os.path.join(self.dataframe.output, self.engine, zone_name)
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
            logger.info(f"    --> use Kfold training ? {self.use_Kfold}")
            logger.info(f"    --> Engine: {self.engine} with params: {self.param_regressor[zone_name]}")

            if self.use_Kfold:
                if self.engine == 'NN':
                    regressor = MLPRegressor(**self.param_regressor[zone_name])
                    normalized_data = True
                elif self.engine == 'RF':
                    regressor = RandomForestRegressor(**self.param_regressor[zone_name])
                    normalized_data = False
                elif self.engine == 'LINEAR':
                    regressor = LinearRegression(**self.param_regressor[zone_name])
                    normalized_data = True

                F[zone], fold_index[zone_name] = Regressor.make_regressor_kfold(regressor, self.nfold[zone_name], X, Y, keep_to_train_zone, normalized_data, pixels_zone, self.feature_names_to_normalize, self.dataframe.Nside, self.feature_names,
                                                                                save_regressor=self.save_regressor, save_info=save_info, save_dir=save_dir)
            else:
                # do not use k-fold training --> standard linear regression with iminuit
                F[zone] = Regressor.make_polynomial_regressor(X, Y, keep_to_train_zone, self.feature_names_to_normalize, self.param_regressor)
                fold_index = None

        # save evalutaion and fold_index
        self.F = F # Y_pred for every entry in each zone
        self.fold_index = fold_index # fold_index --> les pixels de chaque fold dans chaque zone ! --> usefull to save if we want to reapply the regressor

        if save_info and (not fold_index is None):
            logger.info("        * Sauvegarde des index des kfold sous forme d'un dictionnaire...")
            dump(self.fold_index, os.path.join(self.dataframe.output, self.engine, f'kfold_index.joblib'))


    @staticmethod
    def build_kfold(Nside, kfold, group, pixels, title='', savename=None):
        """
        TODO
        """
        map = np.zeros(hp.nside2npix(Nside))
        index = []
        i = 1
        for index_train, index_test in kfold.split(pixels, groups=group):
            index += [index_test]
            map[pixels[index_test]] = i
            i = i+1
        map[map == 0] = np.NaN

        if not savename is None:
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

        return index


    @staticmethod
    def make_regressor_kfold(regressor, nfold, X, Y, keep_to_train, normalized_data, pixels, feature_names_to_normalize, Nside, feature_names, save_regressor=False, save_info=False, save_dir=''):
        """
        TO DO
        """

        kfold = GroupKFold(n_splits=nfold)
        size_group = 1000  * (Nside / 256)**2
        group = [i//size_group for i in range(pixels.size)]
        logger.info(f"    --> We use: {kfold} Kfold with group_size={size_group}")

        if save_info:
            savename = os.path.join(save_dir, 'kfold_repartition.png')
        else:
            savename = None
        index = Regressor.build_kfold(Nside, kfold, group, pixels, title='{}-Fold repartition'.format(nfold), savename=savename)

        Y_pred = np.zeros(pixels.size)
        X.reset_index(drop=True, inplace=True)

        logger.info("    --> Train and eval for the fold :")
        start = time.time()
        for i in range(nfold):
            logger.info(f"        * {i}")
            fold_index = index[i]
            keep_to_train_fold = np.delete(keep_to_train, fold_index)
            logger.info(f"          --> There are {np.sum(keep_to_train_fold == 1)} pixels to train fold {i} which contains {np.sum(keep_to_train == 1) - np.sum(keep_to_train_fold == 1)} pixels (kept for the global training)")

            if normalized_data:
                logger.info("          --> We normalize and center all features (execpt the STREAM) on the training footprint fot his fold training ! (all )")
                X_fold = X.copy()
                X_fold[feature_names_to_normalize] = (X[feature_names_to_normalize] - X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean())/X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std()
                logger.info(f"          --> Mean of Mean and Std on all features : {X_fold[feature_names_to_normalize].mean().mean()} -- {X_fold[feature_names_to_normalize].std().mean()}")
                logger.info(f"          --> Mean of Mean and Std on the fold-training features : {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean().mean()} -- {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std().mean()}\n")
            else:
                logger.info("          --> We do NOT normalize feature in the training set --> not NEEDED")
                X_fold = X.copy()

            X_train, Y_train = X_fold.drop(fold_index)[keep_to_train_fold == 1], np.delete(Y, fold_index)[keep_to_train_fold == 1]
            logger.info("          --> The training is done with sample_weight=1/np.sqrt(Y_train)")
            regressor.fit(X_train, Y_train, sample_weight=1/np.sqrt(Y_train))

            Y_pred_fold = np.zeros(fold_index.size)
            Y_pred_fold = regressor.predict(X_fold.iloc[fold_index])
            Y_pred[fold_index] = Y_pred_fold

            # Save regressor
            if save_regressor:
                dump(regressor, os.path.join(save_dir, f'regressor_fold_{i}.joblib'))

            if save_info:
                #use only reliable pixel (ie) keep_to_train == 1 also in the fold !
                Regressor.plot_efficiency(Y[fold_index], Y_pred_fold, pixels[fold_index], keep_to_train[fold_index], os.path.join(save_dir, f"kfold_efficiency_fold_{i}.png"))

                #for more complex plot as importance feature ect ..--> save regressor and
                if os.path.basename(os.path.dirname(save_dir)) == 'RF':
                    Regressor.plot_importance_feature(regressor, feature_names, os.path.join(save_dir, f"feature_importance_fold_{i}.png"))


        logger.info("    --> Done in: {:.3f} s".format(time.time() - start))
        return Y_pred, index


    @staticmethod
    def make_polynomial_regressor(X, Y, keep_to_train, feature_names_to_normalize, param_regressor):
        """
        TO DO
        """
        def model(x, *par):
            return par[0]*np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

        nbr_features = X.shape[1]
        logger.info(f"[TEST] Number of features used : {nbr_features}")
        nbr_params = nbr_features + 1

        logger.info(f"            ** Taille de l'échantillon (non nan value): {np.sum(Y>0)}")
        logger.info(f"            ** Information sur normalized targets : Mean = {np.nanmean(Y)} and Std = {np.nanstd(Y)}")
        logger.info("[WARNING] We normalize and center features on the training footprint (don't forget to normalized also the non training footprint)")
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


    def build_w_sys_map(self, return_map=True, savemap=True, savedir=None):
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

        w = np.zeros(hp.nside2npix(self.dataframe.Nside))*np.NaN
        w[self.dataframe.pixels] = 1.0/self.F

        if savemap:
            if savedir is None:
                savedir = os.path.join(self.dataframe.output, self.engine)
            filename_weight_save = os.path.join(savedir, f'{self.dataframe.version}_{self.dataframe.tracer}_imaging_weight_{self.dataframe.Nside}.npy')
            logger.info(f"Save photometric weight in a healpix map with {self.dataframe.Nside} here: {filename_weight_save}")
            np.save(filename_weight_save, w)

        if return_map:
            return w


    @staticmethod
    def plot_efficiency(Y, Y_pred, pixels, keep_to_train, path_to_save):
        """
        TO DO = Trace la precision de notre regression, on l'applique sur le jeu d'entrainement ...\n
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
        Plot the giny importance feature for regressor.
        """

        feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names, columns=['feature importance']).sort_values('feature importance', ascending=False)
        feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names)
        feature_all = pd.melt(feature_all, var_name='feature name', value_name='values')

        fig = plt.figure(figsize=(10, 6))
        ax = plt.gca()
        sns.swarmplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], alpha=0.7, size=2, color=".2", palette=sns.color_palette("husl", 8))
        sns.boxplot(ax=ax, x="feature name", y="values", data=feature_all, order=feature_importance.index[:max_num_feature], fliersize=0.6, palette=sns.color_palette("husl", 8), linewidth=0.6, showmeans=False, meanline=True, meanprops=dict(linestyle=':', linewidth=1.5, color='dimgrey'))
        ax.set_xticklabels(feature_importance.index[:max_num_feature], rotation=15, ha='center')
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.tight_layout()
        plt.savefig(path_to_save)


    def plot_maps_and_systematics(self, max_plot_cart=400, ax_lim=0.2, adaptative_binning=False, nobjects_by_bins=2000, n_bins=None, cut_fracarea=True, min_fracarea=0.9, max_fracarea=1.1,):
        """
        Make plot to check and validate the regression.
        the result are saved in the corresponding outpur directory
        """

        from plot import plot_moll
        from systematics import plot_systematic_from_map

        dir_output = os.path.join(self.dataframe.output, self.engine, 'Fig')
        if not os.path.isdir(dir_output):
            os.mkdir(dir_output)
        logger.info(f"Save density maps and systematic plots in the output directory: {dir_output}")

        targets = self.dataframe.targets / (hp.nside2pixarea(self.dataframe.Nside, degrees=True)*self.dataframe.fracarea)
        targets[~self.dataframe.footprint['FOOTPRINT']] = np.NaN

        w = np.zeros(hp.nside2npix(self.dataframe.Nside))
        w[self.dataframe.pixels] = 1.0/self.F
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



        #                     if not use_neural_network:
        #     print("                         *** On sauvegarde importance feature (gini)...")
        #     df_feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names_pandas, columns=['feature importance']).sort_values('feature importance', ascending=False).to_pickle(dir_to_save+f"feature_importance_fold_{i}.pkl")
        #     df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names_pandas)
        #     df_feature_long = pd.melt(df_feature_all, var_name='feature name', value_name='values').to_pickle(dir_to_save+f"feature_importance_all_trees_fold_{i}.pkl")
        #
        # if compute_permutation_importance:
        #     ## The permutation feature importance is defined to be the decrease in a model score
        #     ## when a single feature value is randomly shuffled. This procedure breaks the relationship
        #     ## between the feature and the target, thus the drop in the model score is indicative of
        #     ## how much the model depends on the feature.
        #     print("                         *** On sauvegarde permutation importance feature ...")
        #     permut_importance = permutation_importance(regressor, X_fold.iloc[fold_index], Y[fold_index], n_repeats=15, random_state=4)
        #
        #     a_file = open(dir_to_save+f"permutation_importance_fold_{i}.pkl", "wb")
        #     pickle.dump(permut_importance, a_file)
        #     a_file.close()
        #
        #     fig, ax = plt.subplots()
        #     #sorted_idx = permut_importance.importances_mean.argsort()
        #     ax.boxplot(permut_importance.importances.T,
        #                vert=False, labels=feature_names_pandas_to_plot)
        #     ax.set_title(f"Permutation Importance for Fold {i}")
        #     ax.set_ylabel("Features")
        #     fig.tight_layout()
        #     plt.savefig(dir_to_save+f"permutation_importance_fold_{i}.png")
        #     plt.close()
        #
        #     from sklearn.inspection import permutation_importance
