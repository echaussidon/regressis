# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import logging
logger = logging.getLogger("regressor")

import os, sys, time

import numpy as np
import healpy as hp
import pandas as pd

import matplotlib.pyplot as plt
#plt.style.use('~/Software/desi_ec/ec_style.mplstyle')

from utils import deep_update, regression_least_square

from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from joblib import dump, load

## mettre commentaire cii
zone_name_to_column_name = {'North':'ISNORTH', 'South':'ISSOUTHWITHOUTDES', 'Des':'ISDES',
                            'South_mid':'ISSOUTHMID', 'South_pole':'ISSOUTHPOLE',
                            'Des_mid':'ISDESMID', 'South_all':'ISSOUTH'}


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

    def __init__(self, dataframe, engine, overwrite_regression=False, feature_names=None, use_Kfold=True,
                 updated_param_rf=None, updated_param_mlp=None, updated_param_linear=None, updated_nfold=None, n_jobs=6):
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
                print("ccccc")
                self.feature_names_to_normalize = self.feature_names.copy()
        else:
            self.feature_names_to_normalize = None

        self.nfold = _load_nfold(updated_nfold)

        # create the corresponding output folder --> put here since self.engine can be update with use_Kfold = False
        if os.path.isdir(os.path.join(self.dataframe.output, self.engine)):
            if not overwrite_regression:
                logger.error(f"{os.path.join(self.dataframe.output, self.engine)} already exist and overwrite_regression is set as {overwrite_regression}")
                sys.exit()
            else:
                logger.warning(f"OVERWRITE {os.path.join(self.dataframe.output, self.engine)}")
                logger.warning(f"PLEASE REMOVE THE OUPUT FOLDER TO HAVE CLEAN OUTPUT:\nrm -rf {os.path.join(self.dataframe.output, self.engine)}")
        else:
            logger.info(f"The output folder {os.path.join(self.dataframe.output, self.engine)} is created")
            os.mkdir(os.path.join(self.dataframe.output, self.engine))


    def make_regression(self, save_w_sys_map=False):
        """
        Compute systematic weight with the selected engine method and choosen hyperparameters.
        TO DO
        """

        pixels = self.dataframe.data_regressor['HPXPIXEL'].values
        norm_targets = self.dataframe.data_regressor['NORMALIZED_TARGETS'].values
        keep_to_train = self.dataframe.data_regressor['KEEP_TO_TRAIN'].values
        features = self.dataframe.data_regressor[self.feature_names]

        footprint = np.zeros(hp.nside2npix(self.dataframe.Nside)) ## IL FAUDRA CHANGER CA lrosuqe l'on travaillera avec les petals ...
        footprint[pixels] = 1

        F = np.zeros(pixels.size)
        fold_index = dict()

        for zone_name in self.dataframe.region:
            if not os.path.isdir(os.path.join(self.dataframe.output, self.engine, zone_name)):
                os.mkdir(os.path.join(self.dataframe.output, self.engine, zone_name))

            zone = self.dataframe.data_regressor[zone_name_to_column_name[zone_name]].values ## mask array

            print(f"\n######################\n    {zone_name} : \n")
            X = features[zone]
            Y = norm_targets[zone]
            keep_to_train_zone = keep_to_train[zone]
            pixels_zone = pixels[zone]
            print(f"Sample size {zone_name}: {keep_to_train_zone.sum()}\nTotal Sample Size: {keep_to_train.sum()}\nTraining Fraction: {keep_to_train_zone.sum()/keep_to_train.sum():.2%}\n")
            if self.use_Kfold: ##demander linear without kfold
                F[zone], fold_index[zone_name] = Regressor.make_regressor_kfold(self.engine, self.nfold[zone_name], self.param_regressor[zone_name],
                                                                                X, Y, keep_to_train_zone, pixels_zone, self.feature_names_to_normalize, self.dataframe.Nside,
                                                                                os.path.join(self.dataframe.output, self.engine, zone_name), plot_accuracy=True)
            else:
                F[zone] = Regressor.make_polynomial_regressor(X, Y, keep_to_train_zone, self.feature_names_to_normalize, self.param_regressor)

        ## F ici c'est le nombre de data qu'il y avait en entrée (#petal_pixel ou de healpix pixel --> ne change rien)

        self.plot_maps_and_systematics(save_w_sys_map)



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

        if not savename is None:
            plt.savefig(savename)
        plt.close()

        return index


    @staticmethod
    def make_regressor_kfold(engine, nfold, param_regressor, X, Y, keep_to_train, pixels, feature_names_to_normalize, Nside, dir_to_save, plot_accuracy=False):
        """
        TO DO
        """
        if engine == 'NN':
            print(f"            ** USE NEURAL NETWORK")
            print(f"                        *** Dict ini : {param_regressor}")
            regressor = MLPRegressor(**param_regressor)
        elif engine == 'RF':
            print(f"            ** USE RANDOM FOREST")
            print(f"                        *** Dict ini : {param_regressor}")
            regressor = RandomForestRegressor(**param_regressor)
        elif engine == 'LINEAR':
            print(f"            ** USE LINEAR")
            print(f"                        *** Dict ini : {param_regressor}")
            regressor = LinearRegression(**param_regressor)

        kfold = GroupKFold(n_splits=nfold)
        size_group = 1000  * (Nside / 256)**2
        group = [i//size_group for i in range(pixels.size)]

        print("\nTrace les differents k-fold qui vont etre utilises ...")
        print("    * On utilise : {} avec un group_size = {}".format(kfold, size_group))
        index = Regressor.build_kfold(Nside, kfold, group, pixels, title='{}-Fold repartition'.format(nfold), savename=os.path.join(dir_to_save,'kfold_repartition.png'))

        Y_pred = np.zeros(pixels.size)
        X.reset_index(drop=True, inplace=True)

        print("\nPrediction pour le Fold :")
        start = time.time()
        for i in range(nfold):
            print("     * {}".format(i))
            fold_index = index[i]
            keep_to_train_fold = np.delete(keep_to_train, fold_index)
            print(f"[INFO] There are {np.sum(keep_to_train_fold == 1)} pixels to train fold {i} which contains {np.sum(keep_to_train == 1) - np.sum(keep_to_train_fold == 1)} pixels (kept for the global training)")

            if engine == 'NN' or engine == 'LINEAR':
                print("                         *** On normalise et recentre le jeu d'entrainement ...")
                print("[WARNING :] We normalize and center features on the training footprint fot his fold training !")
                print("[WARNING :] Treat only features which are not the Sgr. stream (already normalized) !")
                X_fold = X.copy()
                X_fold[feature_names_to_normalize] = (X[feature_names_to_normalize] - X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean())/X[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std()
                print(f"[TEST :] Mean of Mean and Std on all features : {X_fold[feature_names_to_normalize].mean().mean()} -- {X_fold[feature_names_to_normalize].std().mean()}")
                print(f"[TEST :] Mean of Mean and Std on the fold-training features : {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].mean().mean()} -- {X_fold[feature_names_to_normalize].drop(fold_index)[keep_to_train_fold == 1].std().mean()}\n")
            else:
                print("                         *** On NE normalise et recentre PAS le jeu d'entrainement ...")
                X_fold = X.copy()

            X_train, Y_train = X_fold.drop(fold_index)[keep_to_train_fold == 1], np.delete(Y, fold_index)[keep_to_train_fold == 1]
            logger.info("The training is done with sample_weight=1/np.sqrt(Y_train)")
            regressor.fit(X_train, Y_train, sample_weight=1/np.sqrt(Y_train))

            Y_pred_fold = np.zeros(fold_index.size)
            Y_pred_fold = regressor.predict(X_fold.iloc[fold_index])
            Y_pred[fold_index] = Y_pred_fold

            # Save regressor
            dump(regressor, os.path.join(dir_to_save, f'regressor_fold_{i}.joblib'))

            #use only reliable pixel (ie) keep_to_train == 1 also in the fold !
            Regressor.plot_efficiency(Y[fold_index], Y_pred_fold, pixels[fold_index], keep_to_train[fold_index], dir_to_save, i)

        logger.info("    * Fait en : {:.3f} s".format(time.time() - start))
        return Y_pred, index


    @staticmethod
    def make_polynomial_regressor(X, Y, keep_to_train, feature_names_to_normalize, param_regressor):
        """
        TO DO
        """
        def model(x, *par):
            return par[0]*np.ones(x.shape[0]) + np.array(par[1:]).dot(x.T)

        nbr_features = X.shape[1]
        print(f"[TEST] Number of features used : {nbr_features}")
        nbr_params = nbr_features + 1

        print(f"            ** Taille de l'échantillon (non nan value): {np.sum(Y>0)}")
        print(f"            ** Information sur normalized targets : Mean = {np.nanmean(Y)} and Std = {np.nanstd(Y)}")
        print("[WARNING] We normalize and center features on the training footprint (don't forget to normalized also the non training footprint)")
        print(feature_names_to_normalize)
        X.loc[:, feature_names_to_normalize] = (X[feature_names_to_normalize] - X[feature_names_to_normalize][keep_to_train == 1].mean())/X[feature_names_to_normalize][keep_to_train == 1].std()
        X_train, Y_train = X[keep_to_train == 1], Y[keep_to_train == 1]
        print(f"[TEST] Mean of Mean and Std training features (should be 0, 1): {X_train[feature_names_to_normalize].mean().mean()} -- {X_train[feature_names_to_normalize].std().mean()}\n")

        dict_ini = {f'a{i}': 0 if i==0 else 0 for i in range(0, nbr_params)}
        dict_ini.update({f'error_a{i}': 0.001 for i in range(0, nbr_params)})
        dict_ini.update({f'limit_a{i}': (-1, 2) if i==0 else (-3,3) for i in range(0, nbr_params)})
        dict_ini.update({'errordef': 1}) #for leastsquare
        Y_cov_inv = np.diag(1/np.sqrt(Y_train))

        param = regression_least_square(model, param_regressor['regulator'], X_train, Y_train, Y_cov_inv, nbr_params, **dict_ini)

        print(f"[TEST] Mean of systematics_correction : {model(X_train, *param).mean()} \n")

        return model(X, *param)


    @staticmethod
    def plot_efficiency(Y, Y_pred, pixels, keep_to_train, dir_to_save, fold_index):
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
        plt.savefig(os.path.join(dir_to_save, f"kfold_efficiency_fold_{fold_index}.png"))
        plt.close()


    @staticmethod
    def plot_feature_importances(path, feature_names_pandas, feature_names_pandas_to_plot, zone, nbr_fold):
        """
        TO DO
        """

        df_feature_importance = pd.DataFrame(regressor.feature_importances_, index=feature_names_pandas, columns=['feature importance']).sort_values('feature importance', ascending=False).to_pickle(dir_to_save+f"feature_importance_fold_{i}.pkl")
        #df_feature_all = pd.DataFrame([tree.feature_importances_ for tree in regressor.estimators_], columns=feature_names_pandas)
        #df_feature_long = pd.melt(df_feature_all, var_name='feature name', value_name='values').to_pickle(dir_to_save+f"feature_importance_all_trees_fold_{i}.pkl")


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
        plt.savefig(path + 'Feature_importances.png')
        plt.close()


    def plot_maps_and_systematics(self, save_w_sys_map=False):
        """
        Make plot to check and validate the regression.
        the result are saved in the corresponding outpur directory

        """

#        if not (use_MLP_correction or use_linear_correction or add_suffixe[:5] == '_zone'):
#            print("\n    * On fait un beau dessin pour feature importances (cf notebook for more detail plots)...")
#            plot_feature_importances(base_directory, feature_names_pandas, feature_names_pandas_to_plot, zone_name_list, nbr_fold)

        print("\n######################\nCorrection des systematiques ...\n")
        #ne pas oublier le fracarea !! --> on en sauvegarde jamais avec le fracarea
        fracarea = self.dataframe['FRACAREA'] -->
        targets = targets / (pixel_area*fracarea)
        targets[~(targets >= 0)] = 0 #pour remettre a zero les pixels non obeserves.
        targets[footprint == 0] = np.NaN

        w = np.zeros(Npix)
        w[pixels] = 1.0/F
        targets_without_systematics = targets*w

        filename_targets_save = base_directory + "targets_without_systematics_pixel_{}_{}{}.npy".format(release, Nside, suffixe_save_name)
        filename_weight_save = base_directory + "systematics_correction{}.npy".format(suffixe_save_name)
        print("\n######################\nSauvegarde de la densite corrigee des effets des systematiques dans : {}".format(filename_targets_save))
        print("    * Sauvegarde des poids pour corriger les effets des systematiques dans : {}".format(filename_weight_save))
        np.save(filename_targets_save, targets_without_systematics*(pixel_area*fracarea))
        np.save(filename_weight_save, w)

        print("    * Sauvegarde des index des kfold sous forme d'un dictionnaire...")
        outfile = open(base_directory+'fold_index.pkl','wb')
        pickle.dump(fold_index, outfile)
        outfile.close()

        print("\n######################\nCalcul et affichage des systematiques ...\n")
        print("[WARNING :] MAP ARE UD_GRADE TO 64 TO SEE SOMETHING!")

        plot_cart(hp.ud_grade(targets, 64, order_in='NESTED'), min=0, max=max_plot_cart, show=False, savename=base_directory + 'Systematics/targerts.pdf')
        plot_cart(hp.ud_grade(targets_without_systematics, 64, order_in='NESTED'), min=0, max=max_plot_cart,  show=False, savename=base_directory + 'Systematics/targets_without_systematics.pdf')
        map_to_plot = w.copy()
        map_to_plot[map_to_plot == 0] = np.NaN
        map_to_plot = map_to_plot - 1
        plot_cart(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.2, max=0.2, label='weight - 1',  show=False, savename=base_directory+'Systematics/weight.pdf')

        plot_moll(hp.ud_grade(targets, 64, order_in='NESTED'), min=0, max=max_plot_cart, show=False, savename=base_directory + 'Systematics/targerts_projected.pdf', galactic_plane=True, ecliptic_plane=True)
        plot_moll(hp.ud_grade(targets_without_systematics, 64, order_in='NESTED'), min=0, max=max_plot_cart,  show=False, savename=base_directory + 'Systematics/targets_without_systematics_projected.pdf', galactic_plane=True, ecliptic_plane=True)
        plot_moll(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.3, max=0.3, label='weight - 1',  show=False, savename=base_directory+'Systematics/weight_projected.pdf', galactic_plane=True, ecliptic_plane=True)

        plot_systematics(dir_to_save=base_directory, zone_to_plot=zone_name_list, suffixe=suffixe_save_name, suffixe_stars=suffixe,
                         release=release, version=version, Nside=Nside, fracarea_name=fracarea_name, ax_lim=ax_lim, remove_LMC=remove_LMC, clear_south=clear_south, nbins=15)
