import os
import sys

import numpy as np
import healpy as hp
import pandas as pd
from astropy.table import Table

import pickle
from desitarget.io import load_pixweight_recarray
from plot import plot_cart, plot_moll

from fct_regressor import plot_feature_importances, make_regressor_kfold, make_polynomial_regressor, _load_targets,  _load_feature_names
from plot_systematics import plot_systematics

#------------------------------------------------------------------------------#
# TO CHANGE

Nside = 128

release='dr9' # Now always dr9
version = 'DA02' # SV3 or Main
suffixe = '_QSO'
add_suffixe_data_frame = ''
add_suffixe = ''

fracarea_name = 'FRACAREA' #not from pixmap
#fracarea_name = 'FRACAREA_12290' #Maskbit 1, 12, 13 --> juste pour les dessins

use_lrg_region = False
use_elg_region = False
use_elg_region_des = False

#Use Sgr Stream for ELG ?:
use_stream = False #only relevant for the ELG NOW !!
use_stars = True  #only relevany for the QSO NOW !!

# DR9 footprint info:
remove_LMC = False
clear_south = True

use_linear_correction, use_MLP_correction = False, False

if use_elg_region:
    zone_name_list = ['North', 'South_mid', 'South_pole']
elif use_elg_region_des:
    zone_name_list = ['North', 'South', 'South_pole', 'Des_mid']
elif use_lrg_region:
    zone_name_list = ['North', 'South_all']
else:
    zone_name_list = ['North', 'South', 'Des']
zone_name_to_column_name = {'North':'ISNORTH', 'South':'ISSOUTHWITHOUTDES', 'Des':'ISDES', 'South_mid':'ISSOUTHMID', 'South_pole':'ISSOUTHPOLE', 'Des_mid':'ISDESMID', 'South_all':'ISSOUTH'}

min_samples_leaf= {'North':20, 'South':20, 'Des':20, 'South_all':20, 'South_mid':20, 'South_pole':20, 'South_mid_no_des':20, 'Des_mid':20}
nbr_fold = {'North':6, 'South':12, 'Des':6, 'South_all':18, 'South_mid':14, 'South_pole':5, 'Des_mid':3}

#------------------------------------------------------------------------------#
# DO NOT CHANGE
print("\n[INFO] TO REMOVE OR INCLUDE NEW FEATURES GO TO fct_regressor.py and change _load_feature_names (ok pas super optimal mais d'accord :D)\n")
feature_names, feature_names_for_normalization, feature_names_pandas, feature_names_pandas_to_plot = _load_feature_names(suffixe, use_stream=use_stream, use_stars=use_stars)

if use_MLP_correction:
    suffixe_save_name = f'NN_{release}{suffixe}_{Nside}{add_suffixe}'
elif use_linear_correction:
    suffixe_save_name = f'Linear_{release}{suffixe}_{Nside}{add_suffixe}'
else:
    suffixe_save_name = f'RF_{release}{suffixe}_{Nside}{add_suffixe}'
base_directory = f'Res/{version}/{suffixe_save_name}/'

if os.path.isdir(base_directory):
    print(f"\n ERROR {base_directory} ALREADY EXISTS --> Please save it before to remove it :")
    sys.exit()
else:
    os.mkdir(base_directory)
    os.mkdir(os.path.join(base_directory, 'Systematics'))
    print(f"INFO: directory {base_directory} is created")

data_frame = pd.read_pickle(f"DataFrame/data_frame_regressor_{release}_{version}_{Nside}{suffixe}{add_suffixe_data_frame}.pkl")
print(f"\n[LOAD] DataFrame/data_frame_regressor_{release}_{version}_{Nside}{suffixe}{add_suffixe_data_frame}.pkl is loaded...")

targets, max_plot_cart, ax_lim = _load_targets(suffixe, release, version, Nside)

Npix, pixel_area = hp.nside2npix(Nside), hp.nside2pixarea(Nside, degrees=True)

if Nside == 256:
    regulator = 2e6
elif Nside == 512:
    regulator = 8*2e6
elif Nside == 128:
    regulator = 1/8*2e6

#------------------------------------------------------------------------------#
pixels = data_frame['HPXPIXEL'].values
norm_targets = data_frame['NORMALIZED_TARGETS'].values
keep_to_train = data_frame['KEEP_TO_TRAIN'].values
features = data_frame[feature_names]

footprint = np.zeros(hp.nside2npix(Nside))
footprint[pixels] = 1

F = np.zeros(pixels.size)
fold_index = dict()

for zone_name in zone_name_list:
    os.mkdir(os.path.join(base_directory, zone_name))

    zone = data_frame[zone_name_to_column_name[zone_name]]
    nbr_fold_zone, min_samples_leaf_zone = nbr_fold[zone_name], min_samples_leaf[zone_name]

    print(f"\n######################\n    {zone_name} : \n")
    X = features[zone]
    Y = norm_targets[zone]
    pixels_zone = pixels[zone]
    keep_to_train_zone = keep_to_train[zone]
    print(f"Taille de l'echantillon {zone_name} : {keep_to_train_zone.sum()}\nTaille de l'echantillon total : {keep_to_train.sum()}\nFraction de l'entrainement : {keep_to_train_zone.sum()/keep_to_train.sum():.2%}\n")
    if not use_linear_correction:
        F[zone], fold_index[zone_name] = make_regressor_kfold(X, Y, keep_to_train_zone, pixels_zone, Nside, feature_names_for_normalization, feature_names_pandas,
                                                         use_MLP_correction, nbr_fold_zone, min_samples_leaf_zone, dir_to_save=base_directory+f'{zone_name}/', plot_accuracy=False)
    else:
        F[zone] = make_polynomial_regressor(X, Y, keep_to_train_zone, regulator, feature_names_for_normalization)

if not (use_MLP_correction or use_linear_correction or add_suffixe[:5] == '_zone'):
    print("\n    * On fait un beau dessin pour feature importances (cf notebook for more detail plots)...")
    plot_feature_importances(base_directory, feature_names_pandas, feature_names_pandas_to_plot, zone_name_list, nbr_fold)

print("\n######################\nCorrection des systematiques ...\n")
#ne pas oublier le fracarea !! --> on en sauvegarde jamais avec le fracarea
fracarea = load_pixweight_recarray("Data/pixweight-total-{}-{}.fits".format(release, Nside), nside=Nside)[fracarea_name]
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
plot_cart(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.3, max=0.3, label='weight - 1',  show=False, savename=base_directory+'Systematics/weight.pdf')

plot_moll(hp.ud_grade(targets, 64, order_in='NESTED'), min=0, max=max_plot_cart, show=False, savename=base_directory + 'Systematics/targerts_projected.pdf', galactic_plane=True, ecliptic_plane=True)
plot_moll(hp.ud_grade(targets_without_systematics, 64, order_in='NESTED'), min=0, max=max_plot_cart,  show=False, savename=base_directory + 'Systematics/targets_without_systematics_projected.pdf', galactic_plane=True, ecliptic_plane=True)
plot_moll(hp.ud_grade(map_to_plot, 64, order_in='NESTED'), min=-0.3, max=0.3, label='weight - 1',  show=False, savename=base_directory+'Systematics/weight_projected.pdf', galactic_plane=True, ecliptic_plane=True)

plot_systematics(dir_to_save=base_directory, zone_to_plot=zone_name_list, suffixe=suffixe_save_name, suffixe_stars=suffixe,
                 release=release, version=version, Nside=Nside, fracarea_name=fracarea_name, ax_lim=ax_lim, remove_LMC=remove_LMC, clear_south=clear_south, nbins=15)
