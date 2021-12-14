#!/usr/bin/env python
# coding: utf-8

# Author : Edmond CHAUSSIDON (CEA Saclay)


import numpy as np
import healpy as hp

import matplotlib.pyplot as plt

from desitarget.geomask import hp_in_box

from astropy.table import Table
import pandas as pd

import sys

from fct_regressor import _load_targets
from desi_footprint import DR9_footprint

#------------------------------------------------------------------------------#
#Variable (TO CHANGE)
#------------------------------------------------------------------------------#

Nside = 128
release='dr9'
version = 'DA02'
suffixe = '_QSO'
additional_suffixe = '' # '' '_median' '_petit_des'

fracarea_name = 'FRACAREA' # use specific fracarea save in the same place than Data
#fracarea_name = 'FRACAREA_12290' # minimal fracarea !! Maskbit 1, 12, 13 --> CHANGE IF YOU WANT TO USE MORE MASK

use_median = False
use_new_norm = False # use for QSO and dr9 !! (radecbox for north and south)

#footprint info
remove_LMC = False
clear_south = True

use_lrg_region = False
use_elg_region = False
use_elg_region_des = False

if use_elg_region:
    zone_name_list = ['North', 'South_mid', 'South_pole']
elif use_elg_region_des:
    zone_name_list = ['North', 'South', 'South_pole', 'Des_mid']
elif use_lrg_region:
    zone_name_list = ['North', 'South_all']
else:
    zone_name_list = ['North', 'South', 'Des']
zone_name_to_column_name = {'North':'ISNORTH', 'South':'ISSOUTHWITHOUTDES', 'Des':'ISDES', 'South_mid':'ISSOUTHMID', 'South_pole':'ISSOUTHPOLE', 'Des_mid':'ISDESMID', 'South_all':'ISSOUTH'}

print(f"[INFO] WE USE REGION: {zone_name_list}")

#------------------------------------------------------------------------------#
#load data
#------------------------------------------------------------------------------#
df_pixmap = Table.read(f'Data/pixweight-{release}-{Nside}.fits', format='fits').to_pandas()

# Load DR9 Legacy Imaging footprint
DR9 = DR9_footprint(Nside, remove_LMC=remove_LMC, clear_south=clear_south)
footprint = DR9.load_footprint()
if additional_suffixe == '_petit_des':
    footprint[hp_in_box(Nside, [0, 360, -90, -30])] = False

# extract the different region from DR9_footprint
north, south, des = DR9.load_photometry()
_, south_mid, south_pole = DR9.load_elg_region()
des_mid = des & ~south_pole
photometry_footprint = pd.DataFrame({'ISNORTH':north, 'ISSOUTH':south, 'ISDES':des, 'ISSOUTHWITHOUTDES':south&~des, 'ISSOUTHMID':south_mid, 'ISSOUTHPOLE':south_pole, 'ISDESMID':des_mid})[footprint]

#pixweight map selection
sel_columns_pixmap = ['STARDENS', 'EBV',
                      'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
features_pixmap = df_pixmap[sel_columns_pixmap][footprint].copy()

stream_map = pd.DataFrame(np.load(f"Data/sagittarius_stream_{Nside}.npy"), columns=['STREAM'])
stream_map = stream_map[footprint]
features = pd.concat([stream_map, features_pixmap, photometry_footprint], axis=1)

pixels = df_pixmap['HPXPIXEL'][footprint]

if fracarea_name == 'FRACAREA': ## not use fracara from pixmap but load specific fracarea
    fracarea = np.load(f'Data/{version}{suffixe[:4]}_fracarea_{Nside}.npy')[footprint]
else:
    fracarea = df_pixmap[fracarea_name][footprint].values

# load targets pixmap
targets, _, _ = _load_targets(suffixe, release, version, Nside)
targets = targets[footprint]

#------------------------------------------------------------------------------#
# clean data
#------------------------------------------------------------------------------#
# remove pixels with wrong value of targets and with 0 fracarea ...
pix_to_remove = np.isnan(targets) # pixels at the border of the footprint have NaN value and are not completly remove when i take footrpint (ok no problemo here)+ pixels that we mask in function of suffixe :)
print("\nNombre de pixel qui ont une fraction de leur surface observee nulle : {}".format(np.sum(fracarea == 0)))
print("Nombre de ces pixels qui ne font pas partie des pixels a retirer definie plus haut : {}".format(np.sum((fracarea == 0) & ~(pix_to_remove))))
print("OK --> ceux sont des pixels du bords pas de soucis")
pix_to_remove |= (fracarea == 0)

pixels = pixels[~pix_to_remove]
fracarea = fracarea[~pix_to_remove]
features = features[~pix_to_remove]
targets = targets[~pix_to_remove]

#pixels with bad behaviour --> reject for training (cf version notebook pour les histogrames et tout)
keep_to_train = np.ones(targets.size)

## IL faudrait mettre tout ca dans un fichier .yml ...

if Nside == 512:
    pix_to_remove_train = (fracarea < 0.5) | (fracarea > 1.5)
    if suffixe == '_QSO': # on a plus que 260 targets by deg^2 !! (r<23.0) etonnament pas de beaucoup ==> cf histogramme
        print("trop de bruit de poisson --> la correction n'est pas bonne a grande echelle ...")
    elif suffixe == '_ELG':
        pix_to_remove_train += (targets < 10) | (targets > 60)
    elif suffixe == '_ELG_HIP':
        pix_to_remove_train += (targets < 9) | (targets > 50)
    elif suffixe == '_LRG':
        pix_to_remove_train += (targets < 2) | (targets > 30)
    elif suffixe == '_LRG_LOWDENS':
        pix_to_remove_train += (targets < 1) | (targets > 25)
    elif suffixe == '_BGS_ANY':
        pix_to_remove_train += (targets < 5) | (targets > 50)
    elif suffixe == '_BGS_FAINT':
        pix_to_remove_train += (targets < 2) | (targets > 25)
    elif suffixe == '_BGS_BRIGHT':
        pix_to_remove_train += (targets < 2) | (targets > 26)
    else:
        print("[INFO]  ON NE RETIRE PAS DE PIXEL PROBELEMATIQUE POUR L INSTANT selon le target density")
elif Nside == 256:
    pix_to_remove_train = (fracarea < 0.9) | (fracarea > 1.1)
    if suffixe == '_QSO' or suffixe == '_ELG_VLO_newsel': # on a plus que 260 targets by deg^2 !! (r<23.0) etonnament pas de beaucoup ==> cf histogramme
        pix_to_remove_train += (targets < 5) | (targets > 35)
    elif suffixe == '_ELG':
        print("[INFO] ALLER A NSIDE == 512")
    elif suffixe == '_LRG' and version == 'SV3':
        pix_to_remove_train += (targets < 20) | (targets > 85)
    elif suffixe == '_LRG' and version == 'MAIN':
        pix_to_remove_train += (targets < 9) | (targets > 75)
    elif suffixe == '_LRG_LOWDENS':
        pix_to_remove_train += (targets < 10) | (targets > 75)
    elif suffixe == '_BGS_ANY':
        pix_to_remove_train += (targets < 30) | (targets > 140)
    else:
        print("[INFO]  ON NE RETIRE PAS DE PIXEL PROBELEMATIQUE POUR L INSTANT selon le target density")
elif version == 'DA02':
    pix_to_remove_train = ~(fracarea > 0)
    print("[INFO] FOR DA02 on a deja fait la selection lorsque l'on cree les pixmap")
else:
    print("[ATTENTION] AUCUNE SELECTION POUR L'ENTRAINEMENT")

keep_to_train[pix_to_remove_train] = 0

print("\n/!\ I y a {} pixel avec des proprietes anormales pour l'entrainement ce qui represente {:2.2%} du footprint de DESI".format(np.sum(pix_to_remove_train), np.sum(pix_to_remove_train)/pix_to_remove_train.size))

plt.figure(figsize=(8,6))
plt.hist(targets, range=(0,100), bins=100)
plt.savefig(f"Res/Build_features/test_remove_targets_{release}_{Nside}{suffixe}{additional_suffixe}.pdf")
plt.close()

#plt.figure(figsize=(8,6))
#plt.hist(fracarea, range=(0.5, 1.4), bins=100)
#plt.savefig(f"Res/Build_features/test_remove_fracarea_{release}_{Nside}{suffixe}{additional_suffixe}.pdf")
#plt.close()

# On regarde qu'elles sont les pixels qui ont un problème
tmp = np.zeros(hp.nside2npix(Nside))
tmp[pixels[pix_to_remove_train == True]] = 1
plt.figure(figsize=(8,6))
hp.mollview(tmp, rot=120, nest=True, title='strange pixel', cmap='jet')
plt.savefig(f"Res/Build_features/{release}-strange-pixel-{Nside}{suffixe}{additional_suffixe}.pdf")

# build normalized targets
normalized_targets = np.zeros(targets.size)

## FAIRE UNE LOOP PORU QUE TOUT AILELS BIEN EN FONCTION DE LA ZONE

for zone_name in zone_name_list:
    pix_zone = features[zone_name_to_column_name[zone_name]].values
    pix_to_use = pix_zone & (keep_to_train == 1)

    # only conserve pixel in the correct radec box
    if use_new_norm:
        #compute normalization on subpart of the footprint which is not contaminated for the north and the south !
        keep_to_norm = np.zeros(hp.nside2npix(Nside))
        if zone_name == 'North':
            keep_to_norm[hp_in_box(Nside, [120, 240, 32.2, 40], inclusive=True)] = 1
        elif zone_name == 'South':
            keep_to_norm[hp_in_box(Nside, [120, 240, 24, 32.2], inclusive=True)] = 1
        else:
            keep_to_norm = np.ones(hp.nside2npix(Nside))
        keep_to_norm = keep_to_norm[footprint][~pix_to_remove]
        pix_to_use &= keep_to_norm

    # compute the mean only on pixel with "correct" behaviour
    if not use_median:
        mean_targets_density_estimators = np.mean(targets[pix_to_use] / fracarea[pix_to_use])
    else:
        mean_targets_density_estimators = np.median(targets[pix_to_use] / fracarea[pix_to_use])

    #compute normalized_targets every where but we don't care we only use keep_to_train == 1 during the training
    normalized_targets[pix_zone] = targets[pix_zone] / (fracarea[pix_zone]*mean_targets_density_estimators)

    print("\n  ** INFO for ", zone_name)
    print(mean_targets_density_estimators)
    print(normalized_targets[pix_to_use].mean())

# plt.figure(figsize=(8,6))
# plt.hist(normalized_targets[keep_to_train == 1], range=(0,5), bins=100)
# plt.savefig(f"Res/Build_features/test_normalized_targets_{release}_{Nside}{suffixe}{additional_suffixe}.pdf")

print("\nNombre de case 'nan' dans les features : ", features.isnull().sum().sum(), ' --> OK ON VA REMPLIR AVEC LES MOYENNES ...')
print("Nombre de lignes (ie) de pixels qui posent probleme dans features : ", features[features.isnull().T.any().T].index.size)
print("Nombre de case 'nan' dans la densite : ", np.isnan(normalized_targets).sum(), ' --> OUF !')
print("Nombre de case 'nan' dans les pixels : ", pixels.isnull().sum(), ' -- > OUF !')

features = features.fillna(features.mean())

print("Nombre de lignes (ie) de pixels qui posent probleme apres remplissage : ", features[features.isnull().T.any().T].index.size)

data_to_add_1 = pd.DataFrame(normalized_targets, index=pixels, columns=['NORMALIZED_TARGETS'])
data_to_add_2 = pd.DataFrame(keep_to_train, index=pixels, columns=['KEEP_TO_TRAIN'])
pd_total_data = pd.concat([features, data_to_add_1, data_to_add_2, pixels], axis=1)
print(f"\nLe data frame final est sauvegarde a : Data/data_frame_regressor_{release}_{version}_{Nside}{suffixe}{additional_suffixe}.pkl")
pd_total_data.to_pickle(f"DataFrame/data_frame_regressor_{release}_{version}_{Nside}{suffixe}{additional_suffixe}.pkl")

#------------------------------------------------------------------------------#
#build dataframe for plot_systematics (we don't remove any pixels :) ) with desitarget features not needed but this is a sanity check
#------------------------------------------------------------------------------#

if fracarea_name == 'FRACAREA': ## not use fracara from pixmap but load specific fracarea
    fracarea = pd.DataFrame(np.load(f'Data/{version}{suffixe[:4]}_fracarea_{Nside}.npy'), columns=[fracarea_name])
else:
    fracarea = pd.DataFrame()

sel_columns_pixmap = ['FRACAREA_12290', 'STARDENS', 'EBV',
                      'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z', 'PSFDEPTH_W1', 'PSFDEPTH_W2',
                      'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z']
#pixweight map selection
features_total = df_pixmap[sel_columns_pixmap].copy()

stream_map = pd.DataFrame(np.load(f"Data/sagittarius_stream_{Nside}.npy"), columns=['STREAM'])
features_total = pd.concat([fracarea, stream_map, features_total], axis=1)

#fill miss values
# normalement il n'y en a plus
#print("\nATTENTION : Il y a plein de valeur nulle dans la carte de medhi (en dehors du footprint notament)")
#print("On va les fixer a  -1 comme dans tous les autres attributs -> pour qu'on ne fasse pas d'erreur plus tard")
#print("notamment dans plot_systematics :)")
#print("\nnombre de pixel a valeur nulle : \n{}".format(features_total.isnull().sum()))

#for name in features_total.columns:
#    value_to_fill = features_total[name].isnull().values
#    features_total[name][value_to_fill] = -1

#print("\nApres correction")
#print("\nnombre de pixel a valeur nulle : \n{}".format(features_total.isnull().sum()))

## CHANGER CA, CA ne sert plus a rien !!
print("\n[TODO] C4EST VRAIMENT NUL CE QUI CE PASSE ICI, FAIRE SUPER MEGA ATETNTION\nATTENTION LE PIXWEIHGT CHANGE EN FONCTION DES TARGETS ECT ... !!!!!!")
table_to_save = Table.from_pandas(features_total)
table_to_save.write(f'Data/pixweight-total-{release}-{Nside}.fits', overwrite=True)
