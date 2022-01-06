# Compute photometric weight for DESI

This code was used to compute the photometric weights for the QSO target selection in Chaussidon et. al 2022 (https://arxiv.org/abs/2108.03640)

## 1. Collect features:

First run these commands to collect the DR9 features and the Sgr Stream. By default the outputs are saved in the data folder. You need to run it on *NERSC*.

 * `collect_dr9_features.py`: It take a while to have enough randoms to compute the default fracarea (FRACAREA_12290). This is only relevant for SV3/MAIN targets. You can set nbr_randoms = 1 to go faster.

 * `collect_sgr_stream.py`: Build the STREAM feature. If Sgr_members_L120_150_GaiaDR2.csv is not in the data folder, you can download it here: https://sites.google.com/fqa.ub.edu/tantoja/research/sagittarius.

## 2. Collect data:

* `collect_desi_targets.py`: Collect standard DESI targets for SV3 and MAIN selection and save the maps at nside=256/512 in data folder.

* `collect_DA02.py`: Collect data from DA02 for BGS_ANY, LRG, ELG, QSO and compute the corresponding fracarea. outputs are saved in data folder.

## 3. Compute weights:

* `generate_weight_SV3.py`: Compute photometric weights for the standard SV3 DESI targets. The three photometric regions are treated independently. Need `collect_desi_targets.py` first.

* `generate_weight_MAIN.py`: Compute photometric weights for the standard MAIN DESI targets. The three photometric regions are treated independently. Need `collect_desi_targets.py` first.

* `generate_weight_DA02.py`: Compute photometric weights for the clustering catalog of DA02. The three photometric regions are treated independently even if there is almost no target in DES. Need `collect_DA02.py` first.
