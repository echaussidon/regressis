.. _developer-changes:

Change Log
==========

Main (2024-01-11)
-----------------

* Add script to generate new external maps including the DIFF_EBV from rongpu and HI maps

* update dataframe.py to collect these new maps by default

* Add correct parameters for HI in systematics.py

(2023-08-30)
-----------------

* Can select different criterion in RFRegressor.

* Can provide one set of features for each regions (typically useful for North vs. South).

* Add automatic downgrading when there are not enough data in pixels.

* Add stuff to read and convert Rongpu EBV maps.

* Add option (default=True) which normalize the wsys.maps during the call.

(2022-06-08)
-----------------

* Add possibility to use an external pixweight when loading the features. Set also the features that we want to plot in the systematic plots automatically --> ideal to test the dependence as a function of a feature after the correction.

* Use now nside from systematic weight class when computing ratio_mock_reality in `mocks.create_flag_imaging_systematic`. The previous implementation did not work with other nside than 256.

* Add BGS_ANY / BGS_BRIGHT / BGS_FAINT to `regression.py/_get_feature_names`

* Follow now the flake8 convention.

* Add `save_table` option in `regressis/systematics.py/plot_systematic_from_map` to save line in .ecsv format in order to respect the DESI publication rules.

* Add DESIFootprint in `regressis/footprint.py`. The file is generated in NERSC at Nside=256. We generate the file instead of called
`desimodel.footprint.is_in_tile` to save computation time. Nside=256 was chosen to save memory in the repo.

* add verbose option in footprint class to enable MPI use.

* add South_mid region in regression. South_mid_ngc = DECaLS NGC and South_mid_sgc = DECaLS with dec > - 30.

1.0.0 (2022-01-10)
------------------

* Implementation of the photometric systematic mitigation.

* Mitigation run with this version in SV3 / MAIN DESI targets and on DA02 spectroscopic data.

0.0.1 (2022-01-03)
------------------

* Init git repo
