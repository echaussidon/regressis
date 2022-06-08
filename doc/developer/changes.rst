.. _developer-changes:

Change Log
==========

Main (2022-06-08)
-----------------

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
