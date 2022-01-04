#!/bin/bash

# log on an interactive node
# salloc -N 1 -C haswell -t 04:00:00 --qos interactive -L SCRATCH,project

# settings in the interactive node
source /project/projectdirs/desi/software/desi_environment.sh
module use /global/common/software/desi/${NERSC_HOST}/desiconda/startup/modulefiles
module load desimodules

# required by make_imaging_weight_map
export GAIA_DIR='/global/cfs/projectdirs/desi/target/gaia_dr2'

# Path & variable
RANDOMS=/global/cfs/projectdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-7.fits
TARGETS=/global/cfs/projectdirs/desi/target/catalogs/dr9/0.49.0/targets/main/resolve/dark/targets-dark-hp-11.fits
OUTDIR=$PWD/../data

NSIDE=128
echo "Build pixweight map for DR9 with Nside:128"
echo " "
## run make_imaging_weight_map on 32 processors
make_imaging_weight_map ${RANDOMS} ${TARGETS} ${OUTDIR}/pixweight-dr9-${NSIDE}.fits --nside ${NSIDE}

# NSIDE=256
# echo "Build pixweight map for DR9 with Nside:256"
# echo " "
# ## run make_imaging_weight_map on 32 processors
# make_imaging_weight_map ${RANDOMS} ${TARGETS} ${OUTDIR}/pixweight-dr9-${NSIDE}.fits --nside ${NSIDE}
#
#
# NSIDE=512
# echo " "
# echo "Build pixweight map for DR9 with Nside:512"
# echo " "
# ## run make_imaging_weight_map on 32 processors
# make_imaging_weight_map ${RANDOMS} ${TARGETS} ${OUTDIR}/pixweight-dr9-${NSIDE}.fits --nside ${NSIDE}
