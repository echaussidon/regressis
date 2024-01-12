import numpy as np
import fitsio
import healpy as hp
import pandas as pd

from regressis.utils import read_fits_to_pandas, build_healpix_map

# USe already precomputed maps
external_maps_256 = read_fits_to_pandas('/global/cfs/cdirs/desi/survey/catalogs/pixweight_maps_all/pixweight_external.fits')

# But You canr ecompute every thing from this file:
rand = read_fits_to_pandas('/dvs_ro/cfs/cdirs/desi/target/catalogs/dr9/0.49.0/randoms/resolve/randoms-1-1.fits', columns=['RA', 'DEC', 'TARGETID'])
tmp = read_fits_to_pandas('/global/cfs/cdirs/desi/survey/catalogs/external_input_maps/mapvalues/randoms-1-1-skymapvalues.fits')
rand = rand.merge(tmp, on='TARGETID')
# 'HALPHA', 'HALPHA_ERROR', 'CALIB_G', 'CALIB_R', 'CALIB_Z', 'EBV_CHIANG_SFDcorr', 'EBV_MPF_Mean_FW15', 'EBV_MPF_Mean_ZptCorr_FW15', 'EBV_MPF_Var_FW15', 'EBV_MPF_VarCorr_FW15', 'EBV_MPF_Mean_FW6P1', 'EBV_MPF_Mean_ZptCorr_FW6P1',
# 'EBV_MPF_Var_FW6P1','EBV_MPF_VarCorr_FW6P1', 'EBV_SGF14', 'BETA_ML', 'BETA_MEAN', 'BETA_RMS', 'HI', 'KAPPA_PLANCK'

for nside in [64, 128, 256]:
    external_maps = pd.DataFrame()
    external_maps['HPXPIXEL'] = np.arange(0, hp.nside2npix(nside))
    for col in external_maps_256.columns:
        if col != 'HPXPIXEL':
            external_maps[col] = hp.ud_grade(external_maps_256[col].values, nside, order_in='NESTED', order_out='NESTED')

    rongpu_maps = read_fits_to_pandas(f'/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/kp3_maps/v1_desi_ebv_{nside}.fits') # save in ring format ...
    for col in ['EBV_DESI_GR', 'EBV_SFD_GR', 'EBV_DESI_RZ', 'EBV_SFD_RZ']:
        external_maps[col] = hp.reorder(rongpu_maps[col], r2n=True)
    for ec in ['GR','RZ']:
        external_maps[f'EBV_DIFF_{ec}'] = hp.reorder(rongpu_maps[f'EBV_DESI_{ec}'] - rongpu_maps[f'EBV_SFD_{ec}'], r2n=True)

    nbr_rand_per_pix = build_healpix_map(nside, rand['RA'], rand['DEC'])
    for col in ['HI']:
        sum_over_pix = build_healpix_map(nside, rand['RA'], rand['DEC'], weights=rand[col])
        external_maps[col] = sum_over_pix / nbr_rand_per_pix

    # do not expect any nan values --> set to 0 (not a problem) we have values every where there is data
    external_maps[external_maps.isnull()] = 0. 

    fits = fitsio.FITS(f'../data/pixweight_external_{nside}.fits', 'rw')
    fits.write(external_maps.to_records())
    fits.close()



## Convert SGR and pixmap from 256 to 64!

#sgr_map = np.load('../data/sagittarius_stream_256.npy')
#np.save('../data/sagittarius_stream_64.npy', hp.ud_grade(sgr_map, 64, order_in='NESTED'))


#maps = read_fits_to_pandas('../data/pixweight-dr9-256.fits')
#maps_tmp = pd.DataFrame()
#maps_tmp['HPXPIXEL'] = np.arange(0, hp.nside2npix(64))
#for col in maps.columns:
#    if col != 'HPXPIXEL':
#        maps_tmp[col] = hp.ud_grade(maps[col].values, 64, order_in='NESTED')
#fits = fitsio.FITS('../data/pixweight-dr9-64.fits', 'rw')
#fits.write(maps_tmp.to_records())
#fits.close()