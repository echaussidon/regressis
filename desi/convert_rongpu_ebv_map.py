import numpy as np
import fitsio
import healpy as hp
import pandas as pd

from regressis.utils import read_fits_to_pandas


#sgr_map = np.load('../data/sagittarius_stream_256.npy')
#np.save('../data/sagittarius_stream_64.npy', hp.ud_grade(sgr_map, 64, order_in='NESTED'))


#rg_maps = read_fits_to_pandas('../data/initial_corrected_ebv_map_nside_64.fits')
#rg_maps_tmp = pd.DataFrame()
#rg_maps_tmp['HPXPIXEL'] = np.arange(0, hp.nside2npix(64))
#for col in rg_maps.columns:
#    if col != 'HPXPIXEL':
#        tmp = np.zeros(hp.nside2npix(64))
#        tmp[rg_maps['HPXPIXEL'].values] = rg_maps[col]
#        rg_maps_tmp[col] = hp.reorder(tmp, r2n=True)
#rg_maps_tmp.rename(columns={"EBV_NEW": "EBV_RGP"},inplace=True)
#fits = fitsio.FITS('../data/new_ebv_rgp_64.fits', 'rw')
#fits.write(rg_maps_tmp.to_records())
#fits.close()


rg_maps = read_fits_to_pandas('../data/initial_corrected_ebv_map_nside_64.fits')
rg_maps_tmp = pd.DataFrame()
rg_maps_tmp['HPXPIXEL'] = np.arange(0, hp.nside2npix(256))
for col in rg_maps.columns:
    if col != 'HPXPIXEL':
        tmp = np.zeros(hp.nside2npix(64))
        tmp[rg_maps['HPXPIXEL'].values] = rg_maps[col]
        rg_maps_tmp[col] = hp.ud_grade(hp.reorder(tmp, r2n=True), 256, order_in='NESTED')
rg_maps_tmp.rename(columns={"EBV_NEW": "EBV_RGP"},inplace=True)
print(rg_maps_tmp.to_records())
fits = fitsio.FITS('../data/new_ebv_rgp_256.fits', 'rw')
fits.write(rg_maps_tmp.to_records())
fits.close()


#maps = read_fits_to_pandas('../data/pixweight-dr9-256.fits')
#maps_tmp = pd.DataFrame()
#maps_tmp['HPXPIXEL'] = np.arange(0, hp.nside2npix(64))
#for col in maps.columns:
#    if col != 'HPXPIXEL':
#        maps_tmp[col] = hp.ud_grade(maps[col].values, 64, order_in='NESTED')
#fits = fitsio.FITS('../data/pixweight-dr9-64.fits', 'rw')
#fits.write(maps_tmp.to_records())
#fits.close()