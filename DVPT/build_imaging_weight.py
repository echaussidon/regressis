import numpy as np
import healpy as hp
import fitsio

CATALOG = '/global/project/projectdirs/desi/users/shadaba/SV3Targets/Catalogs/'
catalog_name = CATALOG + 'sv3target_LRG_ANY_SDECALS.fits'

SYS_WEIGHT = '/global/homes/e/edmondc/Scratch/Imaging_weight/'
Nside = 512 # 512 for LRG and ELG // 256 for QSO
weight_name = SYS_WEIGHT + f'LRG_imaging_weight_{Nside}.npy'

print(f"\n[INFO] Read imaging weight: {weight name}")
weight = np.load(weight_name)

print(f"[INFO] READ File: {catalog_name}")
catalog = fitsio.FITS(catalog_name)[1]['RA', 'DEC'] #take care qso are already mask with basic mask in dr9 1, 12, 13
ra, dec = catalog['RA'][:], catalog['DEC'][:]
pix = hp.ang2pix(Nside, ra, dec, nest=True, lonlat=True)

print("Build weight for each objects in catalog...")
w = np.array([weight[p] for p in pix])

print(f"Info on w: Mean = {w.mean()}, Std = {w.std()} ")
