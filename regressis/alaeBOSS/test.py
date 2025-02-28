""" Just run python test.py and it should work """

from mpytools import Catalog 
import numpy as np

data = Catalog.read("/global/cfs/cdirs/desi/users/edmondc/desi_mocks/EZmocks6gpc/QSO_Y1_v1/data_imaging_mitigation/QSO_Y1_NGC_1.fits")
data = data[['RA', 'DEC', 'Z']]
data['WEIGHT'] = np.ones(len(data))
print(data)

randoms = Catalog.read("/global/cfs/cdirs/desi/users/edmondc/desi_mocks/EZmocks6gpc/QSO_Y1_v1/randoms-x10_QSO_Y1_NGC.fits")
randoms = randoms[['RA', 'DEC', 'Z']][::100]
randoms['WEIGHT'] = np.ones(len(randoms))
print(randoms)

from astropy.table import Table
dd, rr = Table(data.to_array()), Table(randoms.to_array())

from alaeboss import imsys_alaeboss

weight_imlim = imsys_alaeboss(dd, rr)