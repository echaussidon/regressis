""" Just run python test.py and it should work """

from mpytools import Catalog 
import numpy as np

data = Catalog.read(["/global/cfs/cdirs/desi/users/edmondc/desi_mocks/EZmocks6gpc/QSO_Y1_v1/data_imaging_mitigation/QSO_Y1_NGC_1.fits", "/global/cfs/cdirs/desi/users/edmondc/desi_mocks/EZmocks6gpc/QSO_Y1_v1/data_imaging_mitigation/QSO_Y1_SGC_1.fits"])
data = data[['RA', 'DEC', 'Z', 'MASKBITS']]
data['WEIGHT'] = np.ones(len(data))

sel = np.ones(data.size, dtype=bool)
for maskbit in [1, 7, 8, 11, 12, 13]:
    sel &= (data['MASKBITS'] & 2**maskbit == 0)
data = data[sel]
print(data)

randoms = Catalog.read([f"/global/cfs/cdirs/desi/users/edmondc/desi_mocks/EZmocks6gpc/QSO_Y1_v1/randoms-x10_QSO_Y1_{region}.fits" for region in ['NGC', 'SGC']])
randoms = randoms[['RA', 'DEC', 'Z', 'MASKBITS']]#[::10]
randoms['WEIGHT'] = np.ones(len(randoms))

sel = np.ones(randoms.size, dtype=bool)
for maskbit in [1, 7, 8, 11, 12, 13]:
    sel &= (randoms['MASKBITS'] & 2**maskbit == 0)
randoms = randoms[sel]
print(randoms)



from astropy.table import Table
dd, rr = Table(data.to_array()), Table(randoms.to_array())

import LSS.common_tools as common
inDES = common.select_regressis_DES(dd)
print(np.sum(inDES) / np.size(inDES))

dd['WEIGHT'], rr['WEIGHT'] = 1.0, 1.0
print(dd, np.shape(dd))
from alaeboss import imsys_alaeboss
weight_imlim = imsys_alaeboss(dd, rr, regl=['SnotDES'], randoms_as_NS=False, tracer='QSO', specprod='iron', release='Y1', datadir='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/')

#regl=['N', 'S'],

#, 'SnotDES'

#weight_imlim = imsys_alaeboss(dd, rr, regl=['N', 'S'])

# print(weight_imlim)