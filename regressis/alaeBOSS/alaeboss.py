""" 
Taken from LSS repo: 
    * https://github.com/desihub/LSS/blob/main/scripts/mock_tools/addsys.py 

"""

import fitsio
import astropy.io.fits as fits
from astropy.table import Table, join
import healpy as hp
import numpy as np
from matplotlib import pyplot as plt

from LSS.globals import main

ext_coeff = {'G':3.214, 'R':2.165,'Z':1.211,'W1':0.184,'W2':0.113}
nside = 256

def addNS(tab):
    '''
    **ADAPTED FOR THIS CODE **
    given a table that already includes RA,DEC, add PHOTSYS column denoting whether
    the data is in the DECaLS ('S') or BASS/MzLS ('N') photometric region
    '''
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(tab['RA']* u.deg, tab['DEC']* u.deg, frame='icrs')
    gc = c.transform_to('galactic')
    sel_ngc = gc.b > 0

    tab['PHOTSYS'] = 'S' #np.array(['S' for i in range(len(tab))])
    seln = tab['DEC'] > 32.375
    seln &= sel_ngc
    tab['PHOTSYS'][seln] = 'N'
    return tab

def get_debv(mapname = '/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/desi_ebv_lss_256.fits'):
    """
    #DR1 map is in '/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/kp3_maps/'
    #DR1 map named like /global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars/kp3_maps/v1_desi_ebv_256.fits
    #DR2 map is in /global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/
    #DR2 map named like desi_ebv_lss_256.fits
    """
    import healpy as hp
    nside = 256
    nest = False
    eclrs = ['gr','rz']
    debv = Table()
    for ec in eclrs:
        ebvn = fitsio.read(mapname)
        debv_a = ebvn['EBV_DESI_'+ec.upper()]-ebvn['EBV_SFD_'+ec.upper()]
        debv_a = hp.reorder(debv_a,r2n=True)
        debv['EBV_DIFF_'+ec.upper()] = debv_a
    return debv

debv = get_debv()

# for wtmd see: https://github.com/desihub/LSS/blob/e6b85251283aae09b2bcb47f90d0ff15bf9b64f2/py/LSS/imaging/densvar.py#L372
def imsys_alaeboss(data, randoms, wtmd='wt', regl=['N', 'S'], randoms_as_NS=False, tracer='QSO', specprod='iron', release='Y1', datadir='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/'):
    from LSS.imaging import densvar

    if 'ELG' in tracer:
        zrl = [(0.8, 1.1), (1.1, 1.6)]
    if 'QSO' in tracer:
        zrl = [(0.8,1.3), (1.3, 2.1), (2.1, 3.5)]    
    if 'LRG' in tracer:
        zrl = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)] 

    mainp = main(tracer, specprod, release)
    fit_maps = mainp.fit_maps

    data = addNS(data)
    if not randoms_as_NS:
        randoms = addNS(randoms)
    
    weight_imlim = np.ones(len(data))

    for reg in regl:
        pwf = datadir + '/hpmaps/' + tracer + '_mapprops_healpix_nested_nside' + str(nside) + '_' + reg + '.fits'

        sys_tab = Table.read(pwf)
        cols = list(sys_tab.dtype.names)
        for col in cols:
            if 'DEPTH' in col:
                bnd = col.split('_')[-1]
                sys_tab[col] *= 10**(-0.4 * ext_coeff[bnd] * sys_tab['EBV'])
        for ec in ['GR','RZ']:
            if 'EBV_DIFF_' + ec in fit_maps: 
                sys_tab['EBV_DIFF_' + ec] = debv['EBV_DIFF_' + ec]
        
        seld = data['PHOTSYS'] == reg
        selr = randoms['PHOTSYS'] == reg
        for zr in zrl:
            zmin = zr[0]
            zmax = zr[1]
            print('getting weights for region '+reg+' and '+str(zmin)+'<z<'+str(zmax))
            if 'LRG' in tracer:
                if reg == 'N':
                    fitmapsbin = fit_maps
                else:
                    if zmax == 0.6:
                        fitmapsbin = mainp.fit_maps46s
                    elif zmax == 0.8:
                        fitmapsbin = mainp.fit_maps68s
                    elif zmax == 1.1:
                        fitmapsbin = mainp.fit_maps81s
            else:
                fitmapsbin = fit_maps

            wsysl = densvar.get_imweight(data[seld], randoms[selr], zmin, zmax, reg, fitmapsbin, fitmapsbin, plotr=False, sys_tab=sys_tab, zcol='Z', wtmd=wtmd)
            # sel only the correct object in the redshift range
            sel = wsysl != 1
            weight_imlim[seld][sel] = wsysl[sel]

    return weight_imlim 