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


def get_debv(mapname='/global/cfs/cdirs/desicollab/users/rongpu/data/ebv/desi_stars_y3/v0.1/final_maps/lss/desi_ebv_lss_256.fits'):
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

#'N','s'

# for wtmd see: https://github.com/desihub/LSS/blob/e6b85251283aae09b2bcb47f90d0ff15bf9b64f2/py/LSS/imaging/densvar.py#L372
def imsys_alaeboss(data, randoms, wtmd='wt', regl=['S'], randoms_as_NS=False, tracer='QSO', specprod='iron', release='Y1', datadir='/global/cfs/cdirs/desi/survey/catalogs/Y1/LSS/iron/LSScats/v1.5/', version_test=False):
    """ IMPORTANT: version_test == True is for testing only, it is expected to be False for the final run and be modified by hand here not reading any thing official."""

    from LSS.imaging import densvar

    if 'ELG' in tracer:
        zrl = [(0.8, 1.1), (1.1, 1.6)]
    if 'QSO' in tracer:
        zrl = [(0.8,1.3), (1.3, 2.1), (2.1, 3.5)]    
    if 'LRG' in tracer:
        if version_test:
            zrl = [(0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0), (1.0, 1.1)]
        else:
            zrl = [(0.4, 0.6), (0.6, 0.8), (0.8, 1.1)] 

    mainp = main(tracer, specprod, release)
    fit_maps = mainp.fit_maps
    if version_test:
        # use the test maps --> all maps for LRG:
        fit_maps = ['PSFDEPTH_W1', 'STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'EBV_DIFF_GR', 'EBV_DIFF_RZ','HI']

    data = addNS(data)
    if not randoms_as_NS:
        randoms = addNS(randoms)
    
    weight_imlim = np.ones(len(data), dtype='float')

    for reg in regl:
        if reg in ['SnotDES', 'DES']:
            reg_tmp = 'S'
        else: 
            reg_tmp = reg

        if version_test:
            # use the test maps
            pwf = datadir + '/hpmaps/' + 'QSO' + '_mapprops_healpix_nested_nside' + str(nside) + '_' + reg_tmp + '.fits'
        else:
            pwf = datadir + '/hpmaps/' + tracer + '_mapprops_healpix_nested_nside' + str(nside) + '_' + reg_tmp + '.fits'

        sys_tab = Table.read(pwf)
        cols = list(sys_tab.dtype.names)
        for col in cols:
            if 'DEPTH' in col:
                bnd = col.split('_')[-1]
                sys_tab[col] *= 10**(-0.4 * ext_coeff[bnd] * sys_tab['EBV'])
        for ec in ['GR','RZ']:
            if 'EBV_DIFF_' + ec in fit_maps: 
                sys_tab['EBV_DIFF_' + ec] = debv['EBV_DIFF_' + ec]

        for zr in zrl:
            zmin = zr[0]
            zmax = zr[1]
            print('getting weights for region '+reg+' and '+str(zmin)+'<z<'+str(zmax))
            if 'LRG' in tracer:
                if version_test:
                    # all maps for LRG:
                    fitmapsbin = ['PSFDEPTH_W1', 'STARDENS', 'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z', 'GALDEPTH_G', 'GALDEPTH_R', 'GALDEPTH_Z', 'EBV_DIFF_GR','EBV_DIFF_RZ', 'HI']
                else:
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

            # Redshift cuts directly in densvar
            wsysl = densvar.get_imweight(data, randoms, zmin, zmax, reg, fitmapsbin, fitmapsbin, plotr=False, sys_tab=sys_tab, zcol='Z', wtmd=wtmd)

            # print(wsysl)
            # print(wsysl == 1.0)
            # print(np.sum(wsysl == 1.0) / np.size(wsysl))

            # take care to select only the new weights in each redshift bins / regions
            sel = wsysl == 1.0
            weight_imlim[~sel] = wsysl[~sel]

    return weight_imlim 