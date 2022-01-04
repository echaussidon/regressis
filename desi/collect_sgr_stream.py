#!/usr/bin/env python
# coding: utf-8

import os
import logging

import numpy as np
import pandas as pd
import healpy as hp
import fitsio

from astropy.coordinates import SkyCoord
from astropy import units as u

from regressis import utils, setup_logging


logger = logging.getLogger("Collect_sgr_stream")


def _mean_on_healpy_map(nside, map, depth_neighbours=1): # supposed nested and map a list of pixel
    """
    From map at nside, build the average with specific depth.
    It is similar to a convolution with a gaussian? kernel of size depth_neighbours.
    """
    def get_all_neighbours(nside, i, depth_neighbours=1):
        pixel_list = hp.get_all_neighbours(nside, i, nest=True)
        pixel_tmp = pixel_list
        depth_neighbours -= 1
        while depth_neighbours != 0 :
            pixel_tmp = hp.get_all_neighbours(nside, pixel_tmp, nest=True)
            pixel_tmp = np.reshape(pixel_tmp, pixel_tmp.size)
            pixel_list = np.append(pixel_list, pixel_tmp)
            depth_neighbours -= 1
        return pixel_list

    mean_map = np.zeros(len(map))
    for i in range(len(map)):
        neighbour_pixels = get_all_neighbours(nside, i, depth_neighbours)
        mean_map[i] = np.nansum(map[neighbour_pixels], axis=0)/neighbour_pixels.size
    return mean_map


def _match_to_dr9(cat_sag):
    """From a (R.A., Dec.) catalog match all the objects to the DR9 photometry."""
    def _collect_name_for_stream_region():
        # build quickly all the name that we need to explore all
        # the sweep containg Sgr. Stream information

        def build_list_name(ra, dec):
            lst = []
            for i in range(len(ra) - 1):
                for j in range(len(dec) - 1):
                    ra1, ra2 = str(ra[i]), str(ra[i+1])
                    dec1, dec2 = dec[j], dec[j+1]

                    if len(ra1) == 1:
                        ra1 = f'00{ra1}'
                    elif len(ra1) == 2:
                        ra1 = f'0{ra1}'

                    if len(ra2) == 1:
                        ra2 = f'00{ra2}'
                    elif len(ra2) == 2:
                        ra2 = f'0{ra2}'

                    if dec1<0:
                        dec1 = str(dec1)[1:]
                        sgn1 = 'm'
                    else:
                        dec1 = str(dec1)
                        sgn1 = 'p'
                    if len(dec1) == 1:
                        dec1 = f'00{dec1}'
                    else:
                        dec1 = f'0{dec1}'

                    if dec2<0:
                        dec2 = str(dec2)[1:]
                        sgn2 = 'm'
                    else:
                        dec2 = str(dec2)
                        sgn2 = 'p'
                    if len(dec2) == 1:
                        dec2 = f'00{dec2}'
                    else:
                        dec2= f'0{dec2}'

                    lst += [[ra1, sgn1, dec1, ra2, sgn2, dec2]]
            return lst

        ra_list = [0, 10, 20, 30, 40, 50, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300, 310, 320, 330, 340, 350, 360]
        dec_list = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]

        list_name = build_list_name(ra_list, dec_list)

        return list_name

    # where the DR9 sweeps are
    dirname = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/'
    sweepname = os.path.join(dirname, 'sweep-{}{}{}-{}{}{}.fits')
    list_name = _collect_name_for_stream_region()

    coord_sag = SkyCoord(ra=cat_sag['ra'].values*u.degree, dec=cat_sag['dec'].values*u.degree)
    logger.info(f"catalog sag size : {cat_sag.size}")

    sag_dr9 = pd.DataFrame()
    for name in list_name:
        sel_in_sag = (cat_sag['ra'].values < float(name[3])) & (cat_sag['ra'].values > float(name[0]))
        if name[1] == 'm':
            sel_in_sag &= (cat_sag['dec'].values > - float(name[2]))
        else:
            sel_in_sag &= (cat_sag['dec'].values > float(name[2]))
        if name[4] == 'm':
            sel_in_sag &= (cat_sag['dec'].values < - float(name[5]))
        else:
            sel_in_sag &= (cat_sag['dec'].values < float(name[5]))

        if sel_in_sag.sum():
            logger.info(f"[SWEEP] : {name}")
            logger.info(f"    * Number of objetcs in this sweep in sag catalog : {sel_in_sag.sum()}")
            try:
                columns = ['RA', 'DEC',
                           'FLUX_G', 'FLUX_R', 'FLUX_Z', 'FLUX_W1', 'FLUX_W2',
                           'MW_TRANSMISSION_G', 'MW_TRANSMISSION_R', 'MW_TRANSMISSION_Z', 'MW_TRANSMISSION_W1', 'MW_TRANSMISSION_W2']
                sweep = pd.DataFrame(fitsio.FITS(sweepname.format(*name))['SWEEP'][columns].read().byteswap().newbyteorder())
                coord_sweep = SkyCoord(ra=sweep['RA'].values*u.degree, dec=sweep['DEC'].values*u.degree)
                idx, d2d, d3d = coord_sag[sel_in_sag].match_to_catalog_sky(coord_sweep)
                sel = (d2d.arcsec < 1)
                logger.info(f"    * Number of objetcs selected in the sweep file : {sel.sum()}")
                sag_dr9 = pd.concat([sag_dr9, sweep.loc[idx[sel]]], ignore_index=True)
            except:
                print('')

    return sag_dr9


def _build_color_dataFrame(data):
    """
    Return color (Legacy Imaging Surveys like) dataframe from a data (array like) which contains the flux and the transmission in the five bands: g, r, z, W1, W2.
    A specific cut is applied to remove all the object with a missing photometric value and with too faint flux in WISE.
    """
    def mags_from_flux(data):
        # convert flux to magnitude applying NO photometric correction objects in the North.
        # Ok no objects are expected in the North.
        toret = []
        for band in ['G', 'R', 'Z', 'W1', 'W2']:
            flux = data['flux_{}'.format(band)] = data['FLUX_{}'.format(band)][:]/data['MW_TRANSMISSION_{}'.format(band)][:]
            flux[np.isinf(flux) | np.isnan(flux)] = 0.
            mag = np.where(gflux>0, 22.5-2.5*np.log10(gflux), 0.)
            mag[np.isinf(mag) | np.isnan(mag)] = 0.
            toret.append(mag)
        return toret

    def colors(g, r, z, W1, W2):
        # Compute the colors and keep also r as additional information.
        labels = ['g-r', 'r-z', 'g-z', 'g-W1', 'r-W1', 'z-W1', 'g-W2', 'r-W2', 'z-W2', 'W1-W2', 'r']
        colors = np.zeros((len(g), len(labels)))

        colors[:,0] = g-r
        colors[:,1] = r-z
        colors[:,2] = g-z
        colors[:,3] = g-W1
        colors[:,4] = r-W1
        colors[:,5] = z-W1
        colors[:,6] = g-W2
        colors[:,7] = r-W2
        colors[:,8] = z-W2
        colors[:,9] = W1-W2
        colors[:,10] = r

        return colors, labels

    g, r, z, W1, W2 = mags_from_flux(data)

    logger.info('We keep only stars without any photometric problems in DR9')
    sel = (r >= 16.0) & (g > 16.0) & (z > 16.0) & (W1 > 16.0) & (W2 > 16.0) # remove objects with a missing value
    sel &= (W1 < 24) & (W2 < 24) # remove to faint objects in WISE --> cannot be selected

    values, labels = colors(sel.sum(), g[sel], r[sel], z[sel], W1[sel], W2[sel])

    df_sag_colors = pd.DataFrame(values, columns=labels)
    # warning, we do not want to select in function of the index of the dataframe!
    df_sag_colors['RA'] = data['RA'].values[sel]
    df_sag_colors['DEC'] = data['DEC'].values[sel]

    return df_sag_colors


if __name__ == '__main__':

    setup_logging()

    # downlaod the file here: https://sites.google.com/fqa.ub.edu/tantoja/research/sagittarius
    filename = '../data/Sgr_members_L120_150_GaiaDR2.csv'
    logger.info(f"Load file: {filename}")
    cat_sag = pd.read_csv(filename)  # source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, bp_rp

    sag_dr9 = _match_to_dr9(cat_sag)

    sag_colors = _build_color_dataFrame(sag_dr9)

    sel = (sag_colors['r'] > 18) & (sag_colors['z-W1'] < -0.5) # remove true qsos from the catalog
    sgr_map = utils.build_healpix_map(256, sag_colors['RA'][sel], sag_colors['DEC'][sel], in_deg2=True)
    sgr_map /= np.mean(sgr_map[sgr_map > 0])
    sgr_map = _mean_on_healpy_map(256, sgr_map, depth_neighbours=2)

    logger.info('Save map at nside=128, 256, 512 in  ../data/')
    np.save('../data/sagittarius_stream_256.npy', sgr_map)
    np.save('../data/sagittarius_stream_512.npy', hp.ud_grade(sgr_map, 512, order_in='NESTED'))
    np.save('../data/sagittarius_stream_128.npy', hp.ud_grade(sgr_map, 128, order_in='NESTED'))
