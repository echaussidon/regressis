#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import healpy as hp
import fitsio

from astropy.coordinates import SkyCoord
from astropy import units as u

import logging
from regressis import setup_logging


logger = logging.getLogger("Collect_sgr_stream")


def _build_pixmap(data, sel, nside=256):
    """
    From (R.A., Dec.) catalog and selective mask, build the corresponding distribution
    healpix map at nside in nested scheme
    """
    targets = np.zeros(hp.nside2npix(nside))
    pixels = hp.ang2pix(nside, data['RA'][sel], data['DEC'][sel], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    targets[pix] = counts
    return targets


def _mean_on_healpy_map(nside, map, depth_neighbours=1): #supposed Nested and map a list of pixel
    """
    From map at nside, build the average with specific depth.
    It is similar than a convolution in a 2d matrix with a gaussian kernel of size depth_neighbours.
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
    """
    From a (R.A., Dec.) catalog match all the objects to the DR9 photometry
    """
    def _collect_name_for_stream_region():
        # build quickly all the name that we need to explore all
        # the sweep containg Sgr. Stream information
        def reorganise(lst):
            for elt in lst:
                elt[1], elt[2], elt[3] = elt[2], elt[3], elt[1]

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

                    lst += [[ra1, ra2, sgn1, dec1, sgn2, dec2]]
            return lst

        ra_list = [0, 10, 20, 30, 40, 50, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 300, 310, 320, 330, 340, 350, 360]
        dec_list = [-30, -25, -20, -15, -10, -5, 0, 5, 10, 15, 20, 25, 30]
        list_name = build_list_name(ra_list, dec_list)
        return reorganise(list_name)

    # where the DR9 SWEEP ARE
    SWEEP = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/south/sweep/9.0/'
    sweepname = SWEEP + 'sweep-{}{}{}-{}{}{}.fits'
    list_name = _collect_name_for_stream_region()

    coorg_sag = SkyCoord(ra=cat_sag['ra'].values*u.degree, dec=cat_sag['dec'].values*u.degree)
    logger.info(f"catalog sag size : ", cat_sag.size, '\n')

    sag_dr9 = pd.DataFrame()
    for name in list_name:
        sel_in_sag = (cat_sag['ra'].values < float(name[3])) & (cat_sag['ra'].values > float(name[0]))
        if name[1] == 'm':
            sel_in_sag &=  (cat_sag['dec'].values > - float(name[2]))
        else:
            sel_in_sag &=  (cat_sag['dec'].values > float(name[2]))
        if name[4] == 'm':
            sel_in_sag &= (cat_sag['dec'].values < - float(name[5]))
        else:
            sel_in_sag &= (cat_sag['dec'].values < float(name[5]))

        if sel_in_sag.sum() != 0:
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
                logger.info(f"    *  Number of objetcs selected in the sweep file : {sel.sum()}")
                sag_dr9 = pd.concat([sag_dr9, sweep[idx[sel]]], ignore_index=True)
            except:
                print('')

    return sag_dr9


def _build_color_dataFrame(data):
    """
    Return color (Legacy Imaging Surveys like) dataframe from a data (array like) which contains the flux and the transmission in the five bands: g, r, z, W1, W2.
    A specific cut is applied to remove all the object with a missing photometric value and with to faint flux in WISE.
    """
    def magsExtFromFlux(dataArray):
        # convert flux to magnitude applying photometric correction objects in the North.
        from desitarget.cuts import shift_photo_north

        gflux  = dataArray['FLUX_G'][:]/dataArray['MW_TRANSMISSION_G'][:]
        rflux  = dataArray['FLUX_R'][:]/dataArray['MW_TRANSMISSION_R'][:]
        zflux  = dataArray['FLUX_Z'][:]/dataArray['MW_TRANSMISSION_Z'][:]
        W1flux  = dataArray['FLUX_W1'][:]/dataArray['MW_TRANSMISSION_W1'][:]
        W2flux  = dataArray['FLUX_W2'][:]/dataArray['MW_TRANSMISSION_W2'][:]

        W1flux[np.isnan(W1flux)]=0.
        W2flux[np.isnan(W2flux)]=0.
        gflux[np.isnan(gflux)]=0.
        rflux[np.isnan(rflux)]=0.
        zflux[np.isnan(zflux)]=0.
        W1flux[np.isinf(W1flux)]=0.
        W2flux[np.isinf(W2flux)]=0.
        gflux[np.isinf(gflux)]=0.
        rflux[np.isinf(rflux)]=0.
        zflux[np.isinf(zflux)]=0.


        is_north = (dataArray['DEC'][:] >= 32) & (dataArray['RA'][:] >= 60) & (dataArray['RA'][:] <= 310)
        logger.info(f'shift photometry for {is_north.sum()} objects')
        gflux[is_north], rflux[is_north], zflux[is_north] = shift_photo_north(gflux[is_north], rflux[is_north], zflux[is_north])

        g=np.where(gflux>0, 22.5-2.5*np.log10(gflux), 0.)
        r=np.where(rflux>0,22.5-2.5*np.log10(rflux), 0.)
        z=np.where(zflux>0,22.5-2.5*np.log10(zflux), 0.)
        W1=np.where(W1flux>0, 22.5-2.5*np.log10(W1flux), 0.)
        W2=np.where(W2flux>0, 22.5-2.5*np.log10(W2flux), 0.)

        g[np.isnan(g)]=0.
        g[np.isinf(g)]=0.
        r[np.isnan(r)]=0.
        r[np.isinf(r)]=0.
        z[np.isnan(z)]=0.
        z[np.isinf(z)]=0.
        W1[np.isnan(W1)]=0.
        W1[np.isinf(W1)]=0.
        W2[np.isnan(W2)]=0.
        W2[np.isinf(W2)]=0.

        return g, r, z, W1, W2

    def colors(nbEntries, nfeatures, g, r, z, W1, W2):
        # Compute the colors and keep also r as additional information.
        colors = np.zeros((nbEntries,nfeatures))

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

        return colors

    g, r, z, W1, W2 = magsExtFromFlux(data)

    logger.info('We keep only stars without any photometric problems in DR9')
    sel = (r >= 16.0) & (g > 16.0) & (z > 16.0) & (W1 > 16.0) & (W2 > 16.0) # remove objects with a missing value
    sel &= (W1 < 24) & (W2 < 24) # remove to faint objects in WISE --> cannot be selected

    attributes = colors(sel.sum(), 11, g[sel], r[sel], z[sel], W1[sel], W2[sel])
    attributes_label = ['g-r', 'r-z', 'g-z', 'g-W1', 'r-W1', 'z-W1', 'g-W2', 'r-W2', 'z-W2', 'W1-W2', 'r']

    df_sag_colors = pd.DataFrame(attributes, columns=attributes_label)
    df_sag_colors['RA'] = data['RA'][sel]
    df_sag_colors['DEC'] = data['DEC'][sel]

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
    sgr_map = _build_pixmap(sag_colors, sel, 256) / hp.nside2pixarea(256, degrees=True)
    sgr_map /= np.mean(sgr_map[sgr_map > 0])
    sgr_map = _mean_on_healpy_map(256, sgr_map, depth_neighbours=2)

    logger.info('Save map at nside=128, 256, 512 in  ../data/')
    np.save('../data/sagittarius_stream_256.npy', sgr_map)
    np.save('../data/sagittarius_stream_512.npy', hp.ud_grade(sgr_map, 512, order_in='NESTED'))
    np.save('../data/sagittarius_stream_128.npy', hp.ud_grade(sgr_map, 128, order_in='NESTED'))
