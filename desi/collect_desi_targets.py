#!/usr/bin/env python
# coding: utf-8

import os
import sys
import logging

import numpy as np
import healpy as hp

from desitarget.io import read_targets_in_box

from regressis import setup_logging
from regressis.utils import build_healpix_map


logger = logging.getLogger('Collect_desi_target')


def _load_default_desi_tracer(version):
    """
    Return default tracer to use as a function of SV3 or MAIN.
    """
    if version == 'SV3':
        return ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT', 'LRG', 'LRG_LOWDENS', 'ELG', 'ELG_LOP', 'ELG_HIP', 'QSO']
    if version == 'MAIN':
        return ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT', 'LRG', 'ELG', 'ELG_VLO', 'ELG_LOP', 'QSO']
    raise ValueError('Please choose either SV3 or MAIN for version')


def save_desi_targets(version_list, tracer_list, nside_list, dir_out):
    """
    Collect targets in NERSC from desitarget files and generate the density map in Healpix map with nside and nested scheme.

    Parameters
    ----------
    version_list: array like
        Which version will be considered. Either MAIN or SV3
    tracer_list: array like
        Name of tracer which has to be collected. If None load default tracer name with _load_default_desi_tracer
    nside_list: array_like
        Healpix size of the saved maps.
    dir_out: str
        Path where the maps will be saved
    """
    for version in version_list:
        if version == 'SV3':
            from desitarget.sv3.sv3_targetmask import desi_mask, bgs_mask
            bright_dir='/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright/'
            dark_dir='/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/dark/'
            DESI_TARGET='SV3_DESI_TARGET'
            BGS_TARGET='SV3_BGS_TARGET'
        elif version == 'MAIN':
            from desitarget.targetmask import desi_mask, bgs_mask
            bright_dir='/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/bright/'
            dark_dir='/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/dark/'
            DESI_TARGET='SV3_DESI_TARGET'
            BGS_TARGET='SV3_BGS_TARGET'
        else:
            raise ValueError('Please choose either SV3 or MAIN for version')

        if tracer_list is None:
            tracer_list = _load_default_desi_tracer(version)
        tracer_list = np.array(tracer_list)

        sel_bright = np.isin(tracer_list, ['BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT'])
        bright_tracer, dark_tracer = tracer_list[sel_bright], tracer_list[~sel_bright]

        if bright_tracer.size:
            logger.info(f"Collect {version} targets in Bright time to build pixmap...")
            objects = read_targets_in_box(bright_dir, [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', DESI_TARGET, BGS_TARGET])
            desi_target, bgs_target = objects[DESI_TARGET][:], objects[BGS_TARGET][:]
            ra, dec = objects['RA'][:], objects['DEC'][:]

            for nside in nside_list:
                for tracer in bright_tracer:
                    map_path = os.path.join(dir_out, f'{version}_{tracer}_targets_{nside}.npy')
                    logger.info(f"    * build healpix map for {tracer} and save it in: {map_path}")
                    if tracer in ['BGS_FAINT', 'BGS_BRIGHT']:
                        sel = (bgs_target & bgs_mask.mask(tracer)) != 0
                    else:
                        sel = (desi_target & desi_mask.mask(tracer)) != 0
                    np.save(map_path, build_healpix_map(nside, ra[sel], dec[sel], in_deg2=False))

        if dark_tracer.size:
            logger.info(f"Collect {version} targets in Dark time to build pixmap with Nside={nside}...")
            objects = read_targets_in_box(dark_dir, [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', DESI_TARGET])
            desi_target = objects[DESI_TARGET][:]
            ra, dec = objects['RA'][:], objects['DEC'][:]

            for nside in nside_list:
                for tracer in dark_tracer:
                    map_path = os.path.join(dir_out, f'{version}_{tracer}_targets_{nside}.npy')
                    logger.info(f"    * build healpix map for {tracer} and save it in: {map_path}")
                    sel = (desi_target & desi_mask.mask(tracer)) != 0
                    np.save(map_path, build_healpix_map(nside, ra[sel], dec[sel], in_deg2=False))


if __name__ == '__main__':

    setup_logging()

    # Turn off: WARNING  passed shape lies partially beyond the footprint of targets
    logging.getLogger('desiutil.log.dlm58.info').setLevel(logging.ERROR)

    version = ['SV3', 'MAIN']
    tracer = None
    nside = [256, 512]
    dir_out = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/')

    save_desi_targets(version, tracer, nside, dir_out)
