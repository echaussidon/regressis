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


def _get_default_desi_tracer(version):
    """Return default list of tracers to use as a function of SV3 or MAIN."""
    version = version.upper()
    if version == 'SV3':
        return ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT', 'LRG', 'LRG_LOWDENS', 'ELG', 'ELG_LOP', 'ELG_HIP', 'QSO']
    if version == 'MAIN':
        return ['BGS_ANY', 'BGS_BRIGHT', 'BGS_FAINT', 'LRG', 'ELG', 'ELG_VLO', 'ELG_LOP', 'QSO']
    raise ValueError('Please choose either SV3 or MAIN for version')


def save_desi_targets(versions, nsides, dir_out, tracers=None):
    """
    Collect targets in NERSC from desitarget files and generate the density map in Healpix map with required nside and in nested scheme.

    Parameters
    ----------
    versions : array like
        Which versions will be considered. Either MAIN or SV3.
    nsides : array_like
        Healpix size of the maps.
    dir_out : str
        Directory where the maps will be saved.
    tracers : array like
        Name of tracers to be collected. If ``None`` get default tracer name with :func:`_get_default_desi_tracer`.
    """
    if np.ndim(versions) == 0:
        versions = [versions]
    if np.ndim(nsides) == 0:
        nsides = [nsides]

    for version in versions:
        version = version.upper()
        if version == 'SV3':
            from desitarget.sv3.sv3_targetmask import desi_mask, bgs_mask
            bright_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright/'
            dark_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/dark/'
            DESI_TARGET = 'SV3_DESI_TARGET'
            BGS_TARGET = 'SV3_BGS_TARGET'
        elif version == 'MAIN':
            from desitarget.targetmask import desi_mask, bgs_mask
            bright_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/bright/'
            dark_dir = '/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/dark/'
            DESI_TARGET = 'DESI_TARGET'
            BGS_TARGET = 'BGS_TARGET'
        else:
            raise ValueError('Please choose either SV3 or MAIN for version')

        if tracers is None:
            tracers = _get_default_desi_tracer(version)
        if np.ndim(tracers) == 0:
            tracers = [tracers]
        tracers = np.asarray(tracers)

        sel_bright = np.isin(tracers, ['BGS_ANY', 'BGS_FAINT', 'BGS_BRIGHT'])
        bright_tracer, dark_tracer = tracers[sel_bright], tracers[~sel_bright]

        if bright_tracer.size:
            logger.info(f"Collect {version} targets in bright time to build pixmap with nside={nsides}...")
            objects = read_targets_in_box(bright_dir, [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', DESI_TARGET, BGS_TARGET])
            desi_target, bgs_target = objects[DESI_TARGET][:], objects[BGS_TARGET][:]
            ra, dec = objects['RA'][:], objects['DEC'][:]

            for nside in nsides:
                for tracer in bright_tracer:
                    map_path = os.path.join(dir_out, f'{version}_{tracer}_targets_{nside}.npy')
                    logger.info(f"    * build healpix map for {tracer} and save it in: {map_path}")
                    if tracer in ['BGS_FAINT', 'BGS_BRIGHT']:
                        sel = (bgs_target & bgs_mask.mask(tracer)) != 0
                    else:
                        sel = (desi_target & desi_mask.mask(tracer)) != 0
                    np.save(map_path, build_healpix_map(nside, ra[sel], dec[sel], in_deg2=False))

        if dark_tracer.size:
            logger.info(f"Collect {version} targets in dark time to build pixmap with nside={nsides}...")
            objects = read_targets_in_box(dark_dir, [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', DESI_TARGET])
            desi_target = objects[DESI_TARGET][:]
            ra, dec = objects['RA'][:], objects['DEC'][:]

            for nside in nsides:
                for tracer in dark_tracer:
                    map_path = os.path.join(dir_out, f'{version}_{tracer}_targets_{nside}.npy')
                    logger.info(f"    * build healpix map for {tracer} and save it in: {map_path}")
                    sel = (desi_target & desi_mask.mask(tracer)) != 0
                    np.save(map_path, build_healpix_map(nside, ra[sel], dec[sel], in_deg2=False))


if __name__ == '__main__':

    setup_logging()
    # Turn off: WARNING  passed shape lies partially beyond the footprint of targets
    logging.getLogger('desiutil.log.dlm58.info').setLevel(logging.ERROR)

    versions = ['MAIN'] #['SV3', 'MAIN']
    nsides = [256, 512]
    dir_out = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../data/')

    save_desi_targets(versions, nsides, dir_out, tracers=None)
