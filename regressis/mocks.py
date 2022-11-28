#!/usr/bin/env python
# coding: utf-8

import os
import logging

import fitsio
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

from regressis.utils import build_healpix_map


logger = logging.getLogger("Mocks")


def create_flag_imaging_systematic(mock, sel_pnz, wsys, fracarea=None, use_real_density=True, seed=123, show=False, savedir=None):
    """
    Create boolean mask to select which object from mocks with high density have to be kept to have systematic contaminated mocks.

    Parameters
    ----------
    mock : array like
        Array containing (Ra, Dec, Z, RAW_NZ, NZ) column.
        Note:
            * Z, RAW_NZ, NZ are used if use_real_density = False

    sel_pnz : boolean array
        Mask to select high denisty mock with correct n(z) in mock.
    wsys : :class:`PhotoWeight`
        Photometric weight used to compute mock computation. See :class:`PhotoWeight` to build it easily.
    fracarea : boolean array
        healpix map at wsys.nside. In order to correct the sky mean density of the mock
    use_real_density : bool
        If True use density from PhotoWeight class instead of the ratio from mock which is the same for each photometric footprint...
        If False, you do not match the observed density in each photometric footprint
    seed : int
        Numpy seed for reproductibility of np.random.random()
    show : bool
        If True display the figure
    savedir : str
        If None figures are not saved. Otherwise they are save in savedir

    Returns
    -------
    is_in_wsys_footprint : boolean array
        Mask to select which objects in mock are in the region where wsys is computed.
    is_for_wsys_cont : boolean array
        Mask to select which objects need to be kept to have contaminated mock. Already False where is_in_wsys_footprint is False.
    pix_number : int array
        Value of healpix pixel at wsys.nside in nested scheme for each object in mock.
    """
    # Compute the Healpix number for each objects (~13s for LRG catalog) # longer for ELG 30s ...
    # ask to do this computation at nside 512 for instance earlier in the code ?
    logger.info(f"Compute the healpix number at nside={wsys.nside} for each object in the mock")
    pix_number = hp.ang2pix(wsys.nside, mock['RA'], mock['DEC'], nest=True, lonlat=True)

    # Compute the ratio between high density mock (from which we will remove objects) and
    # and expected object density in reality.
    # Either use pre-chosen expected value (same in all the sky) with ratio of the two n(z)
    # Or use the ratio between mock density and wsys.mean_density_region (NOT expected in deg2) computed from real data.
    if use_real_density:
        logger.info('Use the real density from PhotoWeight class')
        mock_pnz_density = build_healpix_map(wsys.nside, mock['RA'], mock['DEC'], precomputed_pix=pix_number, sel=sel_pnz, in_deg2=False)
        if fracarea is None:
            fracarea = np.ones(mock_pnz_density.size)
            logger.info('Do not correct the density by the fracarea (useless if the input do not have any geometrical masks)')
        mock_pnz_density /= fracarea
        # renormalize also by the mean of wsys.map on each wsys.regions to get the exact mean_density_region
        # frac_to_remove = 1 - 1 / (wsys * ratio) --> do the mean
        ratio_mock_reality = {region: np.mean(mock_pnz_density[wsys.mask_region[region]]) / wsys.mean_density_region[region] / np.mean(wsys.map[wsys.mask_region[region]]) for region in wsys.regions}
    else:
        # TODO
        # This is not suitable for all mocks ...
        # add ratio_tmp as argument instead of compute it inside
        logger.info("Use the expected density from mock['RAW_NZ']/mock['NZ']")
        with np.errstate(divide='ignore'):
            ratio_tmp = np.min(np.abs(mock['RAW_NZ'][mock['Z'] > 0.3] / mock['NZ'][mock['Z'] > 0.3]))
        ratio_mock_reality = {region: ratio_tmp for region in wsys.regions}

    # Create flag for Legacy Imaging Survey
    is_in_wsys_footprint = (np.array(sum([wsys.mask_region[region] for region in wsys.regions])) > 0)[pix_number]
    # Build fraction of objetcs to remove
    frac_to_remove = wsys.fraction_to_remove_per_pixel(ratio_mock_reality)
    # Build flag to have contaminate mocks with correct n(z) (given by & sel_pnz)
    r = np.random.RandomState(seed)  # Set a random state for reproductibility
    is_for_wsys_cont = (r.random(pix_number.size) <= (1 - frac_to_remove[pix_number])) & sel_pnz
    logger.info(f'We remove {(~is_for_wsys_cont & is_in_wsys_footprint).sum() / is_in_wsys_footprint.sum():2.2%} of the objects in the mocks which are in wsys footprint !')

    if show or savedir is not None:
        from regressis.plot import plot_moll
        plot_moll(frac_to_remove, title='fraction of objects to remove per pixels', label='', galactic_plane=True, show=show, filename=None if (savedir is None) else os.path.join(savedir, 'contamination_moll.pdf'))

        plt.figure(figsize=(5, 4))
        plt.hist(frac_to_remove, histtype='step', bins=100, label='fraction to remove')
        plt.axvline(np.mean(frac_to_remove[frac_to_remove >= 0]), ls='--', c='k', label='mean on DR9')
        plt.legend()
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(os.path.join(savedir, 'frac_to_remove.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

        plt.figure(figsize=(5, 4))
        plt.hist(mock['Z'][sel_pnz & is_in_wsys_footprint], histtype='step', bins=100, range=(0.0, 2.4), label='high density n(z)')
        plt.hist(mock['Z'][is_for_wsys_cont], alpha=0.6, bins=100, range=(0.0, 2.4), label='final n(z)')
        plt.legend()
        plt.xlabel('z')
        plt.tight_layout()
        if savedir is not None:
            plt.savefig(os.path.join(savedir, 'endind_nz.pdf'))
        if show:
            plt.show()
        else:
            plt.close()

    return is_in_wsys_footprint, is_for_wsys_cont, pix_number


if __name__ == '__main__':
    """
    It is just a simple example of how contaminated DESI mocks with photometric imaging systematics.

    A jupyter notebook version of this code with figures is available in regressis/nb/contaminated_mocks.ipynb

    Remark: You need to be in NERSC and have desi access to run this command.
    """
    from regressis.utils import setup_logging
    setup_logging()

    def _desi_mock_filename(tracer='LRG', ph=0):
        """Collect the name of DESI Mocks in NERSC."""
        mock_dir = '/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/AbacusSummit/CutSky/'
        z_dic = {'LRG': 0.8, 'ELG': 1.1, 'QSO': 1.4}
        fname = f'{mock_dir}{tracer}/z{z_dic[tracer]:5.3f}/cutsky_{tracer}_z{z_dic[tracer]:5.3f}_AbacusSummit_base_c000_ph{ph:003d}.fits'
        return fname

    def _save_flag(flag, colname, savename):
        """Save flag with a specif colname in .fits file called savename"""
        fits = fitsio.FITS(savename, 'rw', clobber=True)  # clobber to overwrite file
        if len(flag) == 1:
            to_save = np.array(flag[0], dtype=[(colname[0], 'bool')])
            fits.write(to_save)
        elif len(flag) == 2:
            fits.write(flag, names=colname)
        fits.close()
        logger.info(f'WRITE flag in {savename}')

    # Load photometric weight
    from regressis import PhotoWeight
    wsys = PhotoWeight.load('/global/u2/e/edmondc/Software/regressis/res/SV3/SV3_LRG_256/RF/SV3_LRG_imaging_weight_256.npy')

    # Load mock
    mock_name = _desi_mock_filename(tracer='LRG', ph=0)
    logger.info(f"Read Mock: {mock_name}")
    mock = fitsio.FITS(mock_name)[1].read()

    # Load high density mock flag
    flag_dir = mock_name + '_flags'
    logger.info(f"Read Mock flag: {os.path.join(flag_dir, 'pnz.fits')}")
    sel_pnz = fitsio.FITS(os.path.join(flag_dir, 'pnz.fits'))[1]['pnz'].read()

    # Compute flag to build contaminated mock
    is_in_wsys_footprint, is_for_wsys_cont, pix_number = create_flag_imaging_systematic(mock, sel_pnz, wsys, use_real_density=True, show=False, savedir=None)

    if False:
        # Save flags in the flag directory
        _save_flag([pix_number, is_in_wsys_footprint, is_for_wsys_cont, wsys(pix_number)], ['HPX', 'IS_IN_WSYS', 'IS_FOR_CONT', 'WSYS'], os.path.join(flag_dir, 'photo_contamination.fits'))
