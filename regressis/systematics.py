#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import logging

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


logger = logging.getLogger("Systematics")


def _get_desi_plot_attrs(feature_names, region):
    """
    Return dictionary of [min, max, label, nbins, function_to_convert] for each feature in feature_names for plotting purposes.
    Limits and number of bins are adapted for the DESI Legacy Imaging Survey DR9.
    """
    def _get_plot_attrs(name, region):
        """
        Return dictionary of [min, max, label, nbins, function_to_convert] for name.
        """
        def id(x):
            return x

        def convert_depth(x):
            return 22.5 - 2.5 * np.log10(5 / np.sqrt(x))

        def convert_stardens(x):
            return np.log10(x)

        sysdict = {}
        sysdict['STARDENS'] = {'north': [150., 4000., 'log10(Stellar Density) per sq. deg.', 35, convert_stardens],
                               'south': [150., 4000., 'log10(Stellar Density) per sq. deg.', 35, convert_stardens],
                               'des': [150., 4000., 'log10(Stellar Density) per sq. deg.', 30, convert_stardens],
                               'global': [150., 4000., 'log10(Stellar Density) per sq. deg.', 30, convert_stardens]}

        sysdict['EBV'] = {'north': [0.001, 0.1, 'E(B-V)', 35, id],
                          'south': [0.001, 0.1, 'E(B-V)', 35, id],
                          'des': [0.001, 0.09, 'E(B-V)', 35, id],
                          'global': [0.001, 0.1, 'E(B-V)', 30, id]}

        sysdict['STREAM'] = {'north': [0., 1., 'Sgr. Stream', 1, id],
                             'south': [0.01, 1., 'Sgr. Stream', 20, id],
                             'des': [0.01, 0.6, 'Sgr. Stream', 15, id],
                             'global': [0.01, 1.5, 'Sgr. Stream', 20, id]}

        sysdict['PSFSIZE_G'] = {'north': [1.3, 2.6, 'PSF Size in g-band', 35, id],
                                'south': [1.1, 2.02, 'PSF Size in g-band', 35, id],
                                'des': [1.19, 1.7, 'PSF Size in g-band', 30, id],
                                'global': [0., 3., 'PSF Size in g-band', 30, id]}

        sysdict['PSFSIZE_R'] = {'north': [1.25, 2.4, 'PSF Size in r-band', 40, id],
                                'south': [0.95, 1.92, 'PSF Size in r-band', 30, id],
                                'des': [1.05, 1.5, 'PSF Size in r-band', 30, id],
                                'global': [0., 3., 'PSF Size in r-band', 30, id]}

        sysdict['PSFSIZE_Z'] = {'north': [0.9, 1.78, 'PSF Size in z-band', 35, id],
                                'south': [0.9, 1.85, 'PSF Size in z-band', 40, id],
                                'des': [0.95, 1.4, 'PSF Size in z-band', 30, id],
                                'global': [0., 3., 'PSF Size in z-band', 30, id]}

        sysdict['PSFDEPTH_G'] = {'north': [300., 1600., 'PSF Depth in g-band', 30, convert_depth],
                                 'south': [750., 4000., 'PSF Depth in g-band', 35, convert_depth],
                                 'des': [1900., 7000., 'PSF Depth in g-band', 30, convert_depth],
                                 'global': [63., 6300., 'PSF Depth in g-band', 30, convert_depth]}

        sysdict['PSFDEPTH_R'] = {'north': [95., 620., 'PSF Depth in r-band', 30, convert_depth],
                                 'south': [260.0, 1600.0, 'PSF Depth in r-band', 30, convert_depth],
                                 'des': [1200., 5523., 'PSF Depth in r-band', 30, convert_depth],
                                 'global': [25., 2500., 'PSF Depth in r-band', 30, convert_depth]}

        sysdict['PSFDEPTH_Z'] = {'north': [60., 275., 'PSF Depth in z-band', 40, convert_depth],
                                 'south': [40.0, 360., 'PSF Depth in z-band', 40, convert_depth],
                                 'des': [145., 570., 'PSF Depth in z-band', 30, convert_depth],
                                 'global': [4., 400., 'PSF Depth in z-band', 30, convert_depth]}

        sysdict['PSFDEPTH_W1'] = {'north': [2.7, 12., 'PSF Depth in W1-band', 40, convert_depth],
                                  'south': [2.28, 5.5, 'PSF Depth in W1-band', 30, convert_depth],
                                  'des': [2.28, 6.8, 'PSF Depth in W1-band', 30, convert_depth],
                                  'global': [0.0, 30.0, 'PSF Depth in W1-band', 30, convert_depth]}

        sysdict['PSFDEPTH_W2'] = {'north': [0.8, 3.9, 'PSF Depth in W2-band', 40, convert_depth],
                                  'south': [0.629, 1.6, 'PSF Depth in W2-band', 30, convert_depth],
                                  'des': [0.62, 2.25, 'PSF Depth in W2-band', 30, convert_depth],
                                  'global': [0.0, 7.0, 'PSF Depth in W2-band', 30, convert_depth]}
                                
        sysdict['HI'] = {'north': [10**19.7, 10**21.3, 'log10(HI)', 35, convert_stardens],
                         'south': [10**19.7, 10**21.3, 'log10(HI)', 35, convert_stardens],
                         'des': [10**19.7, 10**21.3, 'log10(HI)', 30, convert_stardens],
                         'global': [10**19.7, 10**21.3, 'log10(HI)', 40, convert_stardens]}
        
        sysdict['EBV_DIFF_GR'] = {'north': [0.001, 0.1, 'E(B-V) diff GR', 35, id],
                                  'south': [0.001, 0.1, 'E(B-V) diff GR', 35, id],
                                  'des':[0.001, 0.09, 'E(B-V) diff GR', 35, id],
                                  'global': [0.001, 0.1, 'E(B-V) diff GR', 35, id]}
        
        sysdict['EBV_DIFF_RZ'] = {'north': [0.001, 0.1, 'E(B-V) diff RZ', 35, id],
                                  'south': [0.001, 0.1, 'E(B-V) diff RZ', 35, id],
                                  'des':[0.001, 0.09, 'E(B-V) diff RZ', 35, id],
                                  'global': [0.001, 0.1, 'E(B-V) diff RZ', 35, id]}

        sysdict['GALDEPTH_G'] = {'north': [200., 1600., 'Gal Depth in g-band', 30, convert_depth],
                                 'south': [200., 4000., 'Gal Depth in g-band', 30, convert_depth],
                                 'des': [1000., 4500., 'Gal Depth in g-band', 30, convert_depth],
                                 'global': [200., 4500., 'Gal Depth in g-band', 30, convert_depth]}

        sysdict['GALDEPTH_R'] = {'north': [95., 620., 'Gal Depth in r-band', 30, convert_depth],
                                 'south': [100.0, 1550.0, 'Gal Depth in r-band', 30, convert_depth],
                                 'des': [700., 2500., 'Gal Depth in r-band', 30, convert_depth],
                                 'global': [25., 2500., 'Gal Depth in r-band', 30, convert_depth]}

        sysdict['GALDEPTH_Z'] = {'north': [20., 200., 'Gal Depth in z-band', 30, convert_depth],
                                 'south': [20.0, 220, 'Gal Depth in z-band', 40, convert_depth],
                                 'des': [20.0, 400., 'Gal Depth in z-band', 30, convert_depth],
                                 'global': [20., 400., 'Gal Depth in z-band', 30, convert_depth]}

        if name in sysdict.keys():
            return sysdict[name][region]

        if name[:5] == 'CALIB':
            return [-0.001, 0.006, name, 30, id]
        if name[:3] == 'EBV':
            return [0.001, 0.13, name, 30, id]
        if name[:5] == 'KAPPA':
            return [0.001, 0.8, name, 30, id]
        if name[:6] == 'HALPHA':
            return [0.001, 6.5, name, 30, id]
        return [0.1, 10000, name, 30, id]

    region = region.lower()
    if region in ['south_ngc', 'south_sgc', 'south_mid', 'south_mid_ngc', 'south_mid_sgc', 'south_all', 'south_all_ngc', 'south_all_sgc']:
        region = 'south'
    elif region in ['des_mid', 'south_pole']:
        region = 'des'

    return {name: _get_plot_attrs(name, region) for name in feature_names}


def _systematics_med(targets, feature, feature_name, downclip=None, upclip=None, nbins=50, use_mean=True, adaptative_binning=False, nobj_per_bin=1000):
    """
    Return bin centers ``binmid``, median density ``meds`` to do:

    .. code-block:: python

      plt.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.9)

    """
    sel = (feature >= downclip) & (feature < upclip)
    if not np.any(sel):
        logger.info(f"Proceeding without clipping systematics for {feature_name}")
    targets = targets[sel]
    feature = feature[sel]

    if adaptative_binning:  # set this option to have variable bin sizes so that each bin contains the same number of pixel (nobj_per_bin)
        nbr_obj_bins = nobj_per_bin
        ksort = np.argsort(feature)
        bins = feature[ksort[0::nbr_obj_bins]]  # Here, get the bins from the data set (needed to be sorted)
        bins = np.append(bins, feature[ksort[-1]])  # add last point
        nbins = bins.size - 1  # OK
    else:  # create bin with fix size (depends on the up/downclip value or minimal / maximal value of feature if up/downclip is too large)
        nbr_obj_bins, bins = np.histogram(feature, nbins)

    # find in which bins belong each feature value
    wbin = np.digitize(feature, bins, right=True)

    if use_mean:
        norm_targets = targets / np.mean(targets)
        # I expect to see mean of empty slice --> return nan value --> ok
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meds = [np.mean(norm_targets[wbin == bin]) for bin in range(1, nbins + 1)]
    else:
        # build normalized targets : the normalization is done by the median density
        norm_targets = targets / np.nanmedian(targets)
        # digitization of the normalized target density values (first digitized bin is 1 not zero)
        meds = [np.median(norm_targets[wbin == bin]) for bin in range(1, nbins + 1)]

    # error for mean (estimation of the std from sample)
    err_meds = [np.std(norm_targets[wbin == bin]) / np.sqrt((wbin == bin).sum() - 1) if ((wbin == bin).sum() > 1) else np.NaN for bin in range(1, nbins + 1)]

    return bins, (bins[:-1] + bins[1:]) / 2, meds, nbr_obj_bins, err_meds


def _select_good_pixels(region, fracarea, pixels, footprint, cut_fracarea=True, limits_fracarea=(0.9, 1.1)):
    """
    Create pixel mask to only consider pixels in the correct region determined
    thanks to footprint with non zero fracarea and with a specific selection along fracarea if required with cut_fracarea.

    Returns
    -------
    pix_to_keep: bool array
        Which pixels will be kept for the plots.
    """

    # keep pixels in the correct phtometric region
    pix_to_keep = footprint(region)[pixels]

    # keep pixel with observations
    logger.info("Keep only pixels with fracarea > 0")
    pix_to_keep &= (fracarea > 0)

    # keep pixel with enough observations
    if cut_fracarea:
        # max value just due to poisson Noise
        logger.info(f"Keep only pixels with {limits_fracarea[0]} < fracarea < {limits_fracarea[1]}")
        pix_to_keep &= (fracarea > limits_fracarea[0]) & (fracarea < limits_fracarea[1])

    return pix_to_keep


def plot_systematic_from_map(map_list, label_list, fracarea, footprint, pixmaps, pixels, feature_names=None, regions=['North', 'South', 'Des'],
                             ax_lim=0.1, colors=None, figsize=(8.0, 5.2), adaptative_binning=False, nobj_per_bin=2000, n_bins=None,
                             cut_fracarea=True, limits_fracarea=(0.9, 1.1), legend_title=False, hist_legend=True, y_label=None,
                             save_table=False, save_table_suffix='',
                             show=False, save=True, savedir=None):
    """
    Create systematic plots (ie) relative density of pixel map as a function of observational features.
    Here only the following features are displayed : stardens, ebv, stream, psf_depth r,g,z,W1,W2, psf_size r,g,z.

    Parameters
    ----------
    map_list : float array list
        list containing healpix density map in nested scheme.
    label_list : str list
        list containing the label corresponding to the maps.
    fracarea : float array
        Healpix map of the corresponding fracarea (ie) observed fractional sky area for a pixel.
    footprint : :class:`Footprint`
        Corresponding footprint used to keep pixels in desired region.
    pixmaps : dict of DataFrame
        Array containing the heampix map at correct nside of dr9 features. Typically dataframe.features. Should contains one pixmaps for each region
    feature_names : list of str
        List of feature name contained in pixmaps that we want to plot.
    regions : str list
        regions where the systematic plots will be created. Each region is treat individually.
    ax_lim : float
        limit of the y axis (ie) of the relative density - 1.
    colors : list of str
        if not None, use the list as the colors for each map in map_list.
    adaptative_binning : bool
        If True, use same number of pixels in each bins.
    nobj_per_bin : int
        If adaptative_binning is True, fix the number of objects desired in each bin.
    n_bins : int
        If not None, will used as default value for nbins in each histogram.
    cut_fracarea : bool
        If True, apply addiational mask based on fracarea selection
    limits_fracarea : float tuple
        Used if cut_fracarea is True. Fix the bottom and up limit of the fracarea selection.
    legend_title : bool, default=False
        If True display on top of the legend the region name where the systematic plots are drawn.
    save_table : bool
        If true, save in .ecsv format the lines plotted as requiered by the DESI collaboration during the publication process.
    save_table_suffix : str
        If save_table is True, the line will be saved under `f'{save_table_suffix}{sysname}_{label}.ecsv'`
    show : bool
        If True, display the figures.
    save : bool
        If True, save the figures.
    savedir : str
        Path were the figures will be saved.
    """

    if feature_names is None:
        feature_names = ['STARDENS', 'EBV',
                         'PSFSIZE_G', 'PSFSIZE_R', 'PSFSIZE_Z',
                         'PSFDEPTH_G', 'PSFDEPTH_R', 'PSFDEPTH_Z',
                         'PSFDEPTH_W1', 'PSFDEPTH_W2']
    num_max_to_plot = len(feature_names)

    for region in regions:
        logger.info(f'Work with {region}')

        # select the correspongind pixmaps
        pixmap = pixmaps[region]

        sysdic = _get_desi_plot_attrs(feature_names, region)
        sysnames = list(sysdic.keys())

        # extract which pixels we will use to make the plot !
        pix_to_keep = _select_good_pixels(region, fracarea, pixels, footprint, cut_fracarea, limits_fracarea=limits_fracarea)

        # Plot photometric systematic plots:
        fig = plt.figure(1, figsize=figsize)
        gs = GridSpec(4, 3, figure=fig, left=0.1, right=0.92, bottom=0.08, top=0.98, hspace=0.55, wspace=0.25)

        num_to_plot = 0
        for i in range(12):
            if num_to_plot < num_max_to_plot:
                sysname = sysnames[num_to_plot]
                down, up, plotlabel, nbins, conversion = sysdic[sysname]
                if n_bins is not None:
                    nbins = n_bins

                if i not in [9]:
                    ax = fig.add_subplot(gs[i // 3, i % 3])
                    ax.set_xlabel(plotlabel)
                    ax.set_ylim(-ax_lim, ax_lim)
                    ax.set_yticks([-ax_lim, -ax_lim / 2, 0, ax_lim / 2, ax_lim])

                    for ii, (mp, label) in enumerate(zip(map_list, label_list)):
                        bins, binmid, meds, nbr_obj_bins, meds_err = _systematics_med(mp[pix_to_keep], conversion(pixmap[sysname][pix_to_keep].values),
                                                                                      sysname, downclip=conversion(down), upclip=conversion(up),
                                                                                      nbins=nbins, adaptative_binning=adaptative_binning, nobj_per_bin=nobj_per_bin)
                        color = None if colors is None else colors[ii]
                        ax.errorbar(binmid, meds - 1 * np.ones(binmid.size), yerr=meds_err, marker='.', markersize=3, linestyle='-', lw=0.8, color=color, label=label)

                        if save_table:
                            from astropy.table import Table
                            table = Table([binmid, meds, meds_err, nbr_obj_bins], names=['feature', 'relative_density', 'relative_density_err', 'nbr_objetcs_per_bins'])
                            table.write(f'{save_table_suffix}{sysname}_{label}.ecsv', overwrite=True)

                    ax_hist = ax.twinx()
                    ax_hist.set_xlim(ax.get_xlim())
                    ax_hist.set_ylim(ax.get_ylim())
                    if adaptative_binning:
                        normalisation = nbr_obj_bins / 0.1
                    else:
                        normalisation = nbr_obj_bins.sum()
                    ax_hist.bar(binmid, nbr_obj_bins / normalisation, alpha=0.4, color='dimgray', align='center', width=(bins[1:] - bins[:-1]), label='Fraction of no. objects\nby bin')
                    ax_hist.grid(False)
                    ax_hist.set_yticks([])

                    num_to_plot += 1

            if i == 3:
                if y_label is None: y_label = "Relative density - 1"
                ax.set_ylabel(y_label, labelpad=10)

            if i == 10:
                title = region if legend_title else None
                ax.legend(bbox_to_anchor=(-1.3, 0.9), loc='upper left', borderaxespad=0., frameon=False, ncol=1, title=title)
                if hist_legend:
                    ax_hist.legend(bbox_to_anchor=(-1.3, 0.2), loc='upper left', borderaxespad=0., frameon=False, ncol=1)

        if save:
            plt.savefig(os.path.join(savedir, f"{region}_systematics_plot.pdf"))
        if show:
            plt.show()
        else:
            plt.close()

        num_fig = 1
        # put 11 features on the first figure and then 12 for each additional figure
        if num_max_to_plot <= 11:
            num_fig_max = 1
        else:
            num_fig_max = 1 + int(np.ceil((num_max_to_plot - 11) / 12))

        while num_fig < num_fig_max:
            fig = plt.figure(num_fig, figsize=figsize)
            gs = GridSpec(4, 3, figure=fig, left=0.1, right=0.92, bottom=0.08, top=0.98, hspace=0.55, wspace=0.25)

            for i in range(12):
                if num_to_plot < num_max_to_plot:
                    sysname = sysnames[num_to_plot]
                    down, up, plotlabel, nbins, conversion = sysdic[sysname]
                    if n_bins is not None:
                        nbins = n_bins

                    ax = fig.add_subplot(gs[i // 3, i % 3])
                    ax.set_xlabel(plotlabel)
                    ax.set_ylim(-ax_lim, ax_lim)
                    ax.set_yticks([-ax_lim, -ax_lim / 2, 0, ax_lim / 2, ax_lim])

                    for ii, (mp, label) in enumerate(zip(map_list, label_list)):
                        bins, binmid, meds, nbr_obj_bins, meds_err = _systematics_med(mp[pix_to_keep], conversion(pixmap[sysname][pix_to_keep].values),
                                                                                      sysname, downclip=conversion(down), upclip=conversion(up),
                                                                                      nbins=nbins, adaptative_binning=adaptative_binning, nobj_per_bin=nobj_per_bin)
                        color = None if colors is None else colors[ii]
                        ax.errorbar(binmid, meds - 1 * np.ones(binmid.size), yerr=meds_err, marker='.', markersize=3, linestyle='-', lw=0.8, color=color, label=label)

                        if save_table:
                            from astropy.table import Table
                            table = Table([binmid, meds, meds_err, nbr_obj_bins], names=['feature', 'relative_density', 'relative_density_err', 'nbr_objetcs_per_bins'])
                            table.write(f'{save_table_suffix}{sysname}_{label}.ecsv', overwrite=True)

                    ax_hist = ax.twinx()
                    ax_hist.set_xlim(ax.get_xlim())
                    ax_hist.set_ylim(ax.get_ylim())
                    if adaptative_binning:
                        normalisation = nbr_obj_bins / 0.1
                    else:
                        normalisation = nbr_obj_bins.sum()
                    ax_hist.bar(binmid, nbr_obj_bins / normalisation, alpha=0.4, color='dimgray', align='center', width=(bins[1:] - bins[:-1]), label='Fraction of no. objects\nby bin')
                    ax_hist.grid(False)
                    ax_hist.set_yticks([])

                    num_to_plot += 1

                #if i == 3:
                #    ax.set_ylabel(y_label, labelpad=10)

            if save:
                plt.savefig(os.path.join(savedir, f"{region}_systematics_plot_2.pdf"))
            if show:
                plt.show()
            else:
                plt.close()

            num_fig += 1
