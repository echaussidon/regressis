#!/usr/bin/env python
# coding: utf-8

import os
import warnings
import logging

import numpy as np
import fitsio

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

logger = logging.getLogger("systematics")


def _load_systematics_desi():
    """
    These cuts are adapted for the DESI Legacy Imaging Survey DR9. Return dictionary with feature name as label of dictionary for specific photometric regions containing [min, max, label, nbins, function_to_convert].
    """

    def id(x): return x
    def convert_depth(x) : return 22.5 - 2.5*np.log10(5/np.sqrt(x))
    def convert_stardens(x): return np.log10(x)

    sysdict = {}
    sysdict['STARDENS'] = {'North':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35, convert_stardens],
                           'South':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35, convert_stardens],
                           'Des':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30, convert_stardens],
                           'Global':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30, convert_stardens]}
    sysdict['EBV'] = {'North':[0.001, 0.1, 'E(B-V)', 35, id],
                      'South':[0.001, 0.1, 'E(B-V)', 35, id],
                      'Des':[0.001, 0.09, 'E(B-V)', 35, id],
                      'Global':[0.001, 0.1, 'E(B-V)', 30, id]}
    sysdict['STREAM'] = {'North':[0., 1., 'Sgr. Stream', 1, id],
                      'South':[0.01, 1., 'Sgr. Stream', 20, id],
                      'Des':[0.01, 0.6, 'Sgr. Stream', 15, id],
                      'Global':[0.01, 1.5, 'Sgr. Stream', 20, id]}

    sysdict['PSFSIZE_G'] = {'North':[1.3, 2.6, 'PSF Size in g-band', 35, id],
                            'South':[1.1, 2.02, 'PSF Size in g-band', 35, id],
                            'Des':[1.19, 1.7, 'PSF Size in g-band',30, id],
                            'Global':[0., 3., 'PSF Size in g-band', 30, id]}
    sysdict['PSFSIZE_R'] ={'North':[1.25, 2.4, 'PSF Size in r-band', 40, id],
                           'South':[0.95, 1.92, 'PSF Size in r-band', 30, id],
                           'Des':[1.05, 1.5, 'PSF Size in r-band',30, id],
                           'Global':[0., 3., 'PSF Size in r-band', 30, id]}
    sysdict['PSFSIZE_Z'] = {'North':[0.9, 1.78, 'PSF Size in z-band', 35, id],
                            'South':[0.9, 1.85, 'PSF Size in z-band', 40, id],
                            'Des':[0.95, 1.4, 'PSF Size in z-band', 30, id],
                            'Global':[0., 3., 'PSF Size in z-band', 30, id]}

    sysdict['PSFDEPTH_G'] = {'North':[300., 1600., 'PSF Depth in g-band', 30, convert_depth],
                             'South':[750., 4000., 'PSF Depth in g-band', 35, convert_depth],
                             'Des':[1900., 7000., 'PSF Depth in g-band', 30, convert_depth],
                             'Global':[63., 6300., 'PSF Depth in g-band', 30, convert_depth]}
    sysdict['PSFDEPTH_R'] = {'North':[95., 620., 'PSF Depth in r-band', 30, convert_depth],
                             'South':[260.0, 1600.0, 'PSF Depth in r-band', 30, convert_depth],
                             'Des':[1200., 5523., 'PSF Depth in r-band', 30, convert_depth],
                             'Global':[25., 2500., 'PSF Depth in r-band', 30, convert_depth]}
    sysdict['PSFDEPTH_Z'] = {'North':[60., 275., 'PSF Depth in z-band', 40, convert_depth],
                             'South':[40.0, 360., 'PSF Depth in z-band', 40, convert_depth],
                             'Des':[145., 570., 'PSF Depth in z-band', 30, convert_depth],
                             'Global':[4., 400., 'PSF Depth in z-band', 30, convert_depth]}

    sysdict['PSFDEPTH_W1'] = {'North':[2.7, 12., 'PSF Depth in W1-band', 40, convert_depth],
                              'South':[2.28, 5.5, 'PSF Depth in W1-band', 30, convert_depth],
                              'Des':[2.28, 6.8, 'PSF Depth in W1-band', 30, convert_depth],
                              'Global':[0.0, 30.0, 'PSF Depth in W1-band', 30, convert_depth]}
    sysdict['PSFDEPTH_W2'] = {'North':[0.8, 3.9, 'PSF Depth in W2-band', 40, convert_depth],
                              'South':[0.629, 1.6, 'PSF Depth in W2-band', 30, convert_depth],
                              'Des':[0.62, 2.25, 'PSF Depth in W2-band', 30, convert_depth],
                              'Global':[0.0, 7.0, 'PSF Depth in W2-band', 30, convert_depth]}
    return sysdict


def _systematics_med(targets, feature, feature_name, downclip=None, upclip=None, nbins=50, use_mean=True, adaptative_binning=False, nobjects_by_bins=1000):
    """
    Return binmid, meds pour pouvoir faire : plt.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.9)
    """

    sel = (feature >= downclip) & (feature < upclip)
    if not np.any(sel):
        logger.info("Proceeding without clipping systematics for {}".format(feature_name))
    targets = targets[sel]
    feature = feature[sel]

    if adaptative_binning: # set this option to have variable bin sizes so that each bin contains the same number of pixel (nobjects_by_bins)
        nbr_obj_bins = nobjects_by_bins
        ksort = np.argsort(feature)
        bins=feature[ksort[0::nbr_obj_bins]] # Here, get the bins from the data set (needed to be sorted)
        bins=np.append(bins,feature[ksort[-1]]) # add last point
        nbins = bins.size - 1 # OK
    else: # create bin with fix size (depends on the up/downclip value or minimal / maximal value of feature if up/downclip is too large)
        nbr_obj_bins, bins = np.histogram(feature, nbins)

    # find in which bins belong each feature value
    wbin = np.digitize(feature, bins, right=True)

    if use_mean:
        norm_targets = targets/np.mean(targets)
        # I expect to see mean of empty slice --> return nan value --> ok
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            meds = [np.mean(norm_targets[wbin == bin]) for bin in range(1, nbins+1)]
    else:
        # build normalized targets : the normalization is done by the median density
        norm_targets = targets/np.nanmedian(targets)
        # digitization of the normalized target density values (first digitized bin is 1 not zero)
        meds = [np.median(norm_targets[wbin == bin]) for bin in range(1, nbins+1)]

    # error for mean (estimation of the std from sample)
    err_meds = [np.std(norm_targets[wbin == bin]) / np.sqrt((wbin == bin).sum() - 1) if ((wbin == bin).sum() > 1) else np.NaN for bin in range(1, nbins+1)]

    return bins, (bins[:-1] + bins[1:])/ 2, meds, nbr_obj_bins, err_meds


def _select_good_pixels(keyword, fracarea, footprint, cut_fracarea=True, min_fracarea=0.9, max_fracarea=1.1):
    """


    Parameter:
    ----------




    Returns:
    --------
    pix_to_keep: bool array
        Which pixels will be kept to plot the systematic plots
    keyword_sys: str
        With which param the systematic plots will be plotted.
    """

    pix_to_keep = footprint(keyword)
    keyword_sys = footprint.systematic_key_for_region(keyword)

    # keep pixel with observations
    logger.info("Keep only pixels with fracarea > 0")
    pix_to_keep &= (fracarea > 0)

    # keep pixel with enough observations
    if cut_fracarea:
        # max value just due to poisson Noise
        logger.info(f"Keep only pixels with {min_fracarea} < fracarea < {max_fracarea}")
        pix_to_keep &= (fracarea < max_fracarea) & (fracarea > min_fracarea)

    return pix_to_keep, keyword_sys


def plot_systematic_from_map(map_list, label_list, fracarea, footprint, pixmap, savedir, zone_to_plot=['North', 'South', 'Des'],
                             ax_lim=0.3, adaptative_binning=False, nobjects_by_bins=2000, n_bins=None, cut_fracarea=True, min_fracarea=0.9, max_fracarea=1.1,
                             show=False, save=True):

    sysdic = _load_systematics_desi()
    sysnames = list(sysdic.keys())

    for num_fig, keyword in enumerate(zone_to_plot):
        logger.info(f'Work with {keyword}')

        # extract which pixels we will use to make the plot !
        pix_to_keep, keyword_sys = _select_good_pixels(keyword, fracarea, footprint, cut_fracarea, min_fracarea, max_fracarea)

        # Plot photometric systematic plots:
        fig = plt.figure(num_fig, figsize=(16.0,10.0))
        gs = GridSpec(4, 3, figure=fig, left=0.06, right=0.96, bottom=0.08, top=0.96, hspace=0.35 , wspace=0.20)

        num_to_plot = 0
        for i in range(12):
            sysname = sysnames[num_to_plot]
            down, up, plotlabel, nbins, conversion = sysdic[sysname][keyword_sys]
            if n_bins is not None:
                nbins=n_bins

            if i not in [9]:
                ax = fig.add_subplot(gs[i//3, i%3])
                ax.set_xlabel(plotlabel)
                if i in [0, 3, 6]:
                    ax.set_ylabel("Relative density - 1")
                ax.set_ylim([-ax_lim, ax_lim])

                for mp, label in zip(map_list, label_list):
                    bins, binmid, meds, nbr_obj_bins, meds_err = _systematics_med(mp[pix_to_keep], conversion(pixmap[sysname][pix_to_keep]), sysname, downclip=conversion(down), upclip=conversion(up), nbins=nbins, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins)
                    ax.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.8, label=label)

                ax_hist = ax.twinx()
                ax_hist.set_xlim(ax.get_xlim())
                ax_hist.set_ylim(ax.get_ylim())
                if adaptative_binning:
                    normalisation = nbr_obj_bins/0.1
                else:
                    normalisation = nbr_obj_bins.sum()
                ax_hist.bar(binmid, nbr_obj_bins/normalisation, alpha=0.4, color='dimgray', align='center', width=(bins[1:] - bins[:-1]), label='Fraction of nbr objects by bin')
                ax_hist.grid(False)
                ax_hist.set_yticks([])

                num_to_plot += 1

            #if i==2:
             #   ax = fig.add_subplot(gs[i//3, i%3])
             #   ax.axis("off")
             #   ax.text(0.5, 0.5, f"Zone : {keyword}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                #on enleve le STREAM comme feature ok
             #   num_to_plot += 1

            if i==10:
                ax.legend(bbox_to_anchor=(-1.1, 0.8), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large', title=keyword, title_fontsize='large')
                ax_hist.legend(bbox_to_anchor=(-1.1, 0.35), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large')

        if save:
            plt.savefig(os.path.join(savedir, f"{keyword}_systematics_plot.pdf"))
        if show:
            plt.show()
        else:
            plt.close()
