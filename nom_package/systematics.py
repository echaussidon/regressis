## coding: utf-8
## Author : Edmond Chaussidon (CEA)

import logging
logger = logging.getLogger("systematics")

import numpy as np
import warnings

import fitsio
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt

from desitarget.QA import _prepare_systematics
from desitarget.io import load_pixweight_recarray

from desi_footprint import DR9_footprint

def f(x) : return 22.5 - 2.5*np.log10(5/np.sqrt(x))

def g(y) : return 25*10**(2*(y - 22.5)/2.5)

def _load_systematics():
    sysdict = {}

    sysdict['STARDENS'] = {'North':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35],
                           'South':[150., 4000., 'log10(Stellar Density) per sq. deg.', 35],
                           'Des':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30],
                           'Global':[150., 4000., 'log10(Stellar Density) per sq. deg.', 30]}
    sysdict['EBV'] = {'North':[0.001, 0.1, 'E(B-V)', 35],
                      'South':[0.001, 0.1, 'E(B-V)', 35],
                      'Des':[0.001, 0.09, 'E(B-V)', 35],
                      'Global':[0.001, 0.1, 'E(B-V)', 30]}
    sysdict['STREAM'] = {'North':[0., 1., 'Sgr. Stream', 1],
                      'South':[0.01, 1., 'Sgr. Stream', 20],
                      'Des':[0.01, 0.6, 'Sgr. Stream', 15],
                      'Global':[0.01, 1.5, 'Sgr. Stream', 20]}

    sysdict['PSFSIZE_G'] = {'North':[1.3, 2.6, 'PSF Size in g-band', 35],
                            'South':[1.1, 2.02, 'PSF Size in g-band', 35],
                            'Des':[1.19, 1.7, 'PSF Size in g-band',30],
                            'Global':[0., 3., 'PSF Size in g-band', 30]}
    sysdict['PSFSIZE_R'] ={'North':[1.25, 2.4, 'PSF Size in r-band', 40],
                           'South':[0.95, 1.92, 'PSF Size in r-band', 30],
                           'Des':[1.05, 1.5, 'PSF Size in r-band',30],
                           'Global':[0., 3., 'PSF Size in r-band', 30]}
    sysdict['PSFSIZE_Z'] = {'North':[0.9, 1.78, 'PSF Size in z-band', 35],
                            'South':[0.9, 1.85, 'PSF Size in z-band', 40],
                            'Des':[0.95, 1.4, 'PSF Size in z-band', 30],
                            'Global':[0., 3., 'PSF Size in z-band', 30]}

    sysdict['PSFDEPTH_G'] = {'North':[300., 1600., 'PSF Depth in g-band', 30],
                             'South':[750., 4000., 'PSF Depth in g-band', 35],
                             'Des':[1900., 7000., 'PSF Depth in g-band', 30],
                             'Global':[63., 6300., 'PSF Depth in g-band', 30]}
    sysdict['PSFDEPTH_R'] = {'North':[95., 620., 'PSF Depth in r-band', 30],
                             'South':[260.0, 1600.0, 'PSF Depth in r-band', 30],
                             'Des':[1200., 5523., 'PSF Depth in r-band', 30],
                             'Global':[25., 2500., 'PSF Depth in r-band', 30]}
    sysdict['PSFDEPTH_Z'] ={'North':[60., 275., 'PSF Depth in z-band', 40],
                            'South':[40.0, 360., 'PSF Depth in z-band', 40],
                            'Des':[145., 570., 'PSF Depth in z-band', 30],
                            'Global':[4., 400., 'PSF Depth in z-band', 30]}

    sysdict['PSFDEPTH_W1'] = {'North':[2.7, 12., 'PSF Depth in W1-band', 40],
                              'South':[2.28, 5.5, 'PSF Depth in W1-band', 30],
                              'Des':[2.28, 6.8, 'PSF Depth in W1-band', 30],
                              'Global':[0.0, 30.0, 'PSF Depth in W1-band', 30]}
    sysdict['PSFDEPTH_W2'] = {'North':[0.8, 3.9, 'PSF Depth in W2-band', 40],
                              'South':[0.629, 1.6, 'PSF Depth in W2-band', 30],
                              'Des':[0.62, 2.25, 'PSF Depth in W2-band', 30],
                              'Global':[0.0, 7.0, 'PSF Depth in W2-band', 30]}
    return sysdict

def systematics_med(targets, fracarea, feature, feature_name, downclip=None, upclip=None, nbins=50, use_mean=True, adaptative_binning=False, nobjects_by_bins=1000):
    """
    Return binmid, meds pour pouvoir faire : plt.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.9)
    """

    # Selection of pixels with correct fracarea and which respect the up/downclip for the 'feature_name' feature
    sel = (fracarea < 1.1) & (fracarea > 0.9) & (feature >= downclip) & (feature < upclip)
    if not np.any(sel):
        logger.info("Pixel map has no areas (with >90% coverage) with the up/downclip")
        logger.info("Proceeding without clipping systematics for {}".format(feature_name))
        sel = (fracarea < 1.1) & (fracarea > 0.9)
    # keep only pixels with targets / observations
    # --> we are not in the case where there is only one objects per pixels otherwise decrease Nside
    sel &= (targets > 0)
    targets = targets[sel]
    feature = feature[sel]

    if adaptative_binning: #set this option to have variable bin sizes so that each bin contains the same number of pixel (nobjects_by_bins)
        nbr_obj_bins = nobjects_by_bins
        ksort = np.argsort(feature)
        bins=feature[ksort[0::nbr_obj_bins]] #Here, get the bins from the data set (needed to be sorted)
        bins=np.append(bins,feature[ksort[-1]]) #add last point
        nbins = bins.size - 1 #OK
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





################################################################################
################################################################################



def plot_systematic_from_map(map_list, label_list, savedir='', suffixe='', zone_to_plot=['North', 'South', 'Des'], Nside=256, remove_LMC=False, clear_south=True,
                            pixweight_path='/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.1/pixweight/main/resolve/dark/pixweight-1-dark.fits',
                            sgr_stream_path='/global/homes/e/edmondc/Systematics/regressor/Sagittarius_Stream/sagittarius_stream_256.npy',
                            ax_lim=0.3, adaptative_binning=False, nobjects_by_bins=2000, show=False, save=True,
                            n_bins=None, correct_map_with_fracarea=True):

    #load DR9 region
    DR9 = DR9_footprint(Nside, remove_LMC=remove_LMC, clear_south=clear_south)

    # load pixweight file and Sgr. map at correct Nside
    pixmap_tot = load_pixweight_recarray(pixweight_path, nside=Nside)
    sgr_stream_tot = np.load(sgr_stream_path)
    if Nside != 256:
        sgr_stream_tot = hp.ud_grade(sgr_stream_tot, Nside, order_in=True)

    if correct_map_with_fracarea:
        # correct map with the fracarea (for maskbit 1, 12, 13)
        with np.errstate(divide='ignore'): # Ok --> on n'utilise pas les pixels qui n'ont pas été observé, ils sont en-dehors du footprint
            logger.info("Correct maps with FRACAREA_12290 (Maskbit 1, 12, 13)")
            map_list_tot = [mp/pixmap_tot['FRACAREA_12290'] for mp in map_list]
    else:
        logger.info("MAPS ARE NOT CORRECTED with fracarea_12290 (fracarea_12290 is only used in sysmeds to keep pixels which are reliable in term of DR9")
        map_list_tot = map_list

    sysdic = _load_systematics()
    sysnames = list(sysdic.keys())

    for num_fig, key_word in enumerate(zone_to_plot):
        logger.info(f'Work with {key_word}')
        if key_word == 'Global':
            pix_to_keep = DR9.load_footprint()
            key_word_sys = key_word
        elif key_word == 'North':
            pix_to_keep, _, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South':
            _, pix_to_keep, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South_ngc':
            _, pix_to_keep, _ = DR9.load_photometry()
            ngc, _ = DR9.load_ngc_sgc()
            pix_to_keep &= ngc
            key_word_sys = 'South'
        elif key_word == 'South_sgc':
            _, pix_to_keep, _ = DR9.load_photometry()
            _, sgc = DR9.load_ngc_sgc()
            pix_to_keep &= sgc
            key_word_sys = 'South'
        elif key_word == 'Des':
            _, _, pix_to_keep = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'Des_mid':
            _, _, pix_to_keep = DR9.load_photometry()
            _, _, _, tmp = DR9.load_elg_region(ngc_sgc_split=True)
            pix_to_keep = pix_to_keep & ~tmp
            key_word_sys = 'Des'
        elif key_word == 'South_mid':
            _, pix_to_keep, _ = DR9.load_elg_region()
            key_word_sys = 'South'
        elif key_word == 'South_mid_ngc':
            _, pix_to_keep, _, _ = DR9.load_elg_region(ngc_sgc_split=True)
            key_word_sys = 'South'
        elif key_word == 'South_mid_sgc':
            _, _, pix_to_keep, _ = DR9.load_elg_region(ngc_sgc_split=True)
            key_word_sys = 'South'
        elif key_word == 'South_pole':
            _, _, pix_to_keep = DR9.load_elg_region()
            key_word_sys = 'Des'
        else:
            print("WRONG KEY WORD")
            sys.exit()

        pixmap = pixmap_tot[pix_to_keep]
        sgr_stream = sgr_stream_tot[pix_to_keep]
        map_list = [mp[pix_to_keep] for mp in map_list_tot]
        fracarea = pixmap['FRACAREA_12290']

        #Dessin systematiques traditionnelles
        fig = plt.figure(num_fig, figsize=(16.0,10.0))
        gs = GridSpec(4, 3, figure=fig, left=0.06, right=0.96, bottom=0.08, top=0.96, hspace=0.35 , wspace=0.20)

        num_to_plot = 0
        for i in range(12):
            sysname = sysnames[num_to_plot]
            d, u, plotlabel, nbins = sysdic[sysname][key_word_sys]
            if n_bins is not None:
                nbins=n_bins
            down, up = _prepare_systematics(np.array([d, u]), sysname)

            if sysname == 'STREAM':
                feature = _prepare_systematics(sgr_stream, sysname)
            else:
                feature =  _prepare_systematics(pixmap[sysname], sysname)

            if i not in [9]: #[2, 9]: #on affiche les systematics que l'on veut définies dans _load_systematics()
                ax = fig.add_subplot(gs[i//3, i%3])
                ax.set_xlabel(plotlabel)
                if i in [0, 3, 6]:
                    ax.set_ylabel("Relative density - 1")
                ax.set_ylim([-ax_lim, ax_lim])

                for mp, label in zip(map_list, label_list):
                    bins, binmid, meds, nbr_obj_bins, meds_err = systematics_med(mp, fracarea, feature, sysname, downclip=down, upclip=up, nbins=nbins, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins)
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
             #   ax.text(0.5, 0.5, f"Zone : {key_word}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                #on enleve le STREAM comme feature ok
             #   num_to_plot += 1

            if i==10:
                ax.legend(bbox_to_anchor=(-1.1, 0.8), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large')
                ax_hist.legend(bbox_to_anchor=(-1.1, 0.35), loc='upper left', borderaxespad=0., frameon=False, ncol=1, fontsize='large')

        if save:
            plt.savefig(savedir+"{}_systematics_plot_{}.pdf".format(key_word, suffixe))
        if show:
            plt.show()
        else:
            plt.close()
