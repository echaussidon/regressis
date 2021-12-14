import os
import sys
if sys.path[0][:7] == '/global':
    local = False #we are on Nersc
elif sys.path[0][:4] == '/usr':
    local = True #we are in my mac

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
if local:
    plt.style.use('~/Documents/CEA/Software/desi_ec/ec_style.mplstyle')
else:
    plt.style.use('~/Software/desi_ec/ec_style.mplstyle')

import numpy as np
import healpy as hp
import fitsio
from astropy.table import Table

from desitarget.QA import _prepare_systematics
from desitarget.io import load_pixweight_recarray

from systematics import _load_systematics_old, _load_systematics, systematics_med
from fct_regressor import _load_targets

from desi_footprint import DR9_footprint

#------------------------------------------------------------------------------#
adaptative_binning = False
nobjects_by_bins = 2000

def _load_files(suffixe, suffixe_stars, release, version, Nside, fracarea_name, dir_to_save):

    pixel_area = hp.nside2pixarea(Nside, degrees=True)

    pixmap = load_pixweight_recarray(f"Data/pixweight-total-{release}-{Nside}.fits", nside=Nside)
    fracarea = pixmap[fracarea_name]

    targets, _, _ = _load_targets(suffixe_stars, release, version, Nside)
    targets /= (pixel_area*fracarea)

    targets_without_systematics = np.load(f"{dir_to_save}targets_without_systematics_pixel_{release}_{Nside}{suffixe}.npy") / (pixel_area*fracarea)

    return pixmap, targets, targets_without_systematics

#------------------------------------------------------------------------------#
def plot_systematics(dir_to_save="Res/Select_features/", zone_to_plot=['North', 'South', 'Des'],
                     suffixe='', release='dr8', version='SV3', Nside=256, fracarea_name='FRACAREA_12290', nbins=None, suffixe_stars='',
                     systematic_with_medhi=False, ax_lim=0.3, remove_LMC=True, clear_south=True):

    use_mean=True

    DR9 = DR9_footprint(Nside, remove_LMC=remove_LMC, clear_south=clear_south)

    pixmap_tot, targets_tot, targets_without_systematics_tot = _load_files(suffixe, suffixe_stars, release, version, Nside, fracarea_name, dir_to_save)

    sysdic = _load_systematics()
    sysnames = list(sysdic.keys())

    for key_word in zone_to_plot:
        if key_word == 'Global':
            pix_to_keep = DR9.load_footprint()
            key_word_sys = key_word
        elif key_word == 'North':
            pix_to_keep, _, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'South':
            _, pix_to_keep, _ = DR9.load_photometry()
            key_word_sys = key_word
        elif key_word == 'Des':
            _, _, pix_to_keep = DR9.load_photometry()
            key_word_sys = key_word
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
        elif key_word == 'Des_mid':
            _, _, south_pole_tmp = DR9.load_elg_region()
            _, _, des_tmp = DR9.load_elg_region()
            pix_to_keep = des_tmp & ~south_pole_tmp
            key_word_sys = 'Des'
        else:
            print("WRONG KEY WORD")
            sys.exit()

        pixmap = pixmap_tot[pix_to_keep]
        targets = targets_tot[pix_to_keep]
        targets_without_systematics = targets_without_systematics_tot[pix_to_keep]

        print("[TO DO] --> IL FAUT CHANGER CA !! on prend fracarea_12290 et on retire ceux qui sont en dessous 90%")
        fracarea = pixmap['FRACAREA_12290']

        #Dessin systematiques traditionnelles
        fig = plt.figure(1,figsize=(16.0,10.0))
        gs = GridSpec(4, 3, figure=fig, left=0.06, right=0.96, bottom=0.08, top=0.96, hspace=0.35 , wspace=0.20)

        num_to_plot = 0
        for i in range(12):
            sysname = sysnames[num_to_plot]
            d, u, plotlabel, nbins_tmp = sysdic[sysname][key_word_sys]
            if nbins is None:
                nbins = nbins_tmp
            down, up = _prepare_systematics(np.array([d, u]), sysname)
            feature =  _prepare_systematics(pixmap[sysname], sysname)
            if i not in [9]: #[2, 9]: #on affiche les systematics que l'on veut définies dans _load_systematics()
                ax = fig.add_subplot(gs[i//3, i%3])
                ax.set_xlabel(plotlabel)
                if i in [0, 3, 6]:
                    ax.set_ylabel("Relative QSO density - 1")
                ax.set_ylim([-ax_lim, ax_lim])

                _, binmid, meds, _, meds_err = systematics_med(targets, fracarea, feature, sysname, downclip=down, upclip=up, nbins=nbins, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins, use_mean=use_mean)
                ax.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.8, label='No correction')
                bins, binmid, meds, nbr_obj_bins, meds_err = systematics_med(targets_without_systematics, fracarea, feature, sysname, downclip=down, upclip=up, nbins=nbins, adaptative_binning=adaptative_binning, nobjects_by_bins=nobjects_by_bins, use_mean=use_mean)
                ax.errorbar(binmid, meds - 1*np.ones(binmid.size), yerr=meds_err, marker='.', linestyle='-', lw=0.8, label='Systematics correction')

                ax_hist = ax.twinx()
                ax_hist.set_xlim(ax.get_xlim())
                ax_hist.set_ylim(ax.get_ylim())
                if adaptative_binning:
                    normalisation = nbr_obj_bins/0.1
                else:
                    normalisation = nbr_obj_bins.sum()
                ax_hist.bar(binmid, nbr_obj_bins/normalisation, alpha=0.4, color='dimgray', align='center', width=(bins[1:] - bins[:-1]), label='Fraction of nbr objects by bin \n(after correction) ')
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

        plt.savefig(dir_to_save+"Systematics/{}_systematics_plot.pdf".format(key_word))
        plt.close()
