import numpy as np
import healpy as hp

from desitarget.io import read_targets_in_box

Nside_list = [256, 512]

# print(f"\n[INFO] Collect SV3 target in Bright time to build pixmap with Nside={Nside_list}...")
# objects = read_targets_in_box("/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/bright/", [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', 'SV3_DESI_TARGET', 'SV3_BGS_TARGET'])
# desi_target = objects['SV3_DESI_TARGET'][:]
# bgs_target = objects['SV3_BGS_TARGET'][:]
#
# for Nside in Nside_list:
#     print("    * BGS_ANY (bit mask == 60)")
#     sel_bgs = ((desi_target&2**60)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'SV3_BGS_ANY_targets_{Nside}.npy', bgs)
#
#     print("    * BGS_FAINT (bit mask == 0 in SV3_BGS_TARGET)")
#     sel_bgs = ((bgs_target&2**0)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'SV3_BGS_FAINT_targets_{Nside}.npy', bgs)
#
#     print("    * BGS_BRIGHT (bit mask == 1 in SV3_BGS_TARGET )")
#     sel_bgs = ((bgs_target&2**1)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'SV3_BGS_BRIGHT_targets_{Nside}.npy', bgs)
#
# print(f"\n[INFO] Collect SV3 target in Dark time to build pixmap with Nside={Nside_list}...")
# objects = read_targets_in_box("/global/cfs/cdirs/desi/target/catalogs/dr9/0.57.0/targets/sv3/resolve/dark/", [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', 'SV3_DESI_TARGET'])
# desi_target = objects['SV3_DESI_TARGET'][:]
#
# for Nside in Nside_list:
#     print("    * LRG (bit mask == 0)")
#     sel_lrg = ((desi_target&2**0)!=0)
#     lrg = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_lrg]['RA'][:], objects[sel_lrg]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     lrg[pix] = counts
#     np.save(f'SV3_LRG_targets_{Nside}.npy', lrg)
#
#     print("    * LRG_LOWDENS (bit mask == 3)")
#     sel_lrg = ((desi_target&2**3)!=0)
#     lrg = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_lrg]['RA'][:], objects[sel_lrg]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     lrg[pix] = counts
#     np.save(f'SV3_LRG_LOWDENS_targets_{Nside}.npy', lrg)
#
#     print("    * ELG_HIP (bit mask == 6)")
#     sel_elg = ((desi_target&2**6)!=0)
#     elg = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     elg[pix] = counts
#     np.save(f'SV3_ELG_HIP_targets_{Nside}.npy', elg)
#
#     print("    * ELG_LOP (bit mask == 5)")
#     sel_elg = ((desi_target&2**5)!=0)
#     elg = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     elg[pix] = counts
#     np.save(f'SV3_ELG_LOP_targets_{Nside}.npy', elg)
#
#     print("    * ELG (bit mask == 1)")
#     sel_elg = ((desi_target&2**1)!=0)
#     elg = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     elg[pix] = counts
#     np.save(f'SV3_ELG_targets_{Nside}.npy', elg)
#
#     print("    * QSO (bit mask == 2)")
#     sel_qso = ((desi_target&2**2)!=0)
#     qso = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_qso]['RA'][:], objects[sel_qso]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     qso[pix] = counts
#     np.save(f'SV3_QSO_targets_{Nside}.npy', qso)


# print(f"\n[INFO] Collect MAIN target in Bright time to build pixmap with Nside={Nside_list}...")
# objects = read_targets_in_box("/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/bright/", [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', 'DESI_TARGET', 'BGS_TARGET'])
# desi_target = objects['DESI_TARGET'][:]
# bgs_target = objects['BGS_TARGET'][:]
#
# for Nside in Nside_list:
#     print("    * BGS_ANY (bit mask == 60)")
#     sel_bgs = ((desi_target&2**60)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'MAIN_BGS_ANY_targets_{Nside}.npy', bgs)
#
#     print("    * BGS_FAINT (bit mask == 0 in SV3_BGS_TARGET)")
#     sel_bgs = ((bgs_target&2**0)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'MAIN_BGS_FAINT_targets_{Nside}.npy', bgs)
#
#     print("    * BGS_BRIGHT (bit mask == 1 in SV3_BGS_TARGET )")
#     sel_bgs = ((bgs_target&2**1)!=0)
#     bgs = np.zeros(hp.nside2npix(Nside))
#     pixels = hp.ang2pix(Nside, objects[sel_bgs]['RA'][:], objects[sel_bgs]['DEC'][:], nest=True, lonlat=True)
#     pix, counts = np.unique(pixels, return_counts=True)
#     bgs[pix] = counts
#     np.save(f'MAIN_BGS_BRIGHT_targets_{Nside}.npy', bgs)

print(f"\n[INFO] Collect MAIN target in Dark time to build pixmap with Nside={Nside_list}...")
objects = read_targets_in_box("/global/cfs/cdirs/desi/target/catalogs/dr9/1.1.0/targets/main/resolve/dark/", [0, 360, -90, 90], quick=True, columns=['RA', 'DEC', 'DESI_TARGET'])
desi_target = objects['DESI_TARGET'][:]

for Nside in Nside_list:
    print("    * LRG (bit mask == 0)")
    sel_lrg = ((desi_target&2**0)!=0)
    lrg = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, objects[sel_lrg]['RA'][:], objects[sel_lrg]['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    lrg[pix] = counts
    np.save(f'MAIN_LRG_targets_{Nside}.npy', lrg)

    print("    * ELG_LOP (bit mask == 5)")
    sel_elg = ((desi_target&2**5)!=0)
    elg = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    elg[pix] = counts
    np.save(f'MAIN_ELG_LOP_targets_{Nside}.npy', elg)

    print("    * ELG_HIP (bit mask == 6)")
    sel_elg = ((desi_target&2**6)!=0)
    elg = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    elg[pix] = counts
    np.save(f'MAIN_ELG_HIP_targets_{Nside}.npy', elg)

    print("    * ELG_VLO (bit mask == 7)")
    sel_elg = ((desi_target&2**7)!=0)
    elg = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    elg[pix] = counts
    np.save(f'MAIN_ELG_VLO_targets_{Nside}.npy', elg)

    print("    * ELG (bit mask == 1)")
    sel_elg = ((desi_target&2**1)!=0)
    elg = np.zeros(hp.nside2npix(Nside))
    pixels = hp.ang2pix(Nside, objects[sel_elg]['RA'][:], objects[sel_elg]['DEC'][:], nest=True, lonlat=True)
    pix, counts = np.unique(pixels, return_counts=True)
    elg[pix] = counts
    np.save(f'MAIN_ELG_targets_{Nside}.npy', elg)

    # print("    * QSO (bit mask == 2)")
    # sel_qso = ((desi_target&2**2)!=0)
    # qso = np.zeros(hp.nside2npix(Nside))
    # pixels = hp.ang2pix(Nside, objects[sel_qso]['RA'][:], objects[sel_qso]['DEC'][:], nest=True, lonlat=True)
    # pix, counts = np.unique(pixels, return_counts=True)
    # qso[pix] = counts
    # np.save(f'MAIN_QSO_targets_{Nside}.npy', qso)
