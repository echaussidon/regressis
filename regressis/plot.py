#!/usr/bin/env python
# coding: utf-8

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_transpose

# to avoid this warning:
# WARNING: AstropyDeprecationWarning: Transforming a frame instance to a frame class (as opposed to another frame instance)
# will not be supported in the future.  Either explicitly instantiate the target frame, or first convert the source frame instance
# to a `astropy.coordinates.SkyCoord` and use its `transform_to()` method. [astropy.coordinates.baseframe]
import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)


class Sagittarius(coord.BaseCoordinateFrame):
    """
    Basic astropy coordiante class for Sagittarius coordinates.
    Reference: https://docs.astropy.org/en/stable/generated/examples/coordinates/plot_sgr-coordinate-frame.html
    """
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]}


def SGR_MATRIX():
    """Build the transformation matric from Galactic spherical to heliocentric Sgr coordinates based on Law & Majewski 2010."""
    SGR_PHI = (180 + 3.75) * u.degree  # Euler angles (from Law & Majewski 2010)
    SGR_THETA = (90 - 13.46) * u.degree
    SGR_PSI = (180 + 14.111534) * u.degree

    # Generate the rotation matrix using the x-convention (see Goldstein)
    D = rotation_matrix(SGR_PHI, "z")
    C = rotation_matrix(SGR_THETA, "x")
    B = rotation_matrix(SGR_PSI, "z")
    A = np.diag([1., 1., -1.])

    # https://github.com/astropy/astropy/issues/16943
    #from astropy.coordinates.matrix_utilities import matrix_product
    #SGR_matrix = matrix_product(A, B, C, D)

    SGR_matrix = A @ B @ C @ D
    return SGR_matrix


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """Compute the transformation matrix from Galactic spherical to heliocentric Sgr coordinates."""
    return SGR_MATRIX()


@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """Compute the transformation matrix from heliocentric Sgr coordinates to spherical Galactic."""
    return matrix_transpose(SGR_MATRIX())


def add_galactic_plane(ax, rot=120):
    """
    Galactic plane in ircs coordinates.

    Parameters
    ----------
    rot : float
        Rotation of the R.A. axis for sky visualisation plot. In DESI, it should be rot=120.

    Returns
    -------
    ra : float array
        Ordering (i.e. can use directly plt.plot(ra, dec with ls='-')) array containing R.A. values of the galactic plane in IRCS coordinates.
    dec : float array
        Ordering array containing Dec. values of the galactic plane in IRCS coordinates.
    """
    galactic_plane_tmp = SkyCoord(l=np.linspace(0, 2 * np.pi, 200) * u.radian, b=np.zeros(200) * u.radian, frame='galactic', distance=1 * u.Mpc)
    galactic_plane_icrs = galactic_plane_tmp.transform_to('icrs')

    ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    # get the correct order from ra=-180 to ra=180 after rotation
    index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    ax.plot(np.radians(ra[index_galactic]), np.radians(dec[index_galactic]), linestyle='-', linewidth=0.8, color='black', label='Galactic plane', zorder=10)

    return ra[index_galactic], dec[index_galactic]

def add_ecliptic_plane(ax, rot=120):
    """ Same than _get_galactic_coordinates but for the ecliptic plane in IRCS coordiantes"""
    ecliptic_plane_tmp = SkyCoord(lon=np.linspace(0, 2 * np.pi, 200) * u.radian, lat=np.zeros(200) * u.radian, distance=1 * u.Mpc, frame='heliocentrictrueecliptic')
    ecliptic_plane_icrs = ecliptic_plane_tmp.transform_to('icrs')

    ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    ax.plot(np.radians(ra[index_ecliptic]), np.radians(dec[index_ecliptic]), linestyle=':', linewidth=0.8, color='slategrey', label='Ecliptic plane', zorder=10)

    return ra[index_ecliptic], dec[index_ecliptic]

def add_sgr_plane(ax, rot=120):
    """ Same than _get_galactic_coordinates but for the Sagittarius Galactic plane in IRCS coordiantes"""
    sgr_plane_tmp = Sagittarius(Lambda=np.linspace(0, 2 * np.pi, 200) * u.radian, Beta=np.zeros(200) * u.radian, distance=1 * u.Mpc)
    sgr_plane_icrs = sgr_plane_tmp.transform_to(coord.ICRS)

    ra, dec = sgr_plane_icrs.ra.degree - rot, sgr_plane_icrs.dec.degree
    ra[ra > 180] -= 360    # scale conversion to [-180, 180]
    ra = -ra               # reverse the scale: East to the left

    index_sgr = np.argsort(sgr_plane_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    ax.plot(np.radians(ra[index_sgr]), np.radians(dec[index_sgr]), linestyle='--', linewidth=0.8, color='navy', label='Sgr. plane', zorder=10)

    return ra[index_sgr], dec[index_sgr]

def add_sgr_stream(ax, rot=120):
    """ Same than _get_galactic_coordinates but for the bottom and top line of the Sgr. Stream in IRCS coordiantes"""
    sgr_stream_top_tmp = Sagittarius(Lambda=np.linspace(0, 2 * np.pi, 200) * u.radian, Beta=20 * np.pi / 180 * np.ones(200) * u.radian, distance=1 * u.Mpc)
    sgr_stream_top_icrs = sgr_stream_top_tmp.transform_to(coord.ICRS)

    ra_top, dec_top = sgr_stream_top_icrs.ra.degree - rot, sgr_stream_top_icrs.dec.degree
    ra_top[ra_top > 180] -= 360    # scale conversion to [-180, 180]
    ra_top = -ra_top               # reverse the scale: East to the left

    index_sgr_top = np.argsort(sgr_stream_top_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    sgr_stream_bottom_tmp = Sagittarius(Lambda=np.linspace(0, 2 * np.pi, 200) * u.radian, Beta=-15 * np.pi / 180 * np.ones(200) * u.radian, distance=1 * u.Mpc)
    sgr_stream_bottom_icrs = sgr_stream_bottom_tmp.transform_to(coord.ICRS)

    ra_bottom, dec_bottom = sgr_stream_bottom_icrs.ra.degree - rot, sgr_stream_bottom_icrs.dec.degree
    ra_bottom[ra_bottom > 180] -= 360    # scale conversion to [-180, 180]
    ra_bottom = -ra_bottom               # reverse the scale: East to the left

    index_sgr_bottom = np.argsort(sgr_stream_bottom_icrs.ra.wrap_at((180 + rot) * u.deg).degree)

    ax.plot(np.radians(ra[index_sgr_bottom]), np.radians(dec[index_sgr_bottom]), linestyle=':', linewidth=0.8, color='navy', zorder=10)
    ax.plot(np.radians(ra[index_sgr_top]), np.radians(dec[index_sgr_top]), linestyle=':', linewidth=0.8, color='navy', zorder=10)


def add_desi_footprint(ax, rot=120):
    from pathlib import Path
    from astropy.table import Table
    d = Table.read(Path(__file__).parent / "data/desi-14k-footprint-dark.ecsv")
    for cap in ["NGC", "SGC"]:
        sel = d["CAP"] == cap

        ra, dec = d["RA"][sel] - rot, d["DEC"][sel]
        ra[ra > 180] -= 360    # scale conversion to [-180, 180]
        ra = -ra               # reverse the scale: East to the left

        _ = ax.plot(np.radians(ra), np.radians(dec), color='black', lw=1, label='DESI' if cap == 'NGC' else None, zorder=1)
        #ax.add_patch(Polygon(np.array([utils.projection_ra(d["RA"][sel], ra_center=rot), utils.projection_dec(d["DEC"][sel])]).T, facecolor='darkblue', alpha=0.2))
    #ax.plot(utils.projection_ra(d["RA"][sel], ra_center=rot), utils.projection_dec(d["DEC"][sel]), color='darkblue', lw=1.5, zorder=10, label='DESI')


def add_desi_ext_footprint(ax, rot=120):
    from pathlib import Path
    import pandas as pd
    from matplotlib.patches import Polygon
    d = pd.read_csv(Path(__file__).parent / 'data/DESI_ext_fp.txt', sep=' ', comment='#')
    d = d[d['PROGRAM'] == 'DARK']
    for cap in ["NGC", "SGC"]:
        sel = d["CAP"] == cap

        ra, dec = d["RA"][sel] - rot, d["DEC"][sel]
        ra[ra > 180] -= 360    # scale conversion to [-180, 180]
        ra = -ra               # reverse the scale: East to the left

        if cap == 'SGC': 
            sel = ~(ra < 0)
            ra, dec = ra[sel], dec[sel]

        _ = ax.plot(np.radians(ra), np.radians(dec), color='darkblue', alpha=0.8, lw=1, label='DESI ext.' if cap == 'NGC' else None, zorder=10)
        ax.add_patch(Polygon(np.array([np.radians(ra), np.radians(dec)]).T, facecolor='darkblue', alpha=0.1))


def add_desiII_footprint(ax, rot=120):
    """ Based on IBIS imaging footprint, collected with my small eyes... """
    from matplotlib.patches import Polygon
    ngc = np.radians(np.array([[125,16], [260,16], [250,0], [230,0], [225,-7.5], [135,-7.5]]))
    sgc = np.radians(np.array([[45,7], [55,-15], [-55,-15],[-45,7]]))

    vertices = [ngc, sgc]

    for i, verts in enumerate(vertices):
        verts = verts.T
        verts[0] = np.radians(120)-verts[0] 
        verts = verts.T
        poly = Polygon(verts, closed=True, facecolor='gold', edgecolor='orange', lw=1, alpha=0.4, hatch='', label='DESI II' if i == 0 else None, zorder=2)
        ax.add_patch(poly) 

    return vertices


def add_act_footprint(ax, rot=120):
    """ see data/ACT_mask.ipynb to generate this file."""
    from pathlib import Path
    from skimage import measure
    from matplotlib.patches import Polygon
    import fitsio
    
    toplot = hp.reorder(fitsio.FITS(Path(__file__).parent / 'data' / 'Act_dr6_mask_256.fits')['MASK'][:], n2r=True)
    nside = hp.npix2nside(toplot.size)

    # ==== MAKE A RA/DEC GRID TO SAMPLE FOOTPRINT ====
    nra, ndec = 1800, 900  # ~0.2° resolution
    ra_grid = np.linspace(0, 360, nra, endpoint=False)
    dec_grid = np.linspace(-90, 90, ndec)

    RA2D, DEC2D = np.meshgrid(ra_grid, dec_grid[::-1])
    theta, phi = np.radians(90 - DEC2D), np.radians(RA2D)

    vecs = hp.ang2vec(theta.ravel(), phi.ravel())
    pix = hp.vec2pix(nside, vecs[:,0], vecs[:,1], vecs[:,2])
    sampled = toplot[pix].reshape(DEC2D.shape)

    # ==== FIND CONTOURS IN SAMPLE GRID ====
    contours = measure.find_contours(sampled.astype(float), 0.5)

    vertices = []
    for c in contours:
        rows, cols = c[:,0], c[:,1]
        decs = np.interp(rows, np.arange(sampled.shape[0]), dec_grid[::-1])
        ras  = np.interp(cols, np.arange(sampled.shape[1]), ra_grid)
        vertices += [np.column_stack([np.radians(ras), np.radians(decs)])]

    for verts in vertices:
        verts = verts.copy().T
        verts[0] = np.radians(rot)-verts[0] 
        verts = verts.T
        ax.plot(verts[:,0], verts[:,1], color='gray', alpha=1, lw=1, zorder=10)
        poly = Polygon(verts, closed=True, facecolor='gray', alpha=0.2, hatch='')
        ax.add_patch(poly)

    for verts in [vertices[2]]:
        verts = verts.copy().T
        verts[0] = np.radians(rot) - verts[0] + 2*np.pi - 0.004  # we cheat a bit -> slide to the left to fill the edge of the Polygon that left a white space...
        verts = verts.T
        ax.plot(verts[:,0], verts[:,1], color='gray', alpha=1, lw=1, label='ACT DR6', zorder=10)
        poly = Polygon(verts, closed=True, facecolor='gray', alpha=0.2, hatch='')
        ax.add_patch(poly)

    return vertices


def add_so_footprint(ax, rot=120):
    """ Simons Observatory footprint, determined with my small eyes ... """
    from matplotlib.patches import Polygon
    ax.add_patch(Polygon(np.array([[-3.14, -3.14, 3.14, 3.14], [-np.radians(60), np.radians(20), np.radians(20), -np.radians(60)]]).T, facecolor='gray', edgecolor='gray', alpha=0.3, hatch='/', label='SO'))


def plot_moll(map, min=None, max=None, title='', label=r'[$\#$ deg$^{-2}$]', filename=None, show=True, show_legend=True,
              rot=120, projection='mollweide', figsize=(11.0, 7.0), 
              xpad=1.25, labelpad=-37, xlabel_labelpad=10.0, ycb_pos=-0.15, cmap='jet', ticks=None, tick_labels=None,
              galactic_plane=True, ecliptic_plane=False, sgr_plane=False, stream_plane=False, 
              desi_fp=False, desi_ext_fp=False, desi_II_fp=False, act_fp=False, so_fp=False):
    """
    Plot an healpix map in nested scheme with a specific projection.

    Parameters
    ----------
    map : float array
        Healpix map in nested scheme
    min : float
        Minimum value for the colorbar
    max : float
        Maximum value for the colorbar
    title : str
        Title for the figure. Title is just above the colorbar
    label : str
        Colobar label. Label is just on the right of the colorbar
    filename : str
        Path where the figure will be saved. If filename is not None, the figure is saved.
    show : bool
        If true display the figure
    galactic_plane / ecliptic_plane / sgr_plane / stream_plane : bool
        Display the corresponding plane on the figure.
    show_lengend : bool
        If True, display the legend corresponding to the plotted plane. A warning is raised if show_lengend is True and no plane is plotted.
    rot : float
        Rotation of the R.A. axis for sky visualisation plot. In DESI, it should be rot=120.
    projection : str
        Projection used to plot the map. In DESI, it should be mollweide
    figsize : float tuple
        Size of the figure
    xpad : float
        X position of label. Need to be adpated if figsize is modified.
    labelpad : float
        Y position of label. Need to be adpated if figsize is modified.
    xlabel_labelpad : float
        Position of the xlabel (R.A.). Need to be adpated if figsize is modified.
    ycb_pos : float
        Y position of the colorbar. Need to be adpated if figsize is modified or if title is too long.
    cmap : ColorMap class of matplotlib
        Usefull to adapt the color. Especially to create grey area for the Y5 footprint.
        For instance: cmap = plt.get_cmap('jet').copy()
                      cmap.set_extremes(under='darkgrey')  # --> everything under min will be darkgrey
    """
    # transform healpix map to 2d array
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()

    # build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1] + 1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0] + 1)

    ra_edge[ra_edge > 180] -= 360    # scale conversion to [-180, 180]
    ra_edge = -ra_edge               # reverse the scale: East to the left

    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.2, right=0.96, top=0.98)

    # many more in ~/RP/plot_survey.ipynb at NERSC:
    if desi_fp: add_desi_footprint(ax, rot=rot)
    if desi_ext_fp: add_desi_ext_footprint(ax, rot=rot)
    if desi_II_fp: add_desiII_footprint(ax, rot=rot)
    if act_fp: add_act_footprint(ax, rot=rot)
    if so_fp: add_so_footprint(ax, rot=rot)

    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap=cmap, edgecolor='none', lw=0, zorder=1)

    if label is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, ycb_pos, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.8, aspect=40, ticks=ticks)
        cb.outline.set_visible(False)
        cb.set_label(label, x=xpad, labelpad=labelpad)
        if tick_labels is not None:
            cb.ax.set_xticklabels(tick_labels)  # horizontal colorbar
        cb.ax.tick_params(size=0)

    if galactic_plane: add_galactic_plane(ax, rot=rot)  
    if ecliptic_plane: add_ecliptic_plane(ax, rot=rot)
    if sgr_plane: add_sgr_plane(ax, rot=rot)
    if stream_plane: add_sgr_stream(ax, rot=rot)

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array([f'{lab}°' for lab in tick_labels])
    ax.set_xticklabels(tick_labels, zorder=2)

    ax.set_xlabel('R.A. [deg]', labelpad=xlabel_labelpad)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Dec. [deg]')

    ax.grid(True)

    if show_legend:
        leg = ax.legend(loc='lower right')
        leg.set_zorder(1000)  # Dessiner la légende en dernier (zorder élevé)
    if title:
        plt.title(title)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()
