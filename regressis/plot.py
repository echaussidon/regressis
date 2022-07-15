#!/usr/bin/env python
# coding: utf-8

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt

import astropy.units as u
import astropy.coordinates as coord
from astropy.coordinates import SkyCoord, frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

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
            coord.RepresentationMapping('distance', 'distance')]
        }


def SGR_MATRIX():
    """Build the transformation matric from Galactic spherical to heliocentric Sgr coordinates based on Law & Majewski 2010."""
    SGR_PHI = (180 + 3.75) * u.degree # Euler angles (from Law & Majewski 2010)
    SGR_THETA = (90 - 13.46) * u.degree
    SGR_PSI = (180 + 14.111534) * u.degree

    # Generate the rotation matrix using the x-convention (see Goldstein)
    D = rotation_matrix(SGR_PHI, "z")
    C = rotation_matrix(SGR_THETA, "x")
    B = rotation_matrix(SGR_PSI, "z")
    A = np.diag([1.,1.,-1.])
    SGR_matrix = matrix_product(A, B, C, D)
    return SGR_matrix


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """Compute the transformation matrix from Galactic spherical to heliocentric Sgr coordinates."""
    return SGR_MATRIX()


@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """Compute the transformation matrix from heliocentric Sgr coordinates to spherical Galactic."""
    return matrix_transpose(SGR_MATRIX())


def _get_galactic_plane(rot=120):
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
    galactic_plane_tmp = SkyCoord(l=np.linspace(0, 2*np.pi, 200)*u.radian, b=np.zeros(200)*u.radian, frame='galactic', distance=1*u.Mpc)
    galactic_plane_icrs = galactic_plane_tmp.transform_to('icrs')

    ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra=-ra               # reverse the scale: East to the left

    # get the correct order from ra=-180 to ra=180 after rotation
    index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at((180 + rot)*u.deg).degree)

    return ra[index_galactic], dec[index_galactic]


def _get_ecliptic_plane(rot=120):
    """ Same than _get_galactic_coordinates but for the ecliptic plane in IRCS coordiantes"""
    ecliptic_plane_tmp = SkyCoord(lon=np.linspace(0, 2*np.pi, 200)*u.radian, lat=np.zeros(200)*u.radian, distance=1*u.Mpc, frame='heliocentrictrueecliptic')
    ecliptic_plane_icrs = ecliptic_plane_tmp.transform_to('icrs')

    ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra=-ra               # reverse the scale: East to the left

    index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at((180 + rot)*u.deg).degree)

    return ra[index_ecliptic], dec[index_ecliptic]


def _get_sgr_plane(rot=120):
    """ Same than _get_galactic_coordinates but for the Sagittarius Galactic plane in IRCS coordiantes"""
    sgr_plane_tmp = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=np.zeros(200)*u.radian, distance=1*u.Mpc)
    sgr_plane_icrs = sgr_plane_tmp.transform_to(coord.ICRS)

    ra, dec = sgr_plane_icrs.ra.degree - rot, sgr_plane_icrs.dec.degree
    ra[ra>180] -= 360    # scale conversion to [-180, 180]
    ra=-ra               # reverse the scale: East to the left

    index_sgr = np.argsort(sgr_plane_icrs.ra.wrap_at((180 + rot)*u.deg).degree)

    return ra[index_sgr], dec[index_sgr]


def _get_sgr_stream(rot=120):
    """ Same than _get_galactic_coordinates but for the bottom and top line of the Sgr. Stream in IRCS coordiantes"""
    sgr_stream_top_tmp = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=20*np.pi/180*np.ones(200)*u.radian, distance=1*u.Mpc)
    sgr_stream_top_icrs = sgr_stream_top_tmp.transform_to(coord.ICRS)

    ra_top, dec_top = sgr_stream_top_icrs.ra.degree - rot, sgr_stream_top_icrs.dec.degree
    ra_top[ra_top>180] -= 360    # scale conversion to [-180, 180]
    ra_top=-ra_top               # reverse the scale: East to the left

    index_sgr_top = np.argsort(sgr_stream_top_icrs.ra.wrap_at((180 + rot)*u.deg).degree)

    sgr_stream_bottom_tmp = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=-15*np.pi/180*np.ones(200)*u.radian, distance=1*u.Mpc)
    sgr_stream_bottom_icrs = sgr_stream_bottom_tmp.transform_to(coord.ICRS)

    ra_bottom, dec_bottom = sgr_stream_bottom_icrs.ra.degree - rot, sgr_stream_bottom_icrs.dec.degree
    ra_bottom[ra_bottom>180] -= 360    # scale conversion to [-180, 180]
    ra_bottom=-ra_bottom               # reverse the scale: East to the left

    index_sgr_bottom = np.argsort(sgr_stream_bottom_icrs.ra.wrap_at((180 + rot)*u.deg).degree)

    return ra_bottom[index_sgr_bottom], dec_bottom[index_sgr_bottom], ra_top[index_sgr_top], dec_top[index_sgr_top]


def plot_moll(map, min=None, max=None, title='', label=r'[$\#$ deg$^{-2}$]', filename=None, show=True,
              galactic_plane=False, ecliptic_plane=False, sgr_plane=False, stream_plane=False, show_legend=True,
              rot=120, projection='mollweide', figsize=(11.0, 7.0), xpad=1.25, labelpad=-37, xlabel_labelpad=10.0, ycb_pos=-0.15, cmap='jet'):
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
    #transform healpix map to 2d array
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()

    #build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1]+1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0]+1)

    ra_edge[ra_edge>180] -= 360    # scale conversion to [-180, 180]
    ra_edge=-ra_edge               # reverse the scale: East to the left

    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96, top=0.90)

    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap=cmap, edgecolor='none', lw=0)

    if label is not None:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, ycb_pos, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.8, aspect=40)
        cb.outline.set_visible(False)
        cb.set_label(label, x=xpad, labelpad=labelpad)

    if galactic_plane:
        ra, dec = _get_galactic_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle='-', linewidth=0.8, color='black', label='Galactic plane')
    if ecliptic_plane:
        ra, dec = _get_ecliptic_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='slategrey', label='Ecliptic plane')
    if sgr_plane:
        ra, dec = _get_sgr_plane(rot=rot)
        ax.plot(np.radians(ra), np.radians(dec), linestyle='--', linewidth=0.8, color='navy', label='Sgr. plane')
        if stream_plane:
            ra_bottom, dec_bottom, ra_top, dec_top = _get_sgr_stream(rot=rot)
            ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='navy')
            ax.plot(np.radians(ra), np.radians(dec), linestyle=':', linewidth=0.8, color='navy')

    tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    tick_labels = np.remainder(tick_labels + 360 + rot, 360)
    tick_labels = np.array(['{0}°'.format(l) for l in tick_labels])
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel('R.A. [deg]', labelpad=xlabel_labelpad)
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('Dec. [deg]')

    ax.grid(True)

    if show_legend:
        ax.legend(loc='lower right')
    if title:
        plt.title(title)
    if filename != None:
        plt.savefig(filename)
    if show:
        plt.show()
    else:
        plt.close()
