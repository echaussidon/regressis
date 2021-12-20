# coding: utf-8
# Author : Edmond Chaussidon (CEA)

import numpy as np
import healpy as hp
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.coordinates as coord
from astropy.coordinates import frame_transform_graph
from astropy.coordinates.matrix_utilities import rotation_matrix, matrix_product, matrix_transpose

import warnings
from astropy.utils.exceptions import AstropyDeprecationWarning
warnings.simplefilter('ignore', category=AstropyDeprecationWarning)
# to avoid this message:
# WARNING: AstropyDeprecationWarning: Transforming a frame instance to a frame class (as opposed to another frame instance) will not be supported in the future.  Either explicitly instantiate the target frame, or first convert the source frame instance to a `astropy.coordinates.SkyCoord` and use its `transform_to()` method. [astropy.coordinates.baseframe]

#----------------------------------------------------------------------------------------------------#
# Build the Sagittarius plane

class Sagittarius(coord.BaseCoordinateFrame):
    default_representation = coord.SphericalRepresentation
    default_differential = coord.SphericalCosLatDifferential

    frame_specific_representation_info = {
        coord.SphericalRepresentation: [
            coord.RepresentationMapping('lon', 'Lambda'),
            coord.RepresentationMapping('lat', 'Beta'),
            coord.RepresentationMapping('distance', 'distance')]
        }

SGR_PHI = (180 + 3.75) * u.degree # Euler angles (from Law & Majewski 2010)
SGR_THETA = (90 - 13.46) * u.degree
SGR_PSI = (180 + 14.111534) * u.degree

# Generate the rotation matrix using the x-convention (see Goldstein)
D = rotation_matrix(SGR_PHI, "z")
C = rotation_matrix(SGR_THETA, "x")
B = rotation_matrix(SGR_PSI, "z")
A = np.diag([1.,1.,-1.])
SGR_MATRIX = matrix_product(A, B, C, D)


@frame_transform_graph.transform(coord.StaticMatrixTransform, coord.Galactic, Sagittarius)
def galactic_to_sgr():
    """ Compute the transformation matrix from Galactic spherical to
        heliocentric Sgr coordinates.
    """
    return SGR_MATRIX


@frame_transform_graph.transform(coord.StaticMatrixTransform, Sagittarius, coord.Galactic)
def sgr_to_galactic():
    """ Compute the transformation matrix from heliocentric Sgr coordinates to
        spherical Galactic.
    """
    return matrix_transpose(SGR_MATRIX)


# Compute the different plane in ircs coordinates
galactic_plane_tmp = SkyCoord(l=np.linspace(0, 2*np.pi, 200)*u.radian, b=np.zeros(200)*u.radian, frame='galactic', distance=1*u.Mpc)
galactic_plane_icrs = galactic_plane_tmp.transform_to('icrs')
index_galactic = np.argsort(galactic_plane_icrs.ra.wrap_at(300*u.deg).degree)

ecliptic_plane_tmp = SkyCoord(lon=np.linspace(0, 2*np.pi, 200)*u.radian, lat=np.zeros(200)*u.radian, distance=1*u.Mpc, frame='heliocentrictrueecliptic')
ecliptic_plane_icrs = ecliptic_plane_tmp.transform_to('icrs')
index_ecliptic = np.argsort(ecliptic_plane_icrs.ra.wrap_at(300*u.deg).degree)

sgr_plane_tmp = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=np.zeros(200)*u.radian, distance=1*u.Mpc)
sgr_plane_icrs = sgr_plane_tmp.transform_to(coord.ICRS)
index_sgr = np.argsort(sgr_plane_icrs.ra.wrap_at(300*u.deg).degree)

sgr_stream_top_icrs = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=20*np.pi/180*np.ones(200)*u.radian, distance=1*u.Mpc).transform_to(coord.ICRS)
index_sgr_top = np.argsort(sgr_stream_top_icrs.ra.wrap_at(300*u.deg).degree)
sgr_stream_bottom_icrs = Sagittarius(Lambda=np.linspace(0, 2*np.pi, 200)*u.radian, Beta=-15*np.pi/180*np.ones(200)*u.radian, distance=1*u.Mpc).transform_to(coord.ICRS)
index_sgr_bottom = np.argsort(sgr_stream_bottom_icrs.ra.wrap_at(300*u.deg).degree)


def plot_moll(map, min=None, max=None, title='', label=r'[$\#$ deg$^{-2}$]', savename=None, show=True, galactic_plane=False, ecliptic_plane=False, sgr_plane=False, stream_plane=False, show_legend=True, rot=120, projection='mollweide', figsize=(11.0, 7.0), xpad=1.25, labelpad=-37, xlabel_labelpad=10.0, ycb_pos=-0.15):

    #transform healpix map to 2d array
    plt.figure(1)
    m = hp.ma(map)
    map_to_plot = hp.cartview(m, nest=True, rot=rot, flip='geo', fig=1, return_projected_map=True)
    plt.close()

    #build ra, dec meshgrid to plot 2d array
    ra_edge = np.linspace(-180, 180, map_to_plot.shape[1]+1)
    dec_edge = np.linspace(-90, 90, map_to_plot.shape[0]+1)

    ra_edge[ra_edge>180] -=360    # scale conversion to [-180, 180]
    ra_edge=-ra_edge              # reverse the scale: East to the left

    ra_grid, dec_grid = np.meshgrid(ra_edge, dec_edge)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111, projection=projection)
    plt.subplots_adjust(left=0.14, bottom=0.18, right=0.96, top=0.90)

    mesh = plt.pcolormesh(np.radians(ra_grid), np.radians(dec_grid), map_to_plot, vmin=min, vmax=max, cmap='jet', edgecolor='none', lw=0)

    if label!=None:
        ax_cb = inset_axes(ax, width="30%", height="4%", loc='lower left', bbox_to_anchor=(0.346, ycb_pos, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
        cb = plt.colorbar(mesh, ax=ax, cax=ax_cb, orientation='horizontal', shrink=0.8, aspect=40)
        cb.outline.set_visible(False)
        cb.set_label(label, x=xpad, labelpad=labelpad)

    if galactic_plane:
        ra, dec = galactic_plane_icrs.ra.degree - rot, galactic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_galactic]), np.radians(dec[index_galactic]), linestyle='-', linewidth=0.8, color='black', label='Galactic plane')
    if ecliptic_plane:
        ra, dec = ecliptic_plane_icrs.ra.degree - rot, ecliptic_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_ecliptic]), np.radians(dec[index_ecliptic]), linestyle=':', linewidth=0.8, color='slategrey', label='Ecliptic plane')
    if sgr_plane:
        ra, dec = sgr_plane_icrs.ra.degree - rot, sgr_plane_icrs.dec.degree
        ra[ra>180] -=360    # scale conversion to [-180, 180]
        ra=-ra              # reverse the scale: East to the left
        ax.plot(np.radians(ra[index_sgr]), np.radians(dec[index_sgr]), linestyle='--', linewidth=0.8, color='navy', label='Sgr. plane')

        if stream_plane:
            ra, dec = sgr_stream_top_icrs.ra.degree - rot, sgr_stream_top_icrs.dec.degree
            ra[ra>180] -=360    # scale conversion to [-180, 180]
            ra=-ra              # reverse the scale: East to the left
            ax.plot(np.radians(ra[index_sgr_top]), np.radians(dec[index_sgr_top]), linestyle=':', linewidth=0.8, color='navy')

            ra, dec = sgr_stream_bottom_icrs.ra.degree - rot, sgr_stream_bottom_icrs.dec.degree
            ra[ra>180] -=360    # scale conversion to [-180, 180]
            ra=-ra              # reverse the scale: East to the left
            ax.plot(np.radians(ra[index_sgr_bottom]), np.radians(dec[index_sgr_bottom]), linestyle=':', linewidth=0.8, color='navy')

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
    if title!='':
        plt.title(title)
    if savename != None:
        plt.savefig(savename)
    if show:
        plt.show()
    else:
        plt.close()
