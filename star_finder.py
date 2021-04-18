import numpy as np
from astropy import units as u
from astropy import constants as c
from random import choice
from astropy.table import Table
from astropy.io import fits
from astroquery.vizier import Vizier

from astropy.io import fits
import glob
from spectral_cube import lower_dimensional_structures
import pylab as pl
from astropy import visualization

from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import table
from astropy.utils.console import ProgressBar
from photutils.datasets import make_random_gaussians_table, make_model_sources_image

from astropy import visualization
from MIRIS_flux_calculator import flux_pa, flux_paacl, flux_paach, ks_mags, sed_flux_function

import functions
from sensitivity import (fov, pixscale, fiducial_integration_time, wl_paa,
                         aperture_diameter, throughput, dark_rate_pessimistic,
                         readnoise_pessimistic, paa_bandwidth, paac_bandwidth)


name1 = glob.glob('gc_*.fits')
name2 = [fits.getdata(x) for x in name1]

headers = []
for ii in range(3):
    headers.append(fits.getheader(name1[ii]))
ww = [wcs.WCS(headers[0]), wcs.WCS(headers[1]), wcs.WCS(headers[2])]

indices = [[],[],[]]
glon_glat_1 = []
glon_glat_2 = []
glon_glat_3 = []
for ii in range(9142):
    for jj in range(6202):
        if not np.isnan(name2[0][ii][jj]):
            indices[0].append([ii,jj])
            ra1, dec1 = ww[0].wcs_pix2world(ii,jj,0)
            glon_glat_1.append([ra1, dec1])
        if not np.isnan(name2[1][ii][jj]):
            indices[1].append([ii,jj])
            ra1, dec1 = ww[1].wcs_pix2world(ii,jj,0)
            glon_glat_2.append([ra1, dec1])
        if not np.isnan(name2[2][ii][jj]):
            indices[2].append([ii,jj])
            ra1, dec1 = ww[2].wcs_pix2world(ii,jj,0)
            glon_glat_3.append([ra1, dec1])

width = 47.5*u.arcsec
height = 47.5*u.arcsec


def make_source_table(glon=2.5*u.deg, glat=0.1*u.deg, fov=fov,
                      pixscale=pixscale, exptime=fiducial_integration_time,
                      max_rows=int(1e6), kmag_threshold=8.5, wavelength=wl_paa,
                      imsize=2048, diameter=aperture_diameter, linename='paa',
                      bandwidth=paa_bandwidth,
                      readnoise=readnoise_pessimistic,
                      dark_rate=dark_rate_pessimistic,
                      transmission_fraction=throughput):
    Viz = Vizier(row_limit=max_rows)
    cats = Viz.query_region(SkyCoord(glon, glat, frame='galactic'),
                            width=width, height=height, 
                            catalog=["II/348", "II/246"])

    cat2 = cats['II/348/vvv2']
    cat2mass = cats['II/246/out']

    cat2c = SkyCoord(cat2['RAJ2000'], cat2['DEJ2000'], frame='fk5',
                     unit=(u.deg, u.deg)).galactic
    coords2mass = SkyCoord(cat2mass['RAJ2000'], cat2mass['DEJ2000'],
                           frame='fk5', unit=(u.deg, u.deg)).galactic


    vvv_faint = cat2['Ksmag3'] > kmag_threshold
    twomass_bright = cat2mass['Kmag'] < kmag_threshold


    airy_radius = (1.22 * wavelength / diameter).to(u.arcsec, u.dimensionless_angles())
    header = {'CRPIX1': imsize/2,
              'CRPIX2': imsize/2,
              'NAXIS1': imsize,
              'NAXIS2': imsize,
              'CRVAL1': glon.to(u.deg).value,
              'CRVAL2': glat.to(u.deg).value,
              'CTYPE1': 'GLON-CAR',
              'CTYPE2': 'GLAT-CAR',
              'CDELT1': -pixscale.to(u.deg).value,
              'CDELT2': pixscale.to(u.deg).value,
             }
    target_image_wcs = wcs.WCS(header=header)

    collecting_area = np.pi*(diameter/2)**2
    # empirically determined: the integral of the Airy function
    airy_area_ratio = 8/3/np.pi
    psf_area = airy_area_ratio*(airy_radius)**2
    pixel_fraction_of_area = (pixscale**2 / psf_area).decompose()


    # Assemble source table from VVV sources
    pix_coords_vvv = target_image_wcs.wcs_world2pix(cat2c.l.deg, cat2c.b.deg, 0)

    # filt_tbl = SvoFps.get_filter_list(facility='Paranal')
    # ks = filt_tbl[filt_tbl['filterID'] == b'Paranal/VISTA.Ks']
    # zpt = ks['ZeroPoint'].quantity
 #   zpt_vista = 669.5625*u.Jy

    # fluxes = u.Quantity(10**(cat2['Ksmag3'] / -2.5)) * zpt_vista
    #[HERE]
    fluxes = sed_flux_function(kmags=cat2['Ksmag3'], wavelength=linename)
    bad_vvv = (cat2['Ksmag3'].mask | cat2['Hmag3'].mask | (pix_coords_vvv[0] < 0) | (pix_coords_vvv[0] > imsize) |
               (pix_coords_vvv[1] < 0) | (pix_coords_vvv[1] > imsize) | (~vvv_faint))

    phot_fluxes = fluxes / pa_energy * u.photon

    bandwidth_Hz = ((bandwidth / wavelength) * pa_freq).to(u.Hz)

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    bandwidth_Hz).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.ph).value

    cat2.add_column(col=phot_ct, name=f'{linename}_phot_ct')
    cat2.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
    cat2.add_column(col=fluxes, name=f'{linename}_flux')

    nsrc = len(phot_ct_rate[~bad_vvv])

    x = pix_coords_vvv[0][~bad_vvv]
    y = pix_coords_vvv[1][~bad_vvv]

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table = Table({'amplitude': phot_ct[~bad_vvv] * transmission_fraction,
                          'x_mean': np.round(x),
                          'y_mean': np.round(y),
                          'x_0': x,
                          'y_0': y,
                          'radius': np.repeat(airy_radius/pixscale, nsrc),
                          'x_stddev': abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
                          'y_stddev': abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
                          'theta': np.pi * (x-1024),
                         })


    # Assemble source table from 2MASS sources
    pix_coords_2mass = target_image_wcs.wcs_world2pix(coords2mass.l.deg,
                                                      coords2mass.b.deg, 0)

    # filt_tbl = SvoFps.get_filter_list(facility='2MASS')
    # ks = filt_tbl[filt_tbl['filterID'] == b'2MASS/2MASS.Ks']
    # zpt = ks['ZeroPoint'].quantity
#    zpt_2mass = 666.8*u.Jy

#    fluxes = u.Quantity(10**(cat2mass['Kmag'] / -2.5)) * zpt_2mass
    #[HERE]
    fluxes = sed_flux_function(kmags=cat2mass['Kmag'], wavelength=linename)
    bad_2mass = (cat2mass['Hmag'].mask | cat2mass['Kmag'].mask | (pix_coords_2mass[0] < 0) |
                 (pix_coords_2mass[0] > imsize) | (pix_coords_2mass[1] < 0) |
                 (pix_coords_2mass[1] > imsize) | (~twomass_bright))

    phot_fluxes = fluxes / pa_energy * u.photon

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    bandwidth_Hz).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.photon).value


    cat2mass.add_column(col=phot_ct, name=f'{linename}_phot_ct')
    cat2mass.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
    cat2mass.add_column(col=fluxes, name=f'{linename}_flux')

    nsrc = len(phot_ct_rate[~bad_2mass])

    x = pix_coords_2mass[0][~bad_2mass]
    y = pix_coords_2mass[1][~bad_2mass]

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table_2mass = Table({'amplitude': phot_ct[~bad_2mass] * transmission_fraction,
                                'x_mean': np.round(x),
                                'y_mean': np.round(y),
                                'x_0': x,
                                'y_0': y,
                                'radius': np.repeat(airy_radius/pixscale, nsrc),
                                'x_stddev' : abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
                                'y_stddev' : abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
                                'theta' : np.pi * (x-1024),
                               })


    source_table_both = table.vstack([source_table, source_table_2mass])

    return source_table_both, cat2, cat2mass, header

def get_and_plot_miris(glon=2.5*u.deg, glat=0.1*u.deg, fov=fov,
                            #pixscale=pixscale, import as constant
                            #exptime=fiducial_integration_time, import as constant
                            max_rows=int(1e6), kmag_threshold=8.5,
                            wavelength=wl_paa, imsize=2048, #diameter=aperture_diameter,
                            linename='paa',
                            bandwidth=paa_bandwidth,
                            brightness=0, #region='W51-CBAND-feathered.fits', useless i'm pretty sure
                            vary_psf=False): #readnoise=readnoise_pessimistic
                            #dark_rate=dark_rate_pessimistic, import constant
                            #transmission_fraction=throughput, import constant hii=False):
    """
    Dark current / readnoise:
    Pessimistic case is 0.435 ct/s, 22 ct
    Optimistic case is 0.0123 ct/s, 6.2 ct

    In surface brightness, these are (for exptime=500s):

    dark_rn_pess = (((22*u.ph/(500*u.s) + 0.435*u.ph/u.s) * (e_paa/u.ph) / (24*u.cm/2)**2 / np.pi / nu_paa).to(u.mJy)  / (0.806*u.arcsec)**2).to(u.MJy/u.sr)
    dark_rn_opt = (((6.2*u.ph/(500*u.s) + 0.0123*u.ph/u.s) * (e_paa/u.ph) / (24*u.cm/2)**2 / np.pi / nu_paa).to(u.mJy)  / (0.806*u.arcsec)**2).to(u.MJy/u.sr)
    """

    pixscale = 0
    exptime = 0
    diameter = 0

    source_table_both, _, _, header = make_source_table(glon=glon, glat=glat, fov=fov,
                                                pixscale=pixscale,
                                                exptime=exptime,
                                                max_rows=max_rows,
                                                kmag_threshold=kmag_threshold,
                                                wavelength=wavelength,
                                                imsize=imsize,
                                                diameter=diameter,
                                                linename=linename,
                                                bandwidth=bandwidth,
                                                transmission_fraction=transmission_fraction)

    airy_radius = (1.22 * wavelength / diameter).to(u.arcsec, u.dimensionless_angles())

    rslt = functions.make_turbulent_starry_im(size=imsize,
                                            readnoise=readnoise,
                                            bias=0*u.count,
                                            dark_rate=dark_rate,
                                            exptime=exptime, nstars=None,
                                            vary_psf=vary_psf,
                                            sources=source_table_both,
                                            airy_radius=(airy_radius/pixscale).value,
                                            power=3, skybackground=False,
                                            sky=0, hotpixels=False,
                                            biascol=False,
                                            brightness=brightness,
                                            progressbar=ProgressBar)
    stars_background_im, turbulent_stars, turbulence = rslt


    return stars_background_im, turbulent_stars, turbulence, header

def miris_get_flux(glon_and_glat):
    """
    AHhhhhhhh
    """
    for point in glon_and_glat:
        glon = point[0]
        glat = point[1]
        get_and_plot_miris(glon=glon, glat=glat)