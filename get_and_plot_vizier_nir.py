import numpy as np
from astroquery.vizier import Vizier
#from astroquery.svo_fps import SvoFps
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.table import Table
from astropy import table
from astropy.utils.console import ProgressBar
from photutils.datasets import make_random_gaussians_table, make_model_sources_image

from astropy import visualization

import functions
from sensitivity import (fov, pixscale, fiducial_integration_time, wl_paa,
                         aperture_diameter, throughput, dark_rate_pessimistic,
                         readnoise_pessimistic, paa_bandwidth, paac_bandwidth)

pa_wavelength = wl_paa
pa_energy = pa_wavelength.to(u.erg, u.spectral())
pa_freq = pa_wavelength.to(u.Hz, u.spectral())

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
                            radius=fov/2**0.5, catalog=["II/348", "II/246"])

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
    fluxes = functions.flux_function(hmag=cat2["Hmag3"], kmag=cat2['Ksmag3'],
                                     wavelength=wavelength, VVV=True)
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
    fluxes = functions.flux_function(hmag=cat2mass['Hmag'], kmag=cat2mass['Kmag'],
                                     wavelength=wavelength)
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

def get_and_plot_vizier_nir(glon=2.5*u.deg, glat=0.1*u.deg, fov=fov,
                            pixscale=pixscale,
                            exptime=fiducial_integration_time,
                            max_rows=int(1e6), kmag_threshold=8.5,
                            wavelength=wl_paa, imsize=2048, diameter=aperture_diameter,
                            linename='paa',
                            bandwidth=paa_bandwidth,
                            brightness=0, region='W51-CBAND-feathered.fits',
                            vary_psf=False, readnoise=readnoise_pessimistic,
                            dark_rate=dark_rate_pessimistic,
                            transmission_fraction=throughput, hii=False):
    """
    Dark current / readnoise:
    Pessimistic case is 0.435 ct/s, 22 ct
    Optimistic case is 0.0123 ct/s, 6.2 ct

    In surface brightness, these are (for exptime=500s):

    dark_rn_pess = (((22*u.ph/(500*u.s) + 0.435*u.ph/u.s) * (e_paa/u.ph) / (24*u.cm/2)**2 / np.pi / nu_paa).to(u.mJy)  / (0.806*u.arcsec)**2).to(u.MJy/u.sr)
    dark_rn_opt = (((6.2*u.ph/(500*u.s) + 0.0123*u.ph/u.s) * (e_paa/u.ph) / (24*u.cm/2)**2 / np.pi / nu_paa).to(u.mJy)  / (0.806*u.arcsec)**2).to(u.MJy/u.sr)
    """

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

    if hii:
       rslt = functions.make_HII_starry_im(size=imsize, readnoise=readnoise,
                                           bias=0*u.count, dark_rate=dark_rate,
                                           exptime=exptime, region=region,
                                           nstars=None, fov=fov*u.arcmin,
                                           sources=source_table_both,
                                           airy_radius=(airy_radius/pixscale).value,
                                           power=3, skybackground=False, sky=0,
                                           hotpixels=False, vary_psf=vary_psf,
                                           biascol=False,
                                           progressbar=ProgressBar)
       stars_background_im, turbulent_stars, turbulence = rslt
    else:
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
