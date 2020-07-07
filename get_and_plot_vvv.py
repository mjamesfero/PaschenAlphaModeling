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

pa_wavelength = 1.8756*u.um
pa_energy = pa_wavelength.to(u.erg, u.spectral())
pa_freq = pa_wavelength.to(u.Hz, u.spectral())

def get_and_plot_vvv(glon=2.5*u.deg, glat=0.1*u.deg, fov=27.5*u.arcmin,
                     pixscale=0.806*u.arcsec, exptime=500*u.s,
                     max_rows=int(4e5), kmag_threshold=8.5, wavelength=1875,
                     imsize=2048, diameter=24*u.cm,
                     readnoise=22*u.count, dark_rate=0.435*u.count/u.s,
                     transmission_fraction=0.70*0.75,
                    ):

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


    airy_radius = (1.22 * pa_wavelength / diameter).to(u.arcsec, u.dimensionless_angles())
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

#    fluxes = u.Quantity(10**(cat2['Ksmag3'] / -2.5)) * zpt_vista
    fluxes = functions.flux_function(hmag=cat2["Hmag3"], kmag=cat2['Ksmag3'], 
                                     wavelength=wavelength, VVV=True)
    bad_vvv = (cat2['Ksmag3'].mask | (pix_coords_vvv[0] < 0) | (pix_coords_vvv[0] > imsize) |
               (pix_coords_vvv[1] < 0) | (pix_coords_vvv[1] > imsize) | (~vvv_faint))

    phot_fluxes = fluxes[~bad_vvv] / pa_energy * u.photon

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    pa_freq).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.ph).value

    nsrc = len(phot_ct_rate)

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table = Table({'amplitude': phot_ct * transmission_fraction,
                          'x_0': pix_coords_vvv[0][~bad_vvv],
                          'y_0': pix_coords_vvv[1][~bad_vvv],
                          'radius': np.repeat(airy_radius/pixscale, nsrc),
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
    bad_2mass = (cat2mass['Kmag'].mask | (pix_coords_2mass[0] < 0) |
                 (pix_coords_2mass[0] > imsize) | (pix_coords_2mass[1] < 0) |
                 (pix_coords_2mass[1] > imsize) | (~twomass_bright))

    phot_fluxes = fluxes[~bad_2mass] / pa_energy * u.photon

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    pa_freq).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.ph).value

    nsrc = len(phot_ct_rate)

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table_2mass = Table({'amplitude': phot_ct * transmission_fraction,
                                'x_0': pix_coords_2mass[0][~bad_2mass],
                                'y_0': pix_coords_2mass[1][~bad_2mass],
                                'radius': np.repeat(airy_radius/pixscale, nsrc),
                               })


    source_table_both = table.vstack([source_table, source_table_2mass])


    rslt = functions.make_turbulent_starry_im(size=imsize, readnoise=readnoise, bias=0*u.count,
                                              dark_rate=dark_rate, exptime=exptime,
                                              nstars=None,
                                              sources=source_table_both,
                                              airy_radius=(airy_radius/pixscale).value,
                                              power=3, skybackground=False,
                                              sky=0, hotpixels=False,
                                              biascol=False, brightness=0,
                                              progressbar=ProgressBar)
    stars_background_im, turbulent_stars, turbulence = rslt


    return stars_background_im, turbulent_stars, turbulence, header
