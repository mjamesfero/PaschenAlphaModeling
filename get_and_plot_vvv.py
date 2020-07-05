import numpy as np
from astroquery.vizier import Vizier
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.table import Table
from astropy.utils.console import ProgressBar
from photutils.datasets import make_random_gaussians_table, make_model_sources_image

import pylab as pl
from astropy import visualization

import functions


glon, glat = 2.5*u.deg, 0.1*u.deg
fov = 25*u.arcmin

Viz = Vizier(row_limit=4e5)
cats = Viz.query_region(SkyCoord(glon, glat, frame='galactic'),
                        radius=fov/2**0.5, catalog=["II/337", "II/348",
                                                    "II/246"])

cat1, cat2, cat2mass = vvvcats
cat1c = SkyCoord(cat1['RAJ2000'], cat1['DEJ2000'], frame='fk5',
                 unit=(u.deg, u.deg)).galactic
cat2c = SkyCoord(cat2['RAJ2000'], cat2['DEJ2000'], frame='fk5',
                 unit=(u.deg, u.deg)).galactic
coords2mass = SkyCoord(cat2mass['RAJ2000'], cat2mass['DEJ2000'], frame='fk5',
                       unit=(u.deg, u.deg)).galactic



sz = 2048
diameter = (24*u.cm)
pa_wavelength = 1.8756*u.um
fwhm = (1.22 * pa_wavelength / diameter).to(u.arcsec, u.dimensionless_angles())
pixscale = fwhm / 3
fov = pixscale * sz
header = {'CRPIX1': sz/2,
          'CRPIX2': sz/2,
          'NAXIS1': sz,
          'NAXIS2': sz,
          'CRVAL1': glon.to(u.deg).value,
          'CRVAL2': glat.to(u.deg).value,
          'CTYPE1': 'GLON-CAR',
          'CTYPE2': 'GLAT-CAR',
          'CDELT1': -pixscale.to(u.deg).value,
          'CDELT2': pixscale.to(u.deg).value,
         }
target_image_wcs = wcs.WCS(header=header)

pix_coords = target_image_wcs.wcs_world2pix(cat2c.l.deg, cat2c.b.deg, 0)

from astroquery.svo_fps import SvoFps

filt_tbl = SvoFps.get_filter_list(facility='Paranal')
ks = filt_tbl[filt_tbl['filterID'] == b'Paranal/VISTA.Ks']
zpt = ks['ZeroPoint'].quantity

fluxes = u.Quantity(10**(cat2['Ksmag3'] / -2.5)) * zpt
bad = cat2['Ksmag3'].mask | (pix_coords[0] < 0) | (pix_coords[0] > sz) | (pix_coords[1] < 0) | (pix_coords[1] > sz)

pa_energy = pa_wavelength.to(u.erg, u.spectral())
pa_freq = pa_wavelength.to(u.Hz, u.spectral())


phot_fluxes = fluxes[~bad] / pa_energy * u.photon
collecting_area = np.pi*(diameter/2)**2
psf_area = 2*np.pi*(fwhm/(8*np.log(2)))**2
pixel_fraction_of_area = (pixscale**2 / psf_area).decompose()

phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area * pa_freq).decompose()
exptime = 500*u.s
phot_ct = (phot_ct_rate * exptime).to(u.ph).value

nsrc = len(phot_ct_rate)

#Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
source_table = Table({'amplitude': phot_ct,
                      'x_0': pix_coords[0][~bad],
                      'y_0': pix_coords[1][~bad],
                      'radius': np.repeat(fwhm/pixscale, nsrc),
                     })


rslt = functions.make_turbulent_im(size=sz, readnoise=100, bias=100, dark=10,
                                   exptime=exptime.value, nstars=None,
                                   sources=source_table,
                                   fwhm=(fwhm/pixscale).value, power=3, skybackground=False,
                                   sky=20, hotpixels=False, biascol=False,
                                   brightness=0, progressbar=ProgressBar)
stars_background_im, turbulent_stars, turbulence = rslt

# TODO: add in 2MASS bright stars


pl.imshow(stars_background_im,
          norm=visualization.simple_norm(stars_background_im, stretch='asinh',
                                         max_percent=99, min_percent=1e-4))
