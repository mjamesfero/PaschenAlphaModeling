import numpy as np
from astroquery.vizier import Vizier
from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy import wcs
from astropy.table import Table
from astropy.utils.console import ProgressBar
from photutils.datasets import make_random_gaussians_table, make_model_sources_image

import functions


glon, glat = 2.5*u.deg, 0.1*u.deg
fov = 25*u.arcmin

Viz = Vizier(row_limit=3e5)
vvvcats = Viz.query_region(SkyCoord(glon, glat, frame='galactic'),
                           radius=fov/2, catalog=["II/337", "II/348"])

cat1, cat2 = vvvcats
cat1c = SkyCoord(vvvcats[0]['RAJ2000'], vvvcats[0]['DEJ2000'], frame='fk5',
                 unit=(u.deg, u.deg)).galactic
cat2c = SkyCoord(vvvcats[1]['RAJ2000'], vvvcats[1]['DEJ2000'], frame='fk5',
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

import pylab as pl
from astropy import visualization
pl.imshow(stars_background_im,
          norm=visualization.simple_norm(stars_background_im, stretch='asinh',
                                         max_percent=99, min_percent=1))

rslt2 = functions.make_turbulent_im(size=sz, readnoise=0, bias=0, dark=0,
                                   exptime=exptime.value, nstars=None,
                                   sources=source_table,
                                   fwhm=(fwhm/pixscale).value, power=3, skybackground=False,
                                   sky=20, hotpixels=False, biascol=False,
                                   brightness=0, progressbar=ProgressBar)
stars_background_im, turbulent_stars, turbulence = rslt2

pl.imshow(stars_background_im,
          norm=visualization.simple_norm(stars_background_im, stretch='asinh',
                                         max_percent=99.95, min_percent=0.01))


from astropy.modeling import models
model = models.AiryDisk2D((fwhm/pixscale).value)
row=source_table[0]
model.amplitude = float(row['amplitude'])
model.x_0 = row['x_0']
model.y_0 = row['y_0']
model.radius = row['radius']
bbox_size=5
model.bounding_box = [(model.y_0-bbox_size*model.radius,
                       model.y_0+bbox_size*model.radius),
                      (model.x_0-bbox_size*model.radius,
                       model.x_0+bbox_size*model.radius)]
model.render(stars_background_im)
stars_background_im[1890:1920, 281:311].max()

bbox = model.bounding_box
pd = np.array([(np.mean(bb), np.ceil((bb[1] - bb[0]) / 2))
               for bb in bbox]).astype(int).T
limits = [slice(p - d, p + d + 1, 1) for p, d in pd.T]
sub_coords = np.mgrid[limits]
sub_coords = sub_coords[::-1]
print(model(*sub_coords).max())

print(stars_background_im[limits].max())
