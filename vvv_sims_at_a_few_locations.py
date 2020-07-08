import numpy as np
import os

from astropy import units as u
from astropy.io import fits

import json

from get_and_plot_vvv import get_and_plot_vvv

import warnings
warnings.filterwarnings(action='ignore', category=fits.verify.VerifyWarning)

def trytoget(glon, glat, **kwargs):
    fn = f"{glon.value:06.2f}{glat.value:+06.2f}_pa_filter.fits"
    offfn = f"{glon.value:06.2f}{glat.value:+06.2f}_off_filter.fits"
    if os.path.exists(fn) and os.path.exists(offfn):
        stars_background_im = fits.getdata(fn)
        stars_background_im_offset = fits.getdata(offfn)
    else:
        try:
            stars_background_im, turbulent_stars, turbulence, header = get_and_plot_vvv(glon, glat, wavelength=18750*u.AA, **kwargs)
            stars_background_im_offset, turbulent_stars_offset, turbulence_offset, header_offset = get_and_plot_vvv(glon, glat, wavelength=18800*u.AA, **kwargs)
        except Exception as ex:
            print(ex)
            return str(ex)
        header = fits.Header(header)
        fits.PrimaryHDU(data=stars_background_im, header=header).writeto(fn,
                                                                         output_verify='fix',
                                                                         overwrite=True)
        header_offset = fits.Header(header_offset)
        fits.PrimaryHDU(data=stars_background_im_offset, header=header_offset).writeto(offfn,
                                                                         output_verify='fix',
                                                                         overwrite=True)

    fcso = stars_background_im - stars_background_im_offset
    noise = np.sqrt(stars_background_im)
    fcso_noise_ratio = fcso/noise
    return stars_background_im, stars_background_im_offset, fcso_noise_ratio

results = {(glon, glat): trytoget(glon*u.deg, glat*u.deg)
           for glon, glat in
           [(2.5, 0.1), (2.5, 1), (2.5, 2), (2.5, 3), (2.5, -1),
            (-2.5, 0.1), (-2.5, 1), (-2.5, 2), (-2.5, 3), (-2.5, -1),
            (-1.5, 0.1), (-1.5, 1), (-1.5, 2), (-1.5, 3), (-1.5, -1),
            (1.5, 0.1), (1.5, 1), (1.5, 2), (1.5, 3), (1.5, -1),
            (0, 0.1), (0, 1), (0, 2), (0, 3), (0, -1),
            (5.0, 0.1), (5.0, 1), (5.0, 2), (5.0, 3), (5.0, -1),
            (-5.0, 0.1), (-5.0, 1), (-5.0, 2), (-5.0, 3), (-5.0, -1),
            (-10.0, 0.1), (-10.0, 1), (-10.0, 2), (-10.0, 3), (-10.0, -1),
            (-20.0, 0.1), (-20.0, 1), (-20.0, 2), (-20.0, 3), (-20.0, -1),
            (-30.0, 0.1), (-30.0, 1), (-30.0, 2), (-30.0, 3), (-30.0, -1),
            (-40.0, 0.1), (-40.0, 1), (-40.0, 2), (-40.0, 3), (-40.0, -1),
            (-50.0, 0.1), (-50.0, 1), (-50.0, 2), (-50.0, 3), (-50.0, -1),
            (-60.0, 0.1), (-60.0, 1), (-60.0, 2), (-60.0, 3), (-60.0, -1),
            (270.0, 0.1), (270.0, 1), (270.0, 2), (270.0, 3), (270.0, -1),
            (240.0, 0.1), (240.0, 1), (240.0, 2), (240.0, 3), (240.0, -1),
            (210.0, 0.1), (210.0, 1), (210.0, 2), (210.0, 3), (210.0, -1),
            (180.0, 0.1), (180.0, 1), (180.0, 2), (180.0, 3), (180.0, -1),
           ]
          }


# value[0] is stars_background_im
# let's determine the various percentiles: what's the 10%, 25%, etc. background
# level?
# "key" is glon,glat, which has to be stringified so we can save it below
stats_background = {"{0}_{1}".format(*key):
         {'glon': key[0],
          'glat': key[1],
          10: np.percentile(value[0], 10),
          25: np.percentile(value[0], 25),
          50: np.percentile(value[0], 50),
          75: np.percentile(value[0], 75),
          90: np.percentile(value[0], 90),
          95: np.percentile(value[0], 95),
          99: np.percentile(value[0], 99),
         }
         for key, value in results.items()
         if not isinstance(value, str)
        }

with open('background_percentiles.json', 'w') as fh:
    json.dump(obj=stats_background, fp=fh)

stats_offset = {"{0}_{1}".format(*key):
         {'glon': key[0],
          'glat': key[1],
          10: np.percentile(value[1], 10),
          25: np.percentile(value[1], 25),
          50: np.percentile(value[1], 50),
          75: np.percentile(value[1], 75),
          90: np.percentile(value[1], 90),
          95: np.percentile(value[1], 95),
          99: np.percentile(value[1], 99),
         }
         for key, value in results.items()
         if not isinstance(value, str)
        }

with open('background_percentiles_offset.json', 'w') as fh:
    json.dump(obj=stats_offset, fp=fh)

stats_fcso = {"{0}_{1}".format(*key):
         {'glon': key[0],
          'glat': key[1],
          10: np.percentile(value[2], 10),
          25: np.percentile(value[2], 25),
          50: np.percentile(value[2], 50),
          75: np.percentile(value[2], 75),
          90: np.percentile(value[2], 90),
          95: np.percentile(value[2], 95),
          99: np.percentile(value[2], 99),
         }
         for key, value in results.items()
         if not isinstance(value, str)
        }

with open('fcso_percentiles.json', 'w') as fh:
    json.dump(obj=stats_fcso, fp=fh)



from astropy import visualization
import pylab as pl



glon = [x['glon'] for x in stats_background.values()]
glat = [x['glat'] for x in stats_background.values()]
med = [x['50'] for x in stats_background.values()]

pl.clf()
pl.scatter(glon, glat, c=np.log10(np.array(med)/500), marker='s', s=200)
cb = pl.colorbar()
pl.xlabel("Galactic Longitude")
pl.ylabel("Galactic Latitude")
pl.xlim(pl.gca().get_xlim()[::-1])
cb.set_label("Median Background in log counts/s")
pl.savefig('median_background_level.pdf', bbox_inches='tight')
