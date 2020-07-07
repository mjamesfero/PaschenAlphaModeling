import numpy as np
import os

from astropy import units as u
from astropy.io import fits

import json

from get_and_plot_vvv import get_and_plot_vvv

def trytoget(glon, glat, **kwargs):
    fn = f"{glon:06.2f}{glat:+06.2f}.fits"
    if os.path.exists(fn):
        return fits.getdata(fn), 0, 0, fits.getheader(fn)
    else:
        try:
            stars_background_im, turbulent_stars, turbulence, header = get_and_plot_vvv(glon, glat, **kwargs)
            fits.PrimaryHDU(data=stars_background_im, header=header).writeto(fn, overwrite=True)
        except Exception as ex:
            print(ex)
            return str(ex)

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
stats = {"{0}_{1}".format(*key):
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

with open('percentiles_by_glonglat.json', 'w') as fh:
    json.dump(obj=stats, fp=fh)
