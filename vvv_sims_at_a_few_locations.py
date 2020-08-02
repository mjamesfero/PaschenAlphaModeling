import numpy as np
import os

from astropy import units as u
from astropy.io import fits
from astropy.stats import mad_std

import json
from sensitivity import saturation_limit

from get_and_plot_vizier_nir import get_and_plot_vizier_nir

import warnings
warnings.filterwarnings(action='ignore', category=fits.verify.VerifyWarning)

def trytoget(glon, glat, **kwargs):
    fn = f"{glon.value:06.2f}{glat.value:+06.2f}_pa_filter.fits"
    paaclfn = f"{glon.value:06.2f}{glat.value:+06.2f}_paacl_filter.fits"
    paachfn = f"{glon.value:06.2f}{glat.value:+06.2f}_paach_filter.fits"
    if os.path.exists(fn) and os.path.exists(paachfn):
        stars_background_im = fits.getdata(fn)
        stars_background_im_paach = fits.getdata(paachfn)
        stars_background_im_paacl = fits.getdata(paaclfn)
    else:
        try:
            stars_background_im, turbulent_stars, turbulence, header = get_and_plot_vizier_nir(glon, glat, wavelength=18756*u.AA, bandwidth=5*u.nm, linename='paa', **kwargs)
            stars_background_im_paach, turbulent_stars_paach, turbulence_paach, header_paach = get_and_plot_vizier_nir(glon, glat, wavelength=18850*u.AA, bandwidth=10*u.nm, linename='paac_h', **kwargs)
            stars_background_im_paacl, turbulent_stars_paacl, turbulence_paacl, header_paacl = get_and_plot_vizier_nir(glon, glat, wavelength=18660*u.AA, bandwidth=10*u.nm, linename='paac_l', **kwargs)
        except Exception as ex:
            print(ex)
            return str(ex)
        header = fits.Header(header)
        fits.PrimaryHDU(data=stars_background_im, header=header).writeto(fn,
                                                                         output_verify='fix',
                                                                         overwrite=True)
        header_paach = fits.Header(header_paach)
        fits.PrimaryHDU(data=stars_background_im_paach, header=header_paach).writeto(paachfn, output_verify='fix', overwrite=True)
        header_paacl = fits.Header(header_paacl)
        fits.PrimaryHDU(data=stars_background_im_paacl, header=header_paacl).writeto(paaclfn, output_verify='fix', overwrite=True)

    stars_background_im_offset = (stars_background_im_paacl + stars_background_im_paach)/4
    fcso = stars_background_im - stars_background_im_offset
    poisson_noise = np.sqrt(stars_background_im + stars_background_im_offset)
    systematic_noise = mad_std(fcso)
    noise_no_turbulence = np.sqrt(poisson_noise**2 + systematic_noise**2)
    SNR = np.abs(fcso)/noise_no_turbulence
    greater_than = np.count_nonzero(np.abs(SNR) > 1)/(2048**2)

    return stars_background_im, stars_background_im_paacl, stars_background_im_paach, noise_no_turbulence, greater_than

if __name__ == "__main__":
    # Do a "dry run" first to make sure there are no errors...
    stars_background_im, turbulent_stars, turbulence, header = get_and_plot_vizier_nir(10*u.deg, 5*u.deg, wavelength=18756*u.AA, bandwidth=5*u.nm, linename='paa')

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
              10: np.nanpercentile(value[0], 10),
              25: np.nanpercentile(value[0], 25),
              50: np.nanpercentile(value[0], 50),
              75: np.nanpercentile(value[0], 75),
              90: np.nanpercentile(value[0], 90),
              95: np.nanpercentile(value[0], 95),
              99: np.nanpercentile(value[0], 99),
              'saturated': (value[0]>saturation_limit).sum()/value[0].size,
             }
             for key, value in results.items()
             if not isinstance(value, str)
            }

    with open('background_low_nanpercentiles.json', 'w') as fh:
        json.dump(obj=stats_background, fp=fh)

    stats_offset = {"{0}_{1}".format(*key):
             {'glon': key[0],
              'glat': key[1],
              10: np.nanpercentile(value[1], 10),
              25: np.nanpercentile(value[1], 25),
              50: np.nanpercentile(value[1], 50),
              75: np.nanpercentile(value[1], 75),
              90: np.nanpercentile(value[1], 90),
              95: np.nanpercentile(value[1], 95),
              99: np.nanpercentile(value[1], 99),
              'saturated': (value[1]>saturation_limit).sum()/value[1].size,
             }
             for key, value in results.items()
             if not isinstance(value, str)
            }

    with open('background_high_nanpercentiles.json', 'w') as fh:
        json.dump(obj=stats_background, fp=fh)

    stats_offset = {"{0}_{1}".format(*key):
             {'glon': key[0],
              'glat': key[1],
              10: np.nanpercentile(value[2], 10),
              25: np.nanpercentile(value[2], 25),
              50: np.nanpercentile(value[2], 50),
              75: np.nanpercentile(value[2], 75),
              90: np.nanpercentile(value[2], 90),
              95: np.nanpercentile(value[2], 95),
              99: np.nanpercentile(value[2], 99),
              'saturated': (value[2]>saturation_limit).sum()/value[2].size,
             }
             for key, value in results.items()
             if not isinstance(value, str)
            }

    with open('background_nanpercentiles_offset.json', 'w') as fh:
        json.dump(obj=stats_offset, fp=fh)

    stats_fcso = {"{0}_{1}".format(*key):
             {'glon': key[0],
              'glat': key[1],
              10: np.nanpercentile(value[3], 10),
              25: np.nanpercentile(value[3], 25),
              50: np.nanpercentile(value[3], 50),
              75: np.nanpercentile(value[3], 75),
              90: np.nanpercentile(value[3], 90),
              95: np.nanpercentile(value[3], 95),
              99: np.nanpercentile(value[3], 99),
              'saturated': (value[3]>saturation_limit).sum()/value[3].size,
             }
             for key, value in results.items()
             if not isinstance(value, str)
            }

    with open('fcso_nanpercentiles.json', 'w') as fh:
        json.dump(obj=stats_fcso, fp=fh)

    percentage_fcso = {"{0}_{1}".format(*key):
             {'glon': key[0],
              'glat': key[1],
              10: np.nanpercentile(value[4], 10),
              25: np.nanpercentile(value[4], 25),
              50: np.nanpercentile(value[4], 50),
              75: np.nanpercentile(value[4], 75),
              90: np.nanpercentile(value[4], 90),
              95: np.nanpercentile(value[4], 95),
              99: np.nanpercentile(value[4], 99),
             }
             for key, value in results.items()
             if not isinstance(value, str)
            }

    with open('fcso_nanpercentiles_snr.json', 'w') as fh:
        json.dump(obj=percentage_fcso, fp=fh)


    from astropy import visualization
    import pylab as pl



    glon = [x['glon'] for x in stats_background.values()]
    glat = [x['glat'] for x in stats_background.values()]
    med = [x[50] for x in stats_background.values()]

    pl.clf()
    pl.scatter(glon, glat, c=np.log10(np.array(med)/500), marker='s', s=200)
    cb = pl.colorbar()
    pl.xlabel("Galactic Longitude")
    pl.ylabel("Galactic Latitude")
    pl.xlim(pl.gca().get_xlim()[::-1])
    cb.set_label("Median Background in log counts/s")
    pl.savefig('median_background_level.pdf', bbox_inches='tight')
