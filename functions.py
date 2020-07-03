from astropy.convolution import AiryDisk2DKernel
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astroquery.vizier import Vizier
from matplotlib import pyplot as plt
from photutils.datasets import make_random_gaussians_table, make_model_sources_image
from scipy.signal import convolve
from turbustat.simulator import make_extended
import astropy.coordinates as coord
import astropy.units as u
import numpy as np

plt.rcParams['image.origin'] = 'lower'

def make_turbulent_im(size, readnoise, bias, dark, exptime, stars,
                      counts=10000, fwhm=3.2, power=3, skybackground=False,
                      sky=20, hotpixels=False, biascol=False, brightness=1):

    blank_image = np.zeros([size, size])
    ## based off of https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html
    ##pretty much just copy paste and combined functions
    ##readnoise
    shape = blank_image.shape
    noise = np.random.normal(scale=readnoise, size=shape)
    noise_im = blank_image + noise
    ##bias
    bias_im = np.zeros_like(blank_image) + bias
    if biascol:
        number_of_colums = 5
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        col_pattern = rng.randint(0, int(0.1 * bias), size=shape[0])
        for c in columns:
            bias_im[:, c] = bias + col_pattern
    ##readnoise and bias
    bias_noise_im = noise_im + bias_im
    ##dark current
    base_current = dark * exptime
    dark_im = np.random.poisson(base_current, size=shape)
    ##hot pixels
    if hotpixels:
        y_max, x_max = dark_im.shape
        n_hot = int(0.00002 * x_max * y_max)
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        hot_current = 10000 * dark
        dark_im[[hot_y, hot_x]] = hot_current * exptime
    ##dark bias readnoise hot pixels
    dark_bias_noise_im = bias_noise_im + dark_im
    ##optional skybackground
    if skybackground:
        sky_im = np.random.poisson(sky, size=image.shape)
        dark_bias_noise_im += sky_im
    ##stars
    flux_range = [counts/100, counts]
    y_max, x_max = shape
    xmean_range = [0.01 * x_max, 0.99 * x_max]
    ymean_range = [0.01 * y_max, 0.99 * y_max]
    xstddev_range = [1, 4]
    ystddev_range = [1, 4]
    model = models.AiryDisk2D(fwhm)
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])
    sources = make_random_gaussians_table(stars, params,
                                          random_state=12345)
    star_im = make_model_sources_image(shape, model, sources)
    ##stars and background
    stars_background_im = star_im + dark_bias_noise_im
    ##turbulence
    turbulent_data = make_extended(size, power)
    min_val = np.min(turbulent_data)
    turbulence = (turbulent_data - min_val + 1)*brightness
    turbulent_im = convolve(turbulence, AiryDisk2DKernel(fwhm), mode="same")
    ##turbulent image with stars
    turbulent_stars = turbulent_im + stars_background_im

    return stars_background_im, turbulent_stars, turbulence




def make_realistic_im(size, readnoise, bias, dark, exptime, stars,
                      counts=10000, fwhm=3.2, power=3, skybackground=False, sky=20,
                      hotpixels=False, biascol=False, brightness=1):

    blank_image = np.zeros([size, size])
    ## based off of https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html
    ##pretty much just copy paste and combined functions
    ##readnoise
    shape = blank_image.shape
    noise = np.random.normal(scale=readnoise, size=shape)
    noise_im = blank_image + noise
    ##bias
    bias_im = np.zeros_like(blank_image) + bias
    if biascol:
        number_of_colums = 5
        rng = np.random.RandomState(seed=8392)  # 20180520
        columns = rng.randint(0, shape[1], size=number_of_colums)
        col_pattern = rng.randint(0, int(0.1 * bias), size=shape[0])
        for c in columns:
            bias_im[:, c] = bias + col_pattern
    ##readnoise and bias
    bias_noise_im = noise_im + bias_im
    ##dark current
    base_current = dark * exptime
    dark_im = np.random.poisson(base_current, size=shape)
    ##hot pixels
    if hotpixels:
        y_max, x_max = dark_im.shape
        n_hot = int(0.00002 * x_max * y_max)
        rng = np.random.RandomState(16201649)
        hot_x = rng.randint(0, x_max, size=n_hot)
        hot_y = rng.randint(0, y_max, size=n_hot)
        hot_current = 10000 * dark
        dark_im[[hot_y, hot_x]] = hot_current * exptime
    ##dark bias readnoise hot pixels
    dark_bias_noise_im = bias_noise_im + dark_im
    ##optional skybackground
    if skybackground:
        sky_im = np.random.poisson(sky, size=image.shape)
        dark_bias_noise_im += sky_im
    ##stars
    flux_range = [counts/100, counts]
    y_max, x_max = shape
    xmean_range = [0.01 * x_max, 0.99 * x_max]
    ymean_range = [0.01 * y_max, 0.99 * y_max]
    xstddev_range = [1, 4]
    ystddev_range = [1, 4]
    model = models.AiryDisk2D(fwhm)
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])
    sources = make_random_gaussians_table(stars, params,
                                          random_state=12345)
    star_im = make_model_sources_image(shape, model, sources)
    ##stars and background
    stars_background_im = star_im + dark_bias_noise_im
    ##turbulence
    turbulent_data = make_extended(size, power)
    min_val = np.min(turbulent_data)
    turbulence = (turbulent_data - min_val + 1)*brightness
    turbulent_im = convolve(turbulence, AiryDisk2DKernel(fwhm), mode="same")
    ##turbulent image with stars
    turbulent_stars = turbulent_im + stars_background_im

    return stars_background_im, turbulent_stars, turbulence



#ideally this will combine the steps of getting the stars and making the synth image but
#not in the immediate future (i.e. not this week)
def skysurvey(lat_lower, lat_upper lon_lower, lon_upper VVV=False,
              pixel_scale=0.75, patience=3000, rowlim=3e5):
    #lat can be negative but lon must be between [0,360)
    Vizier = Vizier(timeout=patience)
    Vizier.ROW_LIMIT = rowlim
    if VVV==True
        asa = 1
    else:
        if lat_upper < lat_lower:
            lat = f"<{lat_upper} | > {lat_lower}"
        else:
            lat = f"< {lat_upper} & > {lat_lower}"
        if lon_upper < lon_lower:
            lon = f"<{lon_upper} | > {lon_lower}"
        else:
            lon = f"< {lon_upper} & > {lon_lower}"
        rslt = Vizier.query_constraints(catalog="II/246", GLAT=lat,
                                        Kmag='<7', GLON=lon)[0]
        crds = coord.SkyCoord(rslt['RAJ2000'], rslt['DEJ2000'], frame='fk5', unit=(u.deg, u.deg)).galactic

