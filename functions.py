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

def make_model_sources_image_faster(shape, model, source_table,
                                    bbox_size=5,
                                    progressbar=False):
    """
    Make an image containing sources generated from a user-specified
    model.
    FASTER VERSION: only add to a small cutout (a 'bbox')

    Parameters
    ----------
    shape : 2-tuple of int
        The shape of the output 2D image.

    model : 2D astropy.modeling.models object
        The model to be used for rendering the sources.

    source_table : `~astropy.table.Table`
        Table of parameters for the sources.  Each row of the table
        corresponds to a source whose model parameters are defined by
        the column names, which must match the model parameter names.
        Column names that do not match model parameters will be ignored.
        Model parameters not defined in the table will be set to the
        ``model`` default value.

    bbox_size : float
        Size of bounding box in number of radii...

    Returns
    -------
    image : 2D `~numpy.ndarray`
        Image containing model sources.

    See Also
    --------
    make_random_models_table, make_gaussian_sources_image

    Examples
    --------
    .. plot::
        :include-source:

        from collections import OrderedDict
        from astropy.modeling.models import Moffat2D
        from photutils.datasets import (make_random_models_table,
                                        make_model_sources_image)

        model = Moffat2D()
        n_sources = 10
        shape = (100, 100)
        param_ranges = [('amplitude', [100, 200]),
                        ('x_0', [0, shape[1]]),
                        ('y_0', [0, shape[0]]),
                        ('gamma', [5, 10]),
                        ('alpha', [1, 2])]
        param_ranges = OrderedDict(param_ranges)
        sources = make_random_models_table(n_sources, param_ranges,
                                           random_state=12345)

        data = make_model_sources_image(shape, model, sources)
        plt.imshow(data)
    """

    image = np.zeros(shape, dtype=np.float64)
    yidx, xidx = np.indices(shape)

    params_to_set = []
    for param in source_table.colnames:
        if param in model.param_names:
            params_to_set.append(param)

    # Save the initial parameter values so we can set them back when
    # done with the loop.  It's best not to copy a model, because some
    # models (e.g. PSF models) may have substantial amounts of data in
    # them.
    init_params = {param: getattr(model, param) for param in params_to_set}

    if not progressbar:
        progressbar = lambda x: x

    try:
        for source in progressbar(source_table):
            for param in params_to_set:
                setattr(model, param, source[param])

            # ONLY applies to airy!
            model.bounding_box = [(model.y_0-bbox_size*model.radius,
                                   model.y_0+bbox_size*model.radius),
                                  (model.x_0-bbox_size*model.radius,
                                   model.x_0+bbox_size*model.radius)]

            model.render(image)
    finally:
        for param, value in init_params.items():
            setattr(model, param, value)

    return image


def make_turbulent_im(size, readnoise, bias, dark, exptime, nstars=None,
                      sources=None, counts=10000, fwhm=3.2, power=3,
                      skybackground=False, sky=20, hotpixels=False,
                      biascol=False, brightness=1, progressbar=False):
    """
    Parameters
    ----------
    size : int
        Size of the square image in pixels
    readnoise : float
        Amplitude of the Gaussian readnoise to be added to the image
    bias : float
        Bias offset value
    dark : float
        Dark current rate in phot/s
    exptime : float
        Integration time in seconds
    stars : None or int
        The number of random stars to create
    sources : tbl
        A source table similar to that created by `make_random_gaussians_table`.
        Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    """

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
        sky_im = np.random.poisson(sky, size=[size, size])
        dark_bias_noise_im += sky_im
    ##stars
    flux_range = [counts/100, counts]
    y_max, x_max = shape
    xmean_range = [0.01 * x_max, 0.99 * x_max]
    ymean_range = [0.01 * y_max, 0.99 * y_max]
    xstddev_range = [1, 4]
    ystddev_range = [1, 4]
    model = models.AiryDisk2D(fwhm)
    model.bounding_box = [(-5*fwhm, 5*fwhm), (-5*fwhm, 5*fwhm)]
    params = dict([('amplitude', flux_range),
                  ('x_mean', xmean_range),
                  ('y_mean', ymean_range),
                  ('x_stddev', xstddev_range),
                  ('y_stddev', ystddev_range),
                  ('theta', [0, 2*np.pi])])
    if sources is None and nstars is not None:
        sources = make_random_gaussians_table(nstars, params,
                                              random_state=12345)
    elif sources is None and nstars is None:
        raise ValueError("Must specify either nstars or sources")
    star_im = make_model_sources_image_faster(shape, model, sources,
                                              progressbar=progressbar)
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
def skysurvey(lat_lower, lat_upper, lon_lower, lon_upper, VVV=False,
              pixel_scale=0.75, patience=3000, rowlim=3e5):
    #lat can be negative but lon must be between [0,360)
    Vizier = Vizier(timeout=patience)
    Vizier.ROW_LIMIT = rowlim
    if VVV:
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
        # II/246 = 2MASS
        rslt = Vizier.query_constraints(catalog="II/246", GLAT=lat,
                                        Kmag='<7', GLON=lon)[0]
        crds = coord.SkyCoord(rslt['RAJ2000'], rslt['DEJ2000'], frame='fk5', unit=(u.deg, u.deg)).galactic

