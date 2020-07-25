from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, convolve_models
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.modeling import models
from astroquery.vizier import Vizier
from matplotlib import pyplot as plt
from photutils.datasets import make_random_gaussians_table, make_model_sources_image
from scipy.signal import convolve
from turbustat.simulator import make_extended
import astropy.coordinates as coord
from radio_beam import Beam
from astropy.io import fits
import astropy.units as u
import numpy as np

plt.rcParams['image.origin'] = 'lower'

def make_model_sources_image_faster(shape, airy_radius, source_table,
									bbox_size=10,
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
	#needed: to have each model=convolve have [souces] in it 
	#while still doing everything else
	#models = for ii in len(source_table):
	#            convolve_models(models.Gaussian2D(source_table['amplitude_0'][ii], 
	#                                        source_table['x_mean_0'][ii], 
	#                                        source_table['y_mean_0'][ii],
	#                                        source_table['x_stddev_0'][ii],
	#                                        source_table['y_stddev_0'][ii],
	#                                        source_table['theta_0'][ii]), 
	#                        models.AiryDisk2D(source_table['amplitude_1'][ii],
	#                                        source_table['x_0_1'][ii],
	#                                        source_table['y_0_1'][ii],
	#                                        source_table['theta_0'][ii]))

	model1 = models.AiryDisk2D(airy_radius)
	model2 = models.Gaussian2D()
	model1.bounding_box = [(-5*airy_radius, 5*airy_radius), (-5*airy_radius, 5*airy_radius)]
	model2.bounding_box = [(-5*airy_radius, 5*airy_radius), (-5*airy_radius, 5*airy_radius)]

	image = np.zeros(shape, dtype=np.float64)
	yidx, xidx = np.indices(shape)

	test = []

	params_to_set1 = []
	for param in source_table.colnames:
		if param in model1.param_names:
			params_to_set1.append(param)

	params_to_set2 = []
	for param in source_table.colnames:
		if param in model2.param_names:
			params_to_set2.append(param)

	# Save the initial parameter values so we can set them back when
	# done with the loop.  It's best not to copy a model, because some
	# models (e.g. PSF models) may have substantial amounts of data in
	# them.

	init_params1 = {param: getattr(model1, param) for param in params_to_set1}
	init_params2 = {param: getattr(model2, param) for param in params_to_set2}

	if not progressbar:
		progressbar = lambda x: x

	try:
		for source in progressbar(source_table):
			for param in params_to_set1:
				setattr(model1, param, source[param])

			for param in params_to_set2:
				setattr(model2, param, source[param])
			# ONLY applies to airy!
			model = convolve_models(model1, model2)
			model.bounding_box = [(model1.y_0-bbox_size*model1.radius,
								   model1.y_0+bbox_size*model1.radius),
								  (model1.x_0-bbox_size*model1.radius,
								   model1.x_0+bbox_size*model1.radius)]

		  
			model.render(image)

	finally:
		for param, value in init_params1.items():
			setattr(model1, param, value)
		for param, value in init_params2.items():
			setattr(model2, param, value)

	return image


def make_stars_im(size, readnoise=10*u.count, bias=0*u.count,
				  dark_rate=1*u.count/u.s, exptime=1*u.s, nstars=None,
				  sources=None, counts=10000, airy_radius=3.2, skybackground=False,
				  sky=20, hotpixels=False, biascol=False, progressbar=False,
				  seed=8392):
	"""
	Parameters
	----------
	size : int
		Size of the square image in pixels
	readnoise : quantity in counts
		Amplitude of the Gaussian readnoise to be added to the image
	bias : quantity in counts
		Bias offset value
	dark_rate : quantity in count/s
		Dark current rate
	exptime : float
		Integration time in seconds
	stars : None or int
		The number of random stars to create
	airy_radius : float
		airy_radius of the Airy function in *pixel* units
	sources : tbl
		A source table similar to that created by `make_random_gaussians_table`.
		Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
	"""

	blank_image = np.zeros([size, size])
	## based off of https://mwcraig.github.io/ccd-as-book/01-03-Construction-of-an-artificial-but-realistic-image.html
	##pretty much just copy paste and combined functions
	##readnoise
	shape = blank_image.shape
	noise = np.random.normal(scale=readnoise.decompose().value, size=shape)
	noise_im = blank_image + noise
	##bias
	bias_im = np.zeros_like(blank_image) + bias.decompose().value
	if biascol:
		number_of_colums = 5
		rng = np.random.RandomState(seed=seed)  # 20180520
		columns = rng.randint(0, shape[1], size=number_of_colums)
		col_pattern = rng.randint(0, int(0.1 * bias), size=shape[0])
		for c in columns:
			bias_im[:, c] = bias + col_pattern
	##readnoise and bias
	bias_noise_im = noise_im + bias_im
	##dark current
	base_current = (dark_rate * exptime).to(u.count).value
	dark_im = np.random.poisson(base_current, size=shape)
	##hot pixels
	if hotpixels:
		y_max, x_max = dark_im.shape
		n_hot = int(0.00002 * x_max * y_max)
		rng = np.random.RandomState(16201649)
		hot_x = rng.randint(0, x_max, size=n_hot)
		hot_y = rng.randint(0, y_max, size=n_hot)
		hot_current = 10000 * dark_rate
		dark_im[[hot_y, hot_x]] = hot_current * exptime
	##dark bias readnoise hot pixels
	dark_bias_noise_im = bias_noise_im + dark_im
	##optional skybackground
	if skybackground:
		sky_im = np.random.poisson(sky, size=[size, size])
		dark_bias_noise_im += sky_im
	## simulated stars
	
	if sources is None and nstars is not None:
		flux_range = [counts/100, counts]
		y_max, x_max = shape
		xmean_range = [0.01 * x_max, 0.99 * x_max]
		ymean_range = [0.01 * y_max, 0.99 * y_max]
		xstddev_range = [1, 4]
		ystddev_range = [1, 4]
		params = dict([('amplitude', flux_range),
					  ('x_mean', xmean_range),
					  ('y_mean', ymean_range),
					  ('x_stddev', xstddev_range),
					  ('y_stddev', ystddev_range),
					  ('theta', [0, 2*np.pi])])
		sources = make_random_gaussians_table(nstars, params,
											  random_state=12345)
	elif sources is None and nstars is None:
		raise ValueError("Must specify either nstars or sources")
	
	star_im = make_model_sources_image_faster(shape, airy_radius, sources,
											  progressbar=progressbar)
	#star_im = convolve(stars_alone, Gaussian2DKernel(sources['x_stddev'], sources['y_stddev'], sources['theta']), mode="same")
	##stars and background
	stars_background_im = star_im + dark_bias_noise_im

	return stars_background_im

def make_turbulence_im(size, airy_radius=3.2, power=3, brightness=1):
	##turbulence
	turbulent_data = make_extended(size, power)
	min_val = np.min(turbulent_data)
	turbulence = (turbulent_data - min_val + 1)*brightness
	turbulent_im = convolve(turbulence, AiryDisk2DKernel(airy_radius), mode="same")

	return turbulent_im

def make_turbulent_starry_im(size, readnoise, bias, dark_rate, exptime, nstars=None,
							 sources=None, counts=10000, airy_radius=3.2, power=3,
							 skybackground=False, sky=20, hotpixels=False,
							 biascol=False, brightness=1, progressbar=False):

	turbulent_im = make_turbulence_im(size=size, airy_radius=airy_radius, power=power,
									  brightness=brightness)

	stars_background_im = make_stars_im(size=size, readnoise=readnoise,
										bias=bias, dark_rate=dark_rate, exptime=exptime,
										nstars=nstars, sources=sources,
										counts=counts, airy_radius=airy_radius,
										skybackground=skybackground, sky=sky,
										hotpixels=hotpixels, biascol=biascol,
										progressbar=progressbar)

	##turbulent image with stars
	turbulent_stars = turbulent_im + stars_background_im

	return stars_background_im, turbulent_stars, turbulent_im    

def add_HII_region(region='W51-CBAND-feathered.fits', fov=27.5*u.arcmin*u.arcmin, 
				   airy_radius=3.2, power=3):
	
	header_HII = fits.getheader(region)  
	data_HII = fits.getdata(region)  
	HII_region = Beam.from_fits_header(header_HII) 

	size1 = header_HII["NAXIS1"]
	size2 = header_HII["NAXIS2"]

	fwhm_axis = header_HII["BMAJ"]*u.deg
	fwhm = fwhm_axis.to(u.arcsec)
	fwhm_to_sigma = 1. / (8 * np.log(2))**0.5
	beam_sigma = fwhm * fwhm_to_sigma
	omega_B = 2 * np.pi * beam_sigma**2
	S_PaA = 117*(data_HII*u.Jy/u.beam).to(u.MJy/u.sr, equivalencies=u.beam_angular_area(omega_B)) 

	area1 = fov.to(u.rad*u.rad)
	area = area1.to(u.sr)
	surface = S_PaA * area
	surface_brightness = (surface).reshape((size1, size2))

	hii_region = convolve(surface_brightness, AiryDisk2DKernel(airy_radius), mode="same")*u.MJy

	return hii_region

def make_HII_starry_im(readnoise, bias, dark_rate, exptime, size=2048, 
							 region='W51-CBAND-feathered.fits', 
							 nstars=None, fov=27.5*u.arcmin*u.arcmin, 
							 sources=None, counts=10000, airy_radius=3.2, power=3,
							 skybackground=False, sky=20, hotpixels=False,
							 biascol=False, progressbar=False):

	hii_im = add_HII_region(region=region, fov=fov,  
							airy_radius=airy_radius, power=power)

	stars_background_im = make_stars_im(size=size, readnoise=readnoise,
										bias=bias, dark_rate=dark_rate, exptime=exptime,
										nstars=nstars, sources=sources,
										counts=counts, airy_radius=airy_radius,
										skybackground=skybackground, sky=sky,
										hotpixels=hotpixels, biascol=biascol,
										progressbar=progressbar)
	stars_background = stars_background_im*u.Jy
	no_hii = stars_background.to(u.MJy)

	##turbulent image with stars
	hii_stars = hii_im + no_hii

	return no_hii, hii_stars, hii_im



def flux_function(hmag, kmag, wavelength=18750*u.AA, VVV=False):

	"""
	Parameters
	----------
	wavelength : float
		Desired wavelength in Angstroms
	hmag : float
		Given star's H magnitude
	kmag : float
		Given star's Ks magnitude
	VVV : Boolean
		Whether the star is from the VVV catalogs or 2MASS catalogs
	"""

	if VVV:
		m_mag = (kmag - hmag)/u.Quantity(21527.6 - 16508.7, u.AA)
		Pamag = m_mag*(wavelength - 16508.7*u.AA) + hmag
		m_zpt = (672.6 - 1026.4)/u.Quantity(21527.6 - 16508.7, u.AA)
		zpt = (m_zpt*(wavelength - 16508.7*u.AA) + 1026.4) * u.Jy
	else:
		m_mag = (kmag - hmag)/u.Quantity(21590.0 - 16620.0, u.AA)
		Pamag = m_mag*(wavelength - 16620.0*u.AA) + hmag
		m_zpt = (666.8 - 1024.0)/u.Quantity(21590.0 - 16620.0, u.AA)
		zpt = (m_zpt*(wavelength - 16620.0*u.AA) + 1024.0) * u.Jy
	flux_new = u.Quantity(10**(Pamag/ -2.5)) * zpt
	return flux_new

#lagrange interpolation formula. might give a better fit.
def flux_lagrange(jmag, hmag, kmag, zmag=1, ymag=1, wavelength=18750,
				  VVV=False):

	"""
	Parameters
	----------
	wavelength : float
		Desired wavelength in Angstroms
	zmag : float
		Given star's Z magnitude
	ymag : float
		Given star's Y magnitude
	jmag : float
		Given star's J magnitude
	hmag : float
		Given star's H magnitude
	kmag : float
		Given star's Ks magnitude
	VVV : Boolean
		Whether the star is from the VVV catalogs or 2MASS catalogs
	"""
	wave_len = wavelength
	if VVV:
		c0 = (wave_len - 8790.1)
		c1 = (wave_len - 10219.7)
		c2 = (wave_len - 12562.1)
		c3 = (wave_len - 16508.7)
		c4 = (wave_len - 21527.6)
		f0 = [c1*c2*c3*c4]/[(8790.1 - 10219.7)*(8790.1 - 12562.1)*(8790.1 - 16508.7)*(8790.1 - 21527.6)]
		f1 = [c0*c2*c3*c4]/[(10219.7 - 8790.1)*(10219.7 - 12562.1)*(10219.7 - 16508.7)*(10219.7 - 21527.6)]
		f2 = [c0*c1*c3*c4]/[(12562.1 - 10219.7)*(12562.1 - 8790.1)*(12562.1 - 16508.7)*(12562.1 - 21527.6)]
		f3 = [c0*c1*c2*c4]/[(16508.7 - 10219.7)*(16508.7 - 12562.1)*(16508.7 - 8790.1)*(16508.7 - 21527.6)]
		f4 = [c0*c1*c2*c3]/[(21527.6 - 10219.7)*(21527.6 - 12562.1)*(21527.6 - 16508.7)*(21527.6 - 8790.1)]
		new_mag = zmag*f0 + ymag*f1 + jmag*f2 + hmag*f3 + kmag*f4
		zpt = (2264.1*f0 + 2085.3*f1 + 1549.8*f2 + 1026.4*f3 + 672.6*f4)*u.Jy
	else:
		c2 = (wave_len - 12350.0)
		c3 = (wave_len - 16620.0)
		c4 = (wave_len - 21590.0)
		f2 = [c3*c4]/[(12350.0 - 16620.0)*(12350.0 - 21590.0)]
		f3 = [c2*c4]/[(16620.0 - 12350.0)*(16620.0 - 21590.0)]
		f4 = [c2*c3]/[(21590.0 - 12350.0)*(21590.0 - 16620.0)]
		new_mag = jmag*f2 + hmag*f3 + kmag*f4
		zpt = (1594.0*f2 + 1024.0*f3 + 666.8*f4)*u.Jy
	flux_new = u.Quantity(10**(new_mag/ -2.5)) * zpt
	return flux_new







