import numpy as np
from astropy import units as u
from astropy import constants as c
from random import choice
from astropy.table import Table
from astropy.io import fits
import matplotlib.pyplot as plt
from astroquery.vizier import Vizier

from astropy.io import fits
import glob
from spectral_cube import lower_dimensional_structures
import pylab as pl
from astropy import visualization

from astropy import coordinates as coord
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy import table
from astropy.utils.console import ProgressBar
from photutils.datasets import make_random_gaussians_table, make_model_sources_image

from astropy import visualization

import functions
from sensitivity import (wl_paa, throughput, paa_bandwidth, paac_bandwidth)

models_table = Table.read('coelho14_model_paa.fits')
miris_info = {'pixscale': 51.6*u.arcsec, 'fov': (3.67)*u.deg.to(u.arcsec), 
	'readnoise': 45*u.count, 'dark_rate': 0.67*u.count/u.s, 'exptime': 8*u.min,
	'imsize': 256, 'diameter': 80 * u.mm}

pashion_info = {'pixscale': 0.806*u.arcsec, 'fov': 0.806*u.arcsec * 2048, 
	'readnoise': 22*u.count, 'dark_rate': 0.435*u.count/u.s, 'exptime': 500*u.s,
	'imsize': 2048, 'diameter': 24 * u.cm}

jhk_info = {'pixscale': 0.0251953125*u.arcsec, 'fov': 51.68*u.arcsec * 2048, 
	'readnoise': 22*u.count, 'dark_rate': 0.435*u.count/u.s, 'exptime': 500*u.s,
	'imsize': 2048, 'diameter': 24 * u.cm}

kmag_threshold=8.5
max_rows=int(1e6)
wavelength=wl_paa
pa_wavelength = wl_paa
pa_energy = pa_wavelength.to(u.erg, u.spectral())
pa_freq = pa_wavelength.to(u.Hz, u.spectral())
linename='paa'
bandwidth=paa_bandwidth
bandwidth_Hz = ((bandwidth / wavelength) * pa_freq).to(u.Hz)
transmission_fraction=throughput
hii=False

def value_keys(models_table):
	logg_values = []
	teff_values = []
	
	#creates a list of all of the possible logg and teff values
	#this makes matching quicker
	for value in models_table['logg']:
		if value not in logg_values:
			logg_values.append(value)
		
	for value in models_table['teff']:
		if value not in teff_values:
			teff_values.append(value)

	teff_dict = {"attribute": "effective temperature"}
	logg_dict = {"attribute": "surface gravity (log base 10)"}
	
	#this stores all of the indices for each teff and logg value in a dictionary
	#for easy access since those indices don't change from query to query
	for value in teff_values:
		indices = []
		for ii in range(len(models_table)):
			if models_table['teff'][ii] == value:
				indices.append(ii)
		teff_dict[f'value is {value}'] = indices
	
	for value in logg_values:
		indices = []
		for ii in range(len(models_table)):
			if models_table['logg'][ii] == value:
				indices.append(ii)
		logg_dict[f'value is {value}'] = indices

	return teff_dict, logg_dict, teff_values, logg_values




teff_dict, logg_dict, teff_values, logg_values = value_keys(models_table)




def correct_keys(jj, models_table, tbl):
	keys = []
	
	#this gives which teff and logg value the star is closest to
	init_min = np.argmin(np.abs(teff_values - tbl['teff'][jj]))
	teff_val = teff_values[init_min]
	init_min = np.argmin(np.abs(logg_values - tbl['logg'][jj]))
	logg_val = logg_values[init_min]
	
	#this gives the indices that correspond to that teff and logg value
	keys_teff = teff_dict[f'value is {teff_val}']
	keys_logg = logg_dict[f'value is {logg_val}']
	
	#this looks for any overlap between indices for teff and logg for a star
	for key in keys_teff:
		if key in keys_logg:
			keys.append(key)
	
	#if overlap can't be found, only the keys for teff are considered because that
	#seemed more important than prioriting logg
	#the choice function is to randomize which index is chosen 
	#since a lot of stars correspond to a particular teff logg combo
	if len(keys) == 0:
		index = choice(keys_teff)
	else:
		index = choice(keys)
	
	return index



def closest_model(name, limit=6, VVV=False):
	""""
	This assumes that we value teff being correct over logg.
	Unfortunately, trying to find the best fit from existing data requires us to use a strict poset.
	So one has to be prioritized over the other.
	"""
	data_stars = np.loadtxt('./TRILEGAL_data/' + name, unpack=True)
	
	logTe = []
	logg = []
	Av = []
	J = []
	H = []
	Ks = []
	
	zpt_J = 1594*u.Jy
	zpt_H = 1024*u.Jy
	zpt_Ks = 666.8*u.Jy
	
	index = 13
	
	#converting to the zeropoints for VVV
	if VVV:
		index += 2
		zpt_J -= 44.2*u.Jy
		zpt_H += 2.4*u.Jy
		zpt_Ks += 5.8*u.Jy
		
	zpts = Table({'J': [zpt_J],
				 'H': [zpt_H],
				 'Ks': [zpt_Ks],
				 'K': [zpt_Ks],
	})

	for ii in range(len(data_stars[0])):
		if ((data_stars[index, ii] >= limit) & (data_stars[index, ii] < (limit+1))):
			logTe.append(data_stars[5, ii])
			logg.append(data_stars[6, ii])
			Av.append(data_stars[8, ii])
			J.append(data_stars[index - 2, ii])
			H.append(data_stars[index - 1, ii])
			Ks.append(data_stars[index, ii])

	tbl = Table({'logTe': logTe,
				 'logg': logg,
				 'Av': Av,
				 'J': J,
				 'H': H,
				 'Ks': Ks,
				 })

	tbl.add_column(col=(10**tbl['logTe']), name='teff')
	
	data_rows = []
	#only keeping columns that seem relevant or interesting
	good_col = ['paa', 'paac_l', 'paac_h', 'J', 'H', 'K', 'fn']
	kept_col = ['Av', 'J', 'H', 'Ks']
	
	#faster way to make a table
	for jj in range(len(tbl)):
		#an index that prioritizes teff
		index = correct_keys(jj=jj, models_table=models_table, tbl=tbl)
		
		temp_dict = {}
		for hh in good_col:
			temp_dict[f'{hh} model'] = models_table[hh][index]
			
		for hh in kept_col:
			if hh is 'Av':
				temp_dict[f'{hh}'] = tbl[hh][jj]
				temp_dict['A_paa'] = np.round(0.15 * temp_dict['Av'], decimals=5)
			else:
				temp_dict[f'{hh} tril'] = tbl[hh][jj]
				temp_dict[f'{hh} flux tril'] = zpts[hh][0] * 10**(tbl[hh][jj] / -2.5)
		
		temp_dict['flux ratio'] = temp_dict['Ks flux tril'] / temp_dict['K model']
		temp_dict['f_paa'] = temp_dict['flux ratio'] * temp_dict['paa model']
		temp_dict['f_paac_l'] = temp_dict['flux ratio'] * temp_dict['paac_l model'] 
		temp_dict['f_paac_h'] = temp_dict['flux ratio'] * temp_dict['paac_h model'] 
		
		data_rows.append(temp_dict)
		
	k_avg = np.average(Ks, axis=0)
	result = Table(rows=data_rows)
	
	return result, k_avg



def flux_per_pix(pixels, name, VVV=False):
	new_data = closest_model(name, VVV=VVV)
	dimmed_flux = new_data['f_paa'] * 10**(new_data['A_paa']/-2.5)
	total_flux = dimmed_flux.sum()
	fpp = total_flux / (pixels**2)
	return fpp



def pixel_stars(glon, glat, width, height, model='miris'):

	if model is 'miris':
		pixscale = miris_info['pixscale']
		fov = miris_info['fov']
		readnoise = miris_info['readnoise']
		dark_rate = miris_info['dark_rate']
		exptime = miris_info['exptime']
		imsize = miris_info['imsize']
		diameter = miris_info['diameter']

	elif model is 'pashion':
		pixscale = pashion_info['pixscale']
		fov = pashion_info['fov']
		readnoise = pashion_info['readnoise']
		dark_rate = pashion_info['dark_rate']
		exptime = pashion_info['exptime']
		imsize = pashion_info['imsize']
		diameter = pashion_info['diameter']

	else:
		pixscale = jhk_info['pixscale']
		fov = jhk_info['fov']
		readnoise = jhk_info['readnoise']
		dark_rate = jhk_info['dark_rate']
		exptime = jhk_info['exptime']
		imsize = jhk_info['imsize']
		diameter = jhk_info['diameter']
		

	#glon = (right + left)/2 * u.deg
	#glat = (top + bottom)/2 * u.deg
	#width = (right - left) * u.deg
	#height = (top - bottom) * u.deg

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

	Viz = Vizier(row_limit=max_rows)
	cats = Viz.query_region(SkyCoord(glon, glat, frame='galactic'),
		height=height, width=width, catalog=["II/348", "II/246"])

	try:
		cat2 = cats['II/348/vvv2']
	except:
		print('no vvv catalog')
	else:
		cat2c = SkyCoord(cat2['RAJ2000'], cat2['DEJ2000'], frame='fk5',
			unit=(u.deg, u.deg)).galactic
		
		pix_coords_vvv = target_image_wcs.wcs_world2pix(cat2c.l.deg, cat2c.b.deg, 0)

		fluxes = functions.flux_function(hmag=cat2["Hmag3"], kmag=cat2['Ksmag3'],
									 wavelength=wavelength, VVV=True)

		phot_fluxes = fluxes / pa_energy * u.photon

		phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
					bandwidth_Hz).decompose()
		phot_ct = (phot_ct_rate * exptime).to(u.ph).value

		cat2.add_column(col=phot_ct, name=f'{linename}_phot_ct')
		cat2.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
		cat2.add_column(col=fluxes, name=f'{linename}_flux')

		nsrc = len(phot_ct_rate)

		x = pix_coords_vvv[0]
		y = pix_coords_vvv[1]

	#Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
		source_table = Table({'amplitude': phot_ct * transmission_fraction,
							'x_mean': np.round(x),
							'y_mean': np.round(y),
							'x_0': x,
							'y_0': y,
							'radius': np.repeat(airy_radius/pixscale, nsrc),
							'x_stddev': abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
							'y_stddev': abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
							'theta': np.pi * (x-1024),
							})

	try:
		cat2mass = cats['II/246/out']
	except:
		print('no 2MASS catalog')
	else:
		coords2mass = SkyCoord(cat2mass['RAJ2000'], cat2mass['DEJ2000'],
						   frame='fk5', unit=(u.deg, u.deg)).galactic

		pix_coords_2mass = target_image_wcs.wcs_world2pix(coords2mass.l.deg,
													  coords2mass.b.deg, 0)

		fluxes = functions.flux_function(hmag=cat2mass['Hmag'], kmag=cat2mass['Kmag'],
									 wavelength=wavelength)

		phot_fluxes = fluxes / pa_energy * u.photon

		phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
					bandwidth_Hz).decompose()
		phot_ct = (phot_ct_rate * exptime).to(u.photon).value


		cat2mass.add_column(col=phot_ct, name=f'{linename}_phot_ct')
		cat2mass.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
		cat2mass.add_column(col=fluxes, name=f'{linename}_flux')

		nsrc = len(phot_ct_rate)

		x = pix_coords_2mass[0]
		y = pix_coords_2mass[1]

	#Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
		source_table_2mass = Table({'amplitude': phot_ct * transmission_fraction,
									'x_mean': np.round(x),
									'y_mean': np.round(y),
									'x_0': x,
									'y_0': y,
									'radius': np.repeat(airy_radius/pixscale, nsrc),
									'x_stddev' : abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
									'y_stddev' : abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
									'theta' : np.pi * (x-1024),
									})
	try:
		source_table_both = table.vstack([source_table, source_table_2mass])
	except:
		try: 
			source_table_2mass
		except:
			return source_table, cat2, header
		else:
			return source_table_2mass, cat2mass, header

	return source_table_both, cat2, cat2mass, header

#make SED plot
#this is PASHION data
def make_sed_plot(dict_fn):
	teffs = []
	data = []
	labels = []

	for key in dict_fn.keys():
		filenames = dict_fn[key]
		teff_key = []
		data_key = []

		for fn in filenames:
			fh = fits.open(fn)
			header = fh[0].header
			sp = lower_dimensional_structures.OneDSpectrum.from_hdu(fh)
			#sp = specutils.Spectrum1D(data=fh[0].data, wcs=wcs.WCS(header), meta={'header': header})
			x = 10**sp.spectral_axis * u.AA

			sel = (x > 18400 * u.AA) & (x < 19100 * u.AA)

			teff = header['TEFF']
			normcolor = (teff - 3000)/10000
			color = pl.cm.jet(normcolor)

			teff_key.append(teff)
			data_key.append(sp[sel])
			xsel = x[sel]

		teff = np.average(teff_key, axis=0)
		datum = np.average(data_key, axis=0)
		teffs.append(teff)
		data.append(datum)
		labels.append(key)

	data = np.array(data)
	ndata = data / data[:,-1:]
	newx = np.linspace(xsel.min(), xsel.max(), 200)
	from scipy.interpolate import interp1d
	ndata = interp1d(xsel, ndata, kind='cubic')(newx)

	norm = visualization.simple_norm(teffs)

	segments = np.array([list(zip(newx.value,d)) for d in ndata])


	fig = pl.figure(1)
	fig.clf()
	ax = pl.gca()
	colors = ['lightpink', 'hotpink', 'r', 
			'darkorange', 'y', 'lime', 
			'green', 'deepskyblue', 'blue',
			'darkblue', 'indigo', 'darkviolet']
	lines = pl.matplotlib.collections.LineCollection(segments=segments,
												 cmap='jet_r',
												 colors=colors,
												 alpha=1)
	lines.set_array(np.array(teffs))
	ax.add_collection(lines)

	transmission_curve_lower = (0.95 + np.random.randn(newx.size)/1000) * ((newx > 18610*u.AA) & (newx < 18710*u.AA))
	transmission_curve_paa = (0.95 + np.random.randn(newx.size)/1000) * ((newx > 18731*u.AA) & (newx < 18781*u.AA))
	transmission_curve_upper = (0.95 + np.random.randn(newx.size)/1000) * ((newx > 18800*u.AA) & (newx < 18900*u.AA))

	ax.plot(newx, np.array([transmission_curve_lower, transmission_curve_paa, transmission_curve_upper]).T, color='c')
	ax.scatter([1], [6], color=colors[0], marker='_', label='K:6-7')
	ax.scatter([1], [6], color=colors[1], marker='_', label='K:7-8')
	ax.scatter([1], [6], color=colors[2], marker='_', label='K:8-9')
	ax.scatter([1], [6], color=colors[3], marker='_', label='K:9-10')
	ax.scatter([1], [6], color=colors[4], marker='_', label='K:10-11')
	ax.scatter([1], [6], color=colors[5], marker='_', label='K:11-12')
	ax.scatter([1], [6], color=colors[6], marker='_', label='K:12-13')
	ax.scatter([1], [6], color=colors[7], marker='_', label='K:13-14')
	ax.scatter([1], [6], color=colors[8], marker='_', label='K:14-15')
	ax.scatter([1], [6], color=colors[9], marker='_', label='K:15-16')
	ax.scatter([1], [6], color=colors[10], marker='_',label='K:16-17')
	ax.scatter([1], [6], color=colors[11], marker='_',label='K:17-18')
	#ax.plot(xsel, ndata, linewidth=0.1, alpha=0.1, color=color)

	ax.set_xlim(18500, 19000)
	ax.set_ylim(0.9, 1.1)
	ax.set_xlabel("Wavelength [$\\AA$]")
	ax.set_ylabel("Normalized Spectral Luminosity")
	pl.legend(title='Magnitude', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
	pl.title('Average SED for Background Stars')
	#cb = pl.colorbar(mappable=lines)
	#cb.set_alpha(1)
	#cb.draw_all()
	#cb.set_label('Effective Temperature [K]')

	pl.savefig("model_sed_temp.png", bbox_inches='tight')