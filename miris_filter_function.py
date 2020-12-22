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

import pandas 
import csv 

pl.rcParams['image.origin'] = 'lower'
pl.style.use('dark_background')

surfarea = 4*np.pi*c.R_sun**2
unit = u.erg/u.s/u.cm**2

def many_small_lines(xvals, x_data, y_data):
	f_x = []
	pt = 1
	for x_val in xvals:
		
		if (pt < len(x_data)) and (x_val <= x_data[pt]):
			init_min = pt - 1 
			init_max = pt
			xval_1 = x_data[init_min]
			xval_2 = x_data[init_max]
			yval_1 = y_data[init_min]
			yval_2 = y_data[init_max]
			slope = (yval_2-yval_1)/(xval_2-xval_1)
			yval = (slope*(x_val-xval_1)+yval_1)#*0.01
		else:
			init_min = pt  
			init_max = pt + 1
			try:
				xval_1 = x_data[init_min]
				xval_2 = x_data[init_max]
				yval_1 = y_data[init_min]
				yval_2 = y_data[init_max]
				slope = (yval_2-yval_1)/(xval_2-xval_1)
				yval = (slope*(x_val-xval_1)+yval_1)#*0.01
				pt += 1 
			except:
				yval = (y_data[init_min])#*0.01
			
		
		f_x.append(yval)
			
	return f_x

csvFile = pandas.read_csv('MIRIS_continuum_filter_data.csv') 
x_data = []
x_units = []
y_data = []
index = []
x_temp = np.linspace(1800, 1950, 1000)
x = []
for num in x_temp:
    x.append(num*u.AA)

for ii, xval in enumerate(csvFile['x vals']):
	if xval not in x_data:
		x_data.append(xval)
		x_units.append(xval*10*u.AA)
		index.append(ii)
for ii in index:
	yval = csvFile['y vals'][ii]
	y_data.append(yval)

csvFile = pandas.read_csv('MIRIS_pashen_filter_data.csv') 
x_pashen = []
x_units_pa = []
y_pashen = []
index = []

for ii, xval in enumerate(csvFile['x vals']):
	if xval not in x_pashen:
		x_pashen.append(xval)
		x_units_pa.append(xval*10*u.AA)
		index.append(ii)
for ii in index:
	yval = csvFile['y vals'][ii]
	y_pashen.append(yval)

linefit_cont = many_small_lines(x_temp, x_data, y_data)
linefit_pa = many_small_lines(x_temp, x_pashen, y_pashen)

if __name__ == "__main__":
	pl.figure(figsize=(10,5))
	pl.title('MIRIS Filter', fontsize='20')
	pl.scatter(x_data, y_data, marker='*', color='hotpink')
	pl.plot(x, linefit, color='c')
	pl.xlabel('Wavelength (nm)')
	pl.ylabel('Transmittance %')
	pl.ylim(-1,110)
	pl.xlim(1800,1950)

def make_sed_flux(dict_fn, x_units, y_data, x_units_pa, y_pashen):
	teffs = []
	data = []
	labels = []
	flux_pa = []
	flux_paacl = []
	flux_paach = []

	for key in dict_fn.keys():
		filenames = dict_fn[key]
		teff_key = []
		data_key = []
		paa_key = []
		paacl_key = []
		paach_key = []

		for fn in filenames:
			fh = fits.open(fn)
			header = fh[0].header
			sp = lower_dimensional_structures.OneDSpectrum.from_hdu(fh)
			#sp = specutils.Spectrum1D(data=fh[0].data, wcs=wcs.WCS(header), meta={'header': header})
			#import pdb; pdb.set_trace()
			x = 10**sp.spectral_axis * u.AA

			sel = (x > 18000 * u.AA) & (x < 19500 * u.AA)
			mid_pt = int(len(sel)/2)
			filter_func_cont = many_small_lines(x[sel], x_units, y_data)
			filter_func_pa = many_small_lines(x[sel], x_units_pa, y_pashen)
			spectra = sp[sel]
			f_paa = np.dot(spectra, filter_func_pa)*unit * surfarea
			f_paacl = np.dot(spectra[0:mid_pt], filter_func_cont[0:mid_pt])*unit * surfarea
			f_paach = np.dot(spectra[mid_pt:], filter_func_cont[mid_pt:])*unit * surfarea

			#teff_key.append(teff)
			data_key.append(sp[sel])
			paa_key.append(f_paa)
			paacl_key.append(f_paacl)
			paach_key.append(f_paach)
			xsel = x[sel]

		#teff = np.average(teff_key, axis=0)
		datum = np.average(data_key, axis=0)
		paa = np.average(paa_key, axis=0)
		paacl = np.average(paacl_key, axis=0)
		paach = np.average(paach_key, axis=0)
		#teffs.append(teff)
		data.append(datum)
		labels.append(key)
		flux_pa.append(paa)
		flux_paacl.append(paacl)
		flux_paach.append(paach)

	return flux_pa, flux_paacl, flux_paach

