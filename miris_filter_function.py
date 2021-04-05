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
from miris_functions import pixel_stars, closest_model, make_sed_plot

import pandas 
import csv 

pl.rcParams['image.origin'] = 'lower'
#pl.style.use('dark_background')

surfarea = 4*np.pi*c.R_sun**2
unit = u.Jy

#dictionary compilation:
#location: glon=0, glat=0
#area = 0.0075deg^2
result_6_to_7_m = closest_model('center6_7_m.dat')
result_6_to_7_v = closest_model('center6_7_v.dat', VVV=True)
result_7_to_8_m = closest_model('center6_8_m.dat', limit=7)
result_7_to_8_v = closest_model('center6_8_v.dat', limit=7, VVV=True)
#location: glon=0, glat=0
#area = 0.0005deg^2
result_8_to_9_m = closest_model('center6_9_m.dat', limit=8)
result_8_to_9_v = closest_model('center6_9_v.dat', limit=8, VVV=True)
result_9_to_10_m = closest_model('center6_10_m.dat', limit=9)
result_9_to_10_v = closest_model('center6_10_v.dat', limit=9, VVV=True)
result_10_to_11_m = closest_model('center6_11_m.dat', limit=10)
result_10_to_11_v = closest_model('center6_11_v.dat', limit=10, VVV=True)
result_11_to_12_m = closest_model('center6_12_m.dat', limit=11)
result_11_to_12_v = closest_model('center6_12_v.dat', limit=11, VVV=True)
result_12_to_13_m = closest_model('center6_13_m.dat', limit=12)
result_12_to_13_v = closest_model('center6_13_v.dat', limit=12, VVV=True)
#location: glon=0, glat=0
#area = 0.00005deg^2
result_13_to_14_m = closest_model('center6_14_m.dat', limit=13)
result_13_to_14_v = closest_model('center6_14_v.dat', limit=13, VVV=True)
result_14_to_15_m = closest_model('center6_15_m.dat', limit=14)
result_14_to_15_v = closest_model('center6_15_v.dat', limit=14, VVV=True)
result_15_to_16_m = closest_model('center6_16_m.dat', limit=15)
result_15_to_16_v = closest_model('center6_16_v.dat', limit=15, VVV=True)
result_16_to_17_m = closest_model('center6_17_m.dat', limit=16)
result_16_to_17_v = closest_model('center6_17_v.dat', limit=16, VVV=True)
#location: glon=0, glat=0
#area = 0.000005deg^2
result_17_to_18_m = closest_model('center6_18_m.dat', limit=17)
result_17_to_18_v = closest_model('center6_18_v.dat', limit=17, VVV=True)
data_dict_m = {}
data_dict_m['K6-7'] = result_6_to_7_m['fn model']
data_dict_m['K7-8'] = result_7_to_8_m['fn model']
data_dict_m['K8-9'] = result_8_to_9_m['fn model']
data_dict_m['K9-10'] = result_9_to_10_m['fn model']
data_dict_m['K10-11'] = result_10_to_11_m['fn model']
data_dict_m['K11-12'] = result_11_to_12_m['fn model']
data_dict_m['K12-13'] = result_12_to_13_m['fn model']
data_dict_m['K13-14'] = result_13_to_14_m['fn model']
data_dict_m['K14-15'] = result_14_to_15_m['fn model']
data_dict_m['K15-16'] = result_15_to_16_m['fn model']
data_dict_m['K16-17'] = result_16_to_17_m['fn model']
data_dict_m['K17-18'] = result_17_to_18_m['fn model']
data_dict_v = {}
data_dict_v['K6-7'] = result_6_to_7_v['fn model']
data_dict_v['K7-8'] = result_7_to_8_v['fn model']
data_dict_v['K8-9'] = result_8_to_9_v['fn model']
data_dict_v['K9-10'] = result_9_to_10_v['fn model']
data_dict_v['K10-11'] = result_10_to_11_v['fn model']
data_dict_v['K11-12'] = result_11_to_12_v['fn model']
data_dict_v['K12-13'] = result_12_to_13_v['fn model']
data_dict_v['K13-14'] = result_13_to_14_v['fn model']
data_dict_v['K14-15'] = result_14_to_15_v['fn model']
data_dict_v['K15-16'] = result_15_to_16_v['fn model']
data_dict_v['K16-17'] = result_16_to_17_v['fn model']
data_dict_v['K17-18'] = result_17_to_18_v['fn model']

#import pdb; pdb.set_trace()
def many_small_lines(xvals, x_data, y_data):
	f_x = []
	pt = 1
#	pdb.set_trace()
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
    x.append(num*10*u.AA)

for ii, xval in enumerate(csvFile['x vals']):
	if xval not in x_data:
		x_data.append(xval*10)
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
		x_pashen.append(xval*10)
		x_units_pa.append(xval*10*u.AA)
		index.append(ii)
for ii in index:
	yval = csvFile['y vals'][ii]
	y_pashen.append(yval)


#linefit_cont = many_small_lines(x_temp, x_data, y_data)
#linefit_pa = many_small_lines(x_temp, x_pashen, y_pashen)

y_cont = many_small_lines(x, x_units, y_data)
y_pa = many_small_lines(x, x_units_pa, y_pashen)

#if __name__ == "__main__":
#	pl.figure(figsize=(10,5))
#	pl.title('MIRIS Filter', fontsize='20')
#	pl.scatter(x_data, y_data, marker='*', color='hotpink')
#	pl.plot(x, linefit, color='c')
#	pl.xlabel('Wavelength (nm)')
#	pl.ylabel('Transmittance %')
#	pl.ylim(-1,110)
#	pl.xlim(1800,1950)



def make_sed_flux(dict_fn, x_units, y_data, x_units_pa, y_pashen):
	
	teffs = []
	data = []
	labels = []
	flux_pa = []
	flux_paacl = []
	flux_paach = []
	filter_pa = []
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
			x_new = 10**sp.spectral_axis * u.AA
			#pdb.set_trace()
			sel = (x_new > 18000 * u.AA) & (x_new < 19500 * u.AA)
			mid_pt = int(len(sel)/2)
			filter_func_cont = many_small_lines(x_new[sel], x, y_cont)
			filter_func_pa = many_small_lines(x_new[sel], x, y_pa)
			spectra = sp[sel]
			f_paa = np.dot(spectra, filter_func_pa)*unit #* surfarea
			f_paacl = np.dot(spectra, filter_func_cont)*unit
			#f_paacl = np.dot(spectra[0:mid_pt], filter_func_cont[0:mid_pt])*unit #* surfarea
			#f_paach = np.dot(spectra[mid_pt:], filter_func_cont[mid_pt:])*unit #* surfarea

			#teff_key.append(teff)
			unitz = f_paa.unit
			data_key.append(sp[sel])
			paa_key.append(f_paa.value)
			paacl_key.append(f_paacl.value)
			filter_pa.append(filter_func_pa)
			#paach_key.append(f_paach.value)
			xsel = x_new[sel]

		#teff = np.average(teff_key, axis=0)
		datum = np.average(data_key, axis=0)
		paa = np.average(paa_key, axis=0) * unitz
		paacl = np.average(paacl_key, axis=0) * unitz
		#paach = np.average(paach_key, axis=0) * unitz
		#teffs.append(teff)
		data.append(datum)
		labels.append(key)
		flux_pa.append(paa)
		flux_paacl.append(paacl)
		#flux_paach.append(paach)

	return flux_pa, flux_paacl, filter_pa, y_pa#flux_paach
#pdb.set_trace()

flux_pa, flux_paacl, filter_pa, y_pa = make_sed_flux(data_dict_m, x_units, y_data, x_units_pa, y_pashen)