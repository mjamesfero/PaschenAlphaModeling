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
			yval = (slope*(x_val-xval_1)+yval_1)*0.01
		else:
			init_min = pt  
			init_max = pt + 1
			try:
				xval_1 = x_data[init_min]
				xval_2 = x_data[init_max]
				yval_1 = y_data[init_min]
				yval_2 = y_data[init_max]
				slope = (yval_2-yval_1)/(xval_2-xval_1)
				yval = (slope*(x_val-xval_1)+yval_1)*0.01
				pt += 1 
			except:
				yval = (y_data[init_min])*0.01
			
		
		f_x.append(yval)
			
	return f_x

def refine(data1, data2):
	data1_new = []
	data2_new = []
	for ii, xval in enumerate(data1):
		if xval not in data1_new:
			data1_new.append(xval)
			index.append(ii)
			yval = data2[ii]
			data2_new.append(yval)
	return data1_new, data2_new


surfarea = 4*np.pi*c.R_sun**2
unit = u.Jy
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

dict_fn = {}
ranges = ['K6-7', 'K7-8', 'K8-9', 'K9-10', 'K10-11', 'K11-12', 'K12-13', 'K13-14', 
        'K14-15', 'K15-16', 'K16-17', 'K17-18']
csvFile = pandas.read_csv('model_names_m.csv')
index = [[],[],[],[],[],[],[],[],[],[],[],[]]
for ii, xval in enumerate(csvFile['range']):
    if xval == ranges[0]:
        index[0].append(ii)
    elif xval == ranges[1]:
        index[1].append(ii)
    elif xval == ranges[2]:
        index[2].append(ii)
    elif xval == ranges[3]:
        index[3].append(ii)
    elif xval == ranges[4]:
        index[4].append(ii)
    elif xval == ranges[5]:
        index[5].append(ii)
    elif xval == ranges[6]:
        index[6].append(ii)
    elif xval == ranges[7]:
        index[7].append(ii)
    elif xval == ranges[8]:
        index[8].append(ii)
    elif xval == ranges[9]:
        index[9].append(ii)
    elif xval == ranges[10]:
        index[10].append(ii)
    elif xval == ranges[11]:
        index[11].append(ii)

names = [[],[],[],[],[],[],[],[],[],[],[],[]]
for jj in range(len(index)):
    for ii in index[jj]:
        name = csvFile['fn'][ii]
        names[jj].append(name)

for ii, group in enumerate(ranges):
    dict_fn[group] = names[ii]


csvFile = pandas.read_csv('ks_mags.csv') 
ks_mags = []
for ii in range(12):
        mag = csvFile['mag'][ii]
        ks_mags.append(mag)

y_cont_rough = many_small_lines(x, x_units, y_data)
y_pa_rough = many_small_lines(x, x_units_pa, y_pashen)
y_cont, x_pa  = refine(y_cont_rough, x)
y_pa, x_cont = refine(y_pa_rough, x)



def make_sed_flux(dict_fn, x_units, y_data, x_units_pa, y_pashen):
	data = []
	labels = []
	flux_pa = []
	flux_paacl = []
	flux_paach = []
	filter_pa = []
	for key in dict_fn.keys():
		filenames = dict_fn[key]
		data_key = []
		paa_key = []
		paacl_key = []
		paach_key = []
#
		for fn in filenames:
			fh = fits.open(fn)
			sp = lower_dimensional_structures.OneDSpectrum.from_hdu(fh)
			x_new = 10**sp.spectral_axis * u.AA
			sel = (x_new > 18000 * u.AA) & (x_new < 19500 * u.AA)
			mid_pt = int(len(sp[sel])/2)
			filter_func_cont = many_small_lines(x_new[sel], x, y_cont)
			filter_func_pa = many_small_lines(x_new[sel], x, y_pa)
			spectra = sp[sel]
			f_paa = np.dot(spectra, filter_func_pa)*unit #* surfarea
			#f_paacl = np.dot(spectra, filter_func_cont)*unit
			f_paacl = np.dot(spectra[0:mid_pt], filter_func_cont[0:mid_pt])*unit #* surfarea
			f_paach = np.dot(spectra[mid_pt:], filter_func_cont[mid_pt:])*unit #* surfarea

			#teff_key.append(teff)
			unitz = f_paa.unit
			data_key.append(sp[sel])
			paa_key.append(f_paa.value)
			paacl_key.append(f_paacl.value)
			filter_pa.append(filter_func_pa)
			paach_key.append(f_paach.value)
			#xsel = x_new[sel]

		#teff = np.average(teff_key, axis=0)
		datum = np.average(data_key, axis=0)
		paa = np.average(paa_key, axis=0) * unitz
		paacl = np.average(paacl_key, axis=0) * unitz
		paach = np.average(paach_key, axis=0) * unitz
		#teffs.append(teff)
		data.append(datum)
		labels.append(key)
		flux_pa.append(paa)
		flux_paacl.append(paacl)
		flux_paach.append(paach)

	return flux_pa, flux_paacl, flux_paach
#pdb.set_trace()

flux_pa, flux_paacl, flux_paach = make_sed_flux(dict_fn, x_units, y_data, x_units_pa, y_pashen)
fluxes = {}
fluxes['paa'] = flux_pa
fluxes['paacl'] = flux_paacl
fluxes['paach'] = flux_paach

def sed_flux_function(kmags, wavelength):
	flux_int = fluxes[wavelength]
	flux_fin = []
	for kmag in kmags:
		if 6 <= kmag < 7:
			mag_a = ks_mags[0]
			flux_a = flux_int[0].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 7 <= kmag < 8:
			mag_a = ks_mags[1]
			flux_a = flux_int[1].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 8 <= kmag < 9:
			mag_a = ks_mags[2]
			flux_a = flux_int[2].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 9 <= kmag < 10:
			mag_a = ks_mags[3]
			flux_a = flux_int[3].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 10 <= kmag < 11:
			mag_a = ks_mags[4]
			flux_a = flux_int[4].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 11 <= kmag < 12:
			mag_a = ks_mags[5]
			flux_a = flux_int[5].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 12 <= kmag < 13:
			mag_a = ks_mags[6]
			flux_a = flux_int[6].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 13 <= kmag < 14:
			mag_a = ks_mags[7]
			flux_a = flux_int[7].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 14 <= kmag < 15:
			mag_a = ks_mags[8]
			flux_a = flux_int[8].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 15 <= kmag < 16:
			mag_a = ks_mags[9]
			flux_a = flux_int[9].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 16 <= kmag < 17:
			mag_a = ks_mags[10]
			flux_a = flux_int[10].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		elif 17 <= kmag <= 18:
			mag_a = ks_mags[11]
			flux_a = flux_int[11].value
			flux_b = flux_a*10**((mag_a-kmag)/2.5)
		else:
			print(f'You fucked up! Mag is {kmag}!')
			flux_b=69
		flux_fin.append(flux_b)
	return flux_fin
