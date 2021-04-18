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
result_6_to_7_m, k_avg61 = closest_model('center6_7_m.dat')
result_6_to_7_v, k_avg62 = closest_model('center6_7_v.dat', VVV=True)
result_7_to_8_m, k_avg71 = closest_model('center6_8_m.dat', limit=7)
result_7_to_8_v, k_avg72 = closest_model('center6_8_v.dat', limit=7, VVV=True)
#location: glon=0, glat=0
#area = 0.0005deg^2
result_8_to_9_m, k_avg81 = closest_model('center6_9_m.dat', limit=8)
result_8_to_9_v, k_avg82 = closest_model('center6_9_v.dat', limit=8, VVV=True)
result_9_to_10_m, k_avg91 = closest_model('center6_10_m.dat', limit=9)
result_9_to_10_v, k_avg92 = closest_model('center6_10_v.dat', limit=9, VVV=True)
result_10_to_11_m, k_avg101 = closest_model('center6_11_m.dat', limit=10)
result_10_to_11_v, k_avg102 = closest_model('center6_11_v.dat', limit=10, VVV=True)
result_11_to_12_m, k_avg111 = closest_model('center6_12_m.dat', limit=11)
result_11_to_12_v, k_avg112 = closest_model('center6_12_v.dat', limit=11, VVV=True)
result_12_to_13_m, k_avg121 = closest_model('center6_13_m.dat', limit=12)
result_12_to_13_v, k_avg122 = closest_model('center6_13_v.dat', limit=12, VVV=True)
#location: glon=0, glat=0
#area = 0.00005deg^2
result_13_to_14_m, k_avg131 = closest_model('center6_14_m.dat', limit=13)
result_13_to_14_v, k_avg132 = closest_model('center6_14_v.dat', limit=13, VVV=True)
result_14_to_15_m, k_avg141 = closest_model('center6_15_m.dat', limit=14)
result_14_to_15_v, k_avg142 = closest_model('center6_15_v.dat', limit=14, VVV=True)
result_15_to_16_m, k_avg151 = closest_model('center6_16_m.dat', limit=15)
result_15_to_16_v, k_avg152 = closest_model('center6_16_v.dat', limit=15, VVV=True)
result_16_to_17_m, k_avg161 = closest_model('center6_17_m.dat', limit=16)
result_16_to_17_v, k_avg162 = closest_model('center6_17_v.dat', limit=16, VVV=True)
#location: glon=0, glat=0
#area = 0.000005deg^2
result_17_to_18_m, k_avg171 = closest_model('center6_18_m.dat', limit=17)
result_17_to_18_v, k_avg172 = closest_model('center6_18_v.dat', limit=17, VVV=True)
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


df = pandas.DataFrame([k_avg61, k_avg71, k_avg81, k_avg91, k_avg101,
		k_avg111, k_avg121, k_avg131, k_avg141, k_avg151,
		k_avg161, k_avg171]
                 )
#df.to_csv('ks_mags.csv')
#import pdb; pdb.set_trace()
#def many_small_lines(xvals, x_data, y_data):
#	f_x = []
#	pt = 1
#	pdb.set_trace()
#	for x_val in xvals:
#		
#		if (pt < len(x_data)) and (x_val <= x_data[pt]):
#			init_min = pt - 1 
#			init_max = pt
#			xval_1 = x_data[init_min]
#			xval_2 = x_data[init_max]
#	

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
