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

#location: glon=0, glat=0
#area = 0.0075deg^2
res_6_to_7_m = closest_model('center6_7_m.dat')
result_6_to_7_m = []
res_6_to_7_v = closest_model('center6_7_v.dat', VVV=True)
result_6_to_7_v = []
res_7_to_8_m = closest_model('center6_8_m.dat', limit=7)
result_7_to_8_m = []
res_7_to_8_v = closest_model('center6_8_v.dat', limit=7, VVV=True)
result_7_to_8_v = []
for name in res_6_to_7_m['fn model']:
    result_6_to_7_m.append(name)
for name in res_6_to_7_v['fn model']:
    result_6_to_7_v.append(name)
for name in res_7_to_8_m['fn model']:
    result_7_to_8_m.append(name)
for name in res_7_to_8_v['fn model']:
    result_7_to_8_v.append(name)
#location: glon=0, glat=0
#area = 0.0005deg^2
res_8_to_9_m = closest_model('center6_9_m.dat', limit=8)
result_8_to_9_m = []
res_8_to_9_v = closest_model('center6_9_v.dat', limit=8, VVV=True)
result_8_to_9_v = []
res_9_to_10_m = closest_model('center6_10_m.dat', limit=9)
result_9_to_10_m = []
res_9_to_10_v = closest_model('center6_10_v.dat', limit=9, VVV=True)
result_9_to_10_v = []
res_10_to_11_m = closest_model('center6_11_m.dat', limit=10)
result_10_to_11_m = []
res_10_to_11_v = closest_model('center6_11_v.dat', limit=10, VVV=True)
result_10_to_11_v = []
res_11_to_12_m = closest_model('center6_12_m.dat', limit=11)
result_11_to_12_m = []
res_11_to_12_v = closest_model('center6_12_v.dat', limit=11, VVV=True)
result_11_to_12_v = []
res_12_to_13_m = closest_model('center6_13_m.dat', limit=12)
result_12_to_13_m = []
res_12_to_13_v = closest_model('center6_13_v.dat', limit=12, VVV=True)
result_12_to_13_v = []
for name in res_8_to_9_m['fn model']:
    result_8_to_9_m.append(name)
for name in res_8_to_9_v['fn model']:
    result_8_to_9_v.append(name)
for name in res_9_to_10_m['fn model']:
    result_9_to_10_m.append(name)
for name in res_9_to_10_v['fn model']:
    result_9_to_10_v.append(name)
for name in res_10_to_11_m['fn model']:
    result_10_to_11_m.append(name)
for name in res_10_to_11_v['fn model']:
    result_10_to_11_v.append(name)
for name in res_11_to_12_m['fn model']:
    result_11_to_12_m.append(name)
for name in res_11_to_12_v['fn model']:
    result_11_to_12_v.append(name)
for name in res_12_to_13_m['fn model']:
    result_12_to_13_m.append(name)
for name in res_12_to_13_v['fn model']:
    result_12_to_13_v.append(name)
#location: glon=0, glat=0
#area = 0.00005deg^2
res_13_to_14_m = closest_model('center6_14_m.dat', limit=13)
result_13_to_14_m = []
res_13_to_14_v = closest_model('center6_14_v.dat', limit=13, VVV=True)
result_13_to_14_v = []
res_14_to_15_m = closest_model('center6_15_m.dat', limit=14)
result_14_to_15_m = []
res_14_to_15_v = closest_model('center6_15_v.dat', limit=14, VVV=True)
result_14_to_15_v = []
res_15_to_16_m = closest_model('center6_16_m.dat', limit=15)
result_15_to_16_m = []
res_15_to_16_v = closest_model('center6_16_v.dat', limit=15, VVV=True)
result_15_to_16_v = []
res_16_to_17_m = closest_model('center6_17_m.dat', limit=16)
result_16_to_17_m = []
res_16_to_17_v = closest_model('center6_17_v.dat', limit=16, VVV=True)
result_16_to_17_v = []
for name in res_13_to_14_m['fn model']:
    result_13_to_14_m.append(name)
for name in res_13_to_14_v['fn model']:
    result_13_to_14_v.append(name)
for name in res_14_to_15_m['fn model']:
    result_14_to_15_m.append(name)
for name in res_14_to_15_v['fn model']:
    result_14_to_15_v.append(name)
for name in res_15_to_16_m['fn model']:
    result_15_to_16_m.append(name)
for name in res_15_to_16_v['fn model']:
    result_15_to_16_v.append(name)
for name in res_16_to_17_m['fn model']:
    result_16_to_17_m.append(name)
for name in res_16_to_17_v['fn model']:
    result_16_to_17_v.append(name)
#location: glon=0, glat=0
#area = 0.000005deg^2
res_17_to_18_m = closest_model('center6_18_m.dat', limit=17)
result_17_to_18_m = []
res_17_to_18_v = closest_model('center6_18_v.dat', limit=17, VVV=True)
result_17_to_18_v = []
for name in res_17_to_18_m['fn model']:
    result_17_to_18_m.append(name)
for name in res_17_to_18_v['fn model']:
    result_17_to_18_v.append(name)

data_dict_m = {}
data_dict_m['K6-7'] = result_6_to_7_m
data_dict_m['K7-8'] = result_7_to_8_m
data_dict_m['K8-9'] = result_8_to_9_m
data_dict_m['K9-10'] = result_9_to_10_m
data_dict_m['K10-11'] = result_10_to_11_m
data_dict_m['K11-12'] = result_11_to_12_m
data_dict_m['K12-13'] = result_12_to_13_m
data_dict_m['K13-14'] = result_13_to_14_m
data_dict_m['K14-15'] = result_14_to_15_m
data_dict_m['K15-16'] = result_15_to_16_m
data_dict_m['K16-17'] = result_16_to_17_m
data_dict_m['K17-18'] = result_17_to_18_m
data_dict_v = {}
data_dict_v['K6-7'] = result_6_to_7_v
data_dict_v['K7-8'] = result_7_to_8_v
data_dict_v['K8-9'] = result_8_to_9_v
data_dict_v['K9-10'] = result_9_to_10_v
data_dict_v['K10-11'] = result_10_to_11_v
data_dict_v['K11-12'] = result_11_to_12_v
data_dict_v['K12-13'] = result_12_to_13_v
data_dict_v['K13-14'] = result_13_to_14_v
data_dict_v['K14-15'] = result_14_to_15_v
data_dict_v['K15-16'] = result_15_to_16_v
data_dict_v['K16-17'] = result_16_to_17_v
data_dict_v['K17-18'] = result_17_to_18_v

w1 = csv.writer(open("model_names_m.csv", "w"))
for key, val in data_dict_m.items():
    w1.writerow([key, val])

w2 = csv.writer(open("model_names_v.csv", "w"))
for key, val in data_dict_v.items():
    w2.writerow([key, val])

print(result_17_to_18_v)