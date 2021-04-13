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
from MIRIS_flux_calculator import flux_pa, flux_paacl, flux_paach

name1 = glob.glob('gc_*.fits')
name2 = [fits.getdata(x) for x in name1]

headers = []
for ii in range(3):
    headers.append(fits.getheader(name1[ii]))

indices = [[],[],[]]
for ii in range(9142):
    for jj in range(6202):
        if not np.isnan(name2[0][ii][jj]):
            indices[0].append([ii,jj])
        if not np.isnan(name2[1][ii][jj]):
            indices[1].append([ii,jj])
        if not np.isnan(name2[2][ii][jj]):
            indices[2].append([ii,jj])