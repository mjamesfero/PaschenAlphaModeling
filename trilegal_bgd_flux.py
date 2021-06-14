from trilegal_webscrapping import Trilegal
from MIRIS_flux_calculator import sed_flux_function
import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
plt.style.use('dark_background')

trilegal_2mass = Trilegal()
trilegal_vista = Trilegal(phot_system='vista')

def background_flux(glon, glat, field=0.005, VVV=False, wavelength='paa'):
    if VVV:
        data_stars = trilegal_vista.search(gc_l=glon, gc_b=glat, field=field)
    else:
        data_stars = trilegal_2mass.search(gc_l=glon, gc_b=glat, field=field)
    
    kmags = []
    try:
        for datapt in data_stars['Ks']:
            if 6 <= datapt <= 18:
                kmags.append(datapt)
        
        flux_raw = sed_flux_function(kmags, wavelength)
        flux_bgd = np.sum(flux_raw)
    except:
        flux_bgd = 0

    return flux_bgd