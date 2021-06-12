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
    for datapt in data_stars['Ks']:
        if 6 <= datapt <= 18:
            kmags.append(datapt)
    
    flux_raw = sed_flux_function(kmags, wavelength)
    flux_bgd = np.sum(flux_raw)

    return flux_bgd

# glons = []
# glats = []
# flux = []
# for glon, glat in [(2.5, 0.1), (2.5, 1), (2.5, 2), (2.5, 3), (2.5, -1), 
#                 (-2.5, 0.1), (-2.5, 1), (-2.5, 2), (-2.5, 3), (-2.5, -1),
#                 (-1.5, 0.1), (-1.5, 1), (-1.5, 2), (-1.5, 3), (-1.5, -1),
#                 (1.5, 0.1), (1.5, 1), (1.5, 2), (1.5, 3), (1.5, -1),
#                 (0, 0.1), (0, 1), (0, 2), (0, 3), (0, -1),
#                 (5.0, 0.1), (5.0, 1), (5.0, 2), (5.0, 3), (5.0, -1),
#                 (-5.0, 0.1), (-5.0, 1), (-5.0, 2), (-5.0, 3), (-5.0, -1),
#                 (-10.0, 0.1), (-10.0, 1), (-10.0, 2), (-10.0, 3), (-10.0, -1),
#                 (-20.0, 0.1), (-20.0, 1), (-20.0, 2), (-20.0, 3), (-20.0, -1),
#                 (-30.0, 0.1), (-30.0, 1), (-30.0, 2), (-30.0, 3), (-30.0, -1),
#                 (-40.0, 0.1), (-40.0, 1), (-40.0, 2), (-40.0, 3), (-40.0, -1),
#                 (-50.0, 0.1), (-50.0, 1), (-50.0, 2), (-50.0, 3), (-50.0, -1),
#                 (-60.0, 0.1), (-60.0, 1), (-60.0, 2), (-60.0, 3), (-60.0, -1),
#                 (270.0, 0.1), (270.0, 1), (270.0, 2), (270.0, 3), (270.0, -1),
#                 (240.0, 0.1), (240.0, 1), (240.0, 2), (240.0, 3), (240.0, -1),
#                 (210.0, 0.1), (210.0, 1), (210.0, 2), (210.0, 3), (210.0, -1),
#                 (180.0, 0.1), (180.0, 1), (180.0, 2), (180.0, 3), (180.0, -1),
#                ]:
#                glons.append(glon)
#                glats.append(glat)
#                flux.append(background_flux(glon, glat))
               
# res = {}
# res['longitude']= glons
# res['latitude'] = glats
# res['flux per pix'] = flux
# table_m = Table(res)

# fig, ax = plt.subplots(figsize=(20,10))
# sc = ax.scatter(table_m['longitude'], table_m['latitude'], s=1000, c=table_m['flux per pix'])
# ax.set_ylabel('Galactic Latitude [Deg]', fontsize='20')
# ax.set_xlabel('Galactic Longitude [Deg]', fontsize='20')
# plt.title('Background Flux in the Galactic Plane, m > 16', fontsize='30')
# cbar = fig.colorbar(sc)
# cbar.set_label("Flux per pixel [Jy]", fontsize='20')
# plt.colorbar()