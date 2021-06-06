from trilegal_webscrapping import Trilegal
from MIRIS_flux_calculator import sed_flux_function

trilegal_2mass = Trilegal()
trilegal_vista = Trilegal(phot_system='vista')

def background_flux(glon, glat, field, VVV=False, wavelength='paa'):
    if VVV:
        data_stars = trilegal_vista.search(gc_l=glon, gc_b=glat, field=field)
    else:
        data_stars = trilegal_2mass.search(gc_l=glon, gc_b=glat, field=field)
    
    kmags = []
    for datapt in data_stars['Ks']:
        if 6 <= datapt <= 18:
            kmags.append(datapt)
    
    flux_bgd = sed_flux_function(kmags, wavelength)

    return flux_bgd
