import os
import glob
import numpy as np
from astropy import wcs
from astropy import table
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astroquery.vizier import Vizier
from astropy.coordinates import SkyCoord
from trilegal_bgd_flux import background_flux
from MIRIS_flux_calculator import sed_flux_function

# name1 = glob.glob('gc_*.fits')
# name2 = [fits.getdata(x) for x in name1]

# headers = []
# for ii in range(3):
#     headers.append(fits.getheader(name1[ii]))
# ww = [wcs.WCS(headers[0]), wcs.WCS(headers[1]), wcs.WCS(headers[2])]

# indices = [[],[],[]]
# glon_glat_1 = []
# glon_glat_2 = []
# glon_glat_3 = []
# for ii in range(9142):
#     for jj in range(6202):
#         if not np.isnan(name2[0][ii][jj]):
#             indices[0].append([ii,jj])
#             ra1, dec1 = ww[0].wcs_pix2world(ii,jj,0)
#             glon_glat_1.append([ra1, dec1])
#         if not np.isnan(name2[1][ii][jj]):
#             indices[1].append([ii,jj])
#             ra1, dec1 = ww[1].wcs_pix2world(ii,jj,0)
#             glon_glat_2.append([ra1, dec1])
#         if not np.isnan(name2[2][ii][jj]):
#             indices[2].append([ii,jj])
#             ra1, dec1 = ww[2].wcs_pix2world(ii,jj,0)
#             glon_glat_3.append([ra1, dec1])

width = 47.5*u.arcsec
height = 47.5*u.arcsec
field = (47.5/3600)**2

linename = 'paa'
#unknown
kmag_threshold = 8.5
wavelength = 1
diameter = 1
imsize = 2048
pixscale = 1
pa_energy = 1
bandwidth = 1
pa_freq = 1
exptime = 1 
transmission_fraction = 1 

def make_source_table(glon, glat):
    Viz = Vizier(row_limit=int(1e6))
    cats = Viz.query_region(SkyCoord(glon, glat, frame='galactic'),
                            width=width, height=height, 
                            catalog=["II/348", "II/246"])

    cat2 = cats['II/348/vvv2']
    cat2mass = cats['II/246/out']

    cat2c = SkyCoord(cat2['RAJ2000'], cat2['DEJ2000'], frame='fk5',
                     unit=(u.deg, u.deg)).galactic
    coords2mass = SkyCoord(cat2mass['RAJ2000'], cat2mass['DEJ2000'],
                           frame='fk5', unit=(u.deg, u.deg)).galactic


    vvv_faint = cat2['Ksmag3'] > kmag_threshold
    twomass_bright = cat2mass['Kmag'] < kmag_threshold


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


    # Assemble source table from VVV sources
    pix_coords_vvv = target_image_wcs.wcs_world2pix(cat2c.l.deg, cat2c.b.deg, 0)

    fluxes = sed_flux_function(kmags=cat2['Ksmag3'], wavelength=linename)
    bad_vvv = (cat2['Ksmag3'].mask | cat2['Hmag3'].mask | (pix_coords_vvv[0] < 0) | (pix_coords_vvv[0] > imsize) |
               (pix_coords_vvv[1] < 0) | (pix_coords_vvv[1] > imsize) | (~vvv_faint))

    phot_fluxes = fluxes / pa_energy * u.photon

    bandwidth_Hz = ((bandwidth / wavelength) * pa_freq).to(u.Hz)

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    bandwidth_Hz).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.ph).value

    cat2.add_column(col=phot_ct, name=f'{linename}_phot_ct')
    cat2.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
    cat2.add_column(col=fluxes, name=f'{linename}_flux')

    nsrc = len(phot_ct_rate[~bad_vvv])

    x = pix_coords_vvv[0][~bad_vvv]
    y = pix_coords_vvv[1][~bad_vvv]

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table = Table({'amplitude': phot_ct[~bad_vvv] * transmission_fraction,
                          'x_mean': np.round(x),
                          'y_mean': np.round(y),
                          'x_0': x,
                          'y_0': y,
                          'radius': np.repeat(airy_radius/pixscale, nsrc),
                          'x_stddev': abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
                          'y_stddev': abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
                          'theta': np.pi * (x-1024),
                         })


    # Assemble source table from 2MASS sources
    pix_coords_2mass = target_image_wcs.wcs_world2pix(coords2mass.l.deg,
                                                      coords2mass.b.deg, 0)

    fluxes = sed_flux_function(kmags=cat2mass['Kmag'], wavelength=linename)
    bad_2mass = (cat2mass['Hmag'].mask | cat2mass['Kmag'].mask | (pix_coords_2mass[0] < 0) |
                 (pix_coords_2mass[0] > imsize) | (pix_coords_2mass[1] < 0) |
                 (pix_coords_2mass[1] > imsize) | (~twomass_bright))

    phot_fluxes = fluxes / pa_energy * u.photon

    phot_ct_rate = (phot_fluxes * collecting_area * pixel_fraction_of_area *
                    bandwidth_Hz).decompose()
    phot_ct = (phot_ct_rate * exptime).to(u.photon).value


    cat2mass.add_column(col=phot_ct, name=f'{linename}_phot_ct')
    cat2mass.add_column(col=phot_ct_rate, name=f'{linename}_phot_ct_rate')
    cat2mass.add_column(col=fluxes, name=f'{linename}_flux')

    nsrc = len(phot_ct_rate[~bad_2mass])

    x = pix_coords_2mass[0][~bad_2mass]
    y = pix_coords_2mass[1][~bad_2mass]

    #Must have columns: amplitude x_mean y_mean x_stddev y_stddev theta
    source_table_2mass = Table({'amplitude': phot_ct[~bad_2mass] * transmission_fraction,
                                'x_mean': np.round(x),
                                'y_mean': np.round(y),
                                'x_0': x,
                                'y_0': y,
                                'radius': np.repeat(airy_radius/pixscale, nsrc),
                                'x_stddev' : abs(1.2 * (x - 1024)/4096 * (y - 1024)/4096),
                                'y_stddev' : abs(0.8 * (-x + 1024)/4096 * (y- 1024)/4096),
                                'theta' : np.pi * (x-1024),
                               })


    source_table_both = table.vstack([source_table, source_table_2mass])

    return source_table_both, cat2, cat2mass, header

def trytoget_miris(glon, glat):
    fn = f"glon_{glon}_glat_{glat}_pa_filter_miris.fits"
    paaclfn = f"glon_{glon}_glat_{glat}_paacl_filter_miris.fits"
    paachfn = f"glon_{glon}_glat_{glat}_paach_filter_miris.fits"
    if os.path.exists(fn) and os.path.exists(paachfn):
        stars_background_im = fits.getdata(fn)
        stars_background_im_paach = fits.getdata(paachfn)
        stars_background_im_paacl = fits.getdata(paaclfn)

def miris_bgd_flux(glon_and_glat):
    """
    Find the expected background flux per each MIRIS pixel.
    Useless
    """
    glons = []
    glats = []
    fluxes = []
    for point in glon_and_glat:
        glon = point[0]
        glat = point[1]
        flux = background_flux(glon, glat, field=field)
        glons.append(glon)
        glats.append(glat)
        fluxes.append(flux)
    
    result = {}
    result['longitude']= glons
    result['latitude'] = glats
    result['flux per pix'] = fluxes

    return result