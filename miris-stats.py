import numpy as np
from astropy import units as u
from pyspeckit.spectrum.models import hydrogen
from astropy import convolution
from astropy import modeling

# SPAM calcs
# sb_sensitivity is the anticipated surface brightness sensitivity
#sb_sensitivity = 1.5e-16 *u.erg/u.s/u.cm**2/u.arcsec**2
sb_sensitivity = 0.77*u.mJy/(52*u.arcsec)**2
miris_fwhm = 52*u.arcsec
aperture_diameter = 80*u.mm

collecting_area = (aperture_diameter/2)**2 * np.pi

#http://miris.kasi.re.kr/miris/pages/instrument/#filters
paa_bandwidth = 24*u.nm
paach_bandwidth = 25*u.nm
paacl_bandwidth = 25*u.nm
I_bandwidth = 0.910*u.micron
H_bandwidth = 0.550*u.micron

wl_paa = 1.876*u.micron
wl_paach = 1.84*u.micron
wl_paacl = 1.92*u.micron
e_paa = wl_paa.to(u.erg, u.spectral())
nu_paa = wl_paa.to(u.Hz, u.spectral())

#wl_paa = hydrogen.wavelength['paschena']*u.um

wl_halpha = hydrogen.wavelength['balmera']*u.um
nu_halpha = wl_halpha.to(u.Hz, u.spectral())

sb_sensitivity_MJySr = (sb_sensitivity / nu_paa).to(u.MJy/u.sr)

fwhm = (1.22*wl_paa / aperture_diameter).to(u.arcsec, u.dimensionless_angles())
psf_area = 2*np.pi*(fwhm**2) / (8*np.log(2))
ps_sensitivity = sb_sensitivity * psf_area

pixscale = (51.6*u.arcsec)
#fov = pixscale * 2048

#readnoise_pessimistic = 22*u.count
#readnoise_optimistic = 6.2*u.count
#dark_rate_optimistic = 0.123*u.count/u.s
#dark_rate_pessimistic = 0.435*u.count/u.s

#fiducial_integration_time = 500*u.s

#throughput = 0.7

#saturation_limit = 1.8e5

#yy,xx = np.mgrid[-25:25:1,-25:25:1]
#airymod = modeling.models.AiryDisk2D(amplitude=1, x_0=0, y_0=0, radius=fwhm/pixscale)(yy,xx)

#max_unsaturated_rate = saturation_limit / fiducial_integration_time * (airymod.max() / airymod.sum()) * throughput
paa_bandwidth_Hz = (paa_bandwidth / wl_paa) * nu_paa
#max_unsaturated_flux = (e_paa / collecting_area * max_unsaturated_rate / paa_bandwidth_Hz).to(u.mJy)
#mag_zeropoint_paa = 870*u.Jy # interpolated
#max_unsaturated_mag = -2.5 * np.log10(max_unsaturated_flux / mag_zeropoint_paa)


#rn_pess = readnoise_pessimistic/fiducial_integration_time
# Poisson noise
#darkn_pess = ((dark_rate_pessimistic * fiducial_integration_time)**0.5).value * u.count / fiducial_integration_time
#dark_rn_pess = (((rn_pess + darkn_pess) * (e_paa/u.count) /
 #                collecting_area / nu_paa).to(u.mJy) /
 #               pixscale**2).to(u.MJy/u.sr) / throughput

#rn_opt = readnoise_optimistic/fiducial_integration_time
#darkn_opt = ((dark_rate_optimistic * fiducial_integration_time * u.count)**0.5).value * u.count / fiducial_integration_time
#dark_rn_opt = (((rn_opt + darkn_opt) * (e_paa/u.count) /
#                collecting_area / nu_paa).to(u.mJy) /
 #              pixscale**2).to(u.MJy/u.sr) / throughput

#meergal_fwhm = 0.8*u.arcsec
#meergal_sb_sens = 40*u.uJy/(meergal_fwhm**2 / (8*np.log(2)) * np.pi)

#thor_fwhm = 10*u.arcsec
#thor_sb_sens = 500*u.uJy/(thor_fwhm**2 / (8*np.log(2)) * np.pi)

#vphas_limiting_mag = 20

#supercosmos_sb_sens = (5*u.Rayleigh / nu_halpha * (nu_halpha.to(u.erg, u.spectral()) / u.ph)).to(u.MJy/u.sr)
#supercosmos_fwhm = 3*u.arcsec
