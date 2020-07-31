import numpy as np
from astropy import units as u
from pyspeckit.spectrum.models import hydrogen
from astropy import convolution
from astropy import modeling

# SPAM calcs
# sb_sensitivity is the anticipated surface brightness sensitivity
sb_sensitivity = 1.5e-16 *u.erg/u.s/u.cm**2/u.arcsec**2
aperture_diameter = 24*u.cm

collecting_area = (aperture_diameter/2)**2 * np.pi

paa_bandwidth = 5*u.nm
paac_bandwidth = 10*u.nm

wl_paa = hydrogen.wavelength['paschena']*u.um
e_paa = wl_paa.to(u.erg, u.spectral())
nu_paa = wl_paa.to(u.Hz, u.spectral())

wl_halpha = hydrogen.wavelength['balmera']*u.um
nu_halpha = wl_halpha.to(u.Hz, u.spectral())

sb_sensitivity_MJySr = (sb_sensitivity / nu_paa).to(u.MJy/u.sr)

fwhm = (1.22*wl_paa / aperture_diameter).to(u.arcsec, u.dimensionless_angles())
psf_area = 2*np.pi*(fwhm**2) / (8*np.log(2))
ps_sensitivity = sb_sensitivity * psf_area

pixscale = (0.806*u.arcsec)
fov = pixscale * 2048

readnoise_pessimistic = 22*u.ph
readnoise_optimistic = 6.2*u.ph
dark_rate_optimistic = 0.123*u.ph/u.s
dark_rate_pessimistic = 0.435*u.ph/u.s

fiducial_integration_time = 500*u.s

throughput = 0.7

saturation_limit = 1.8e5

yy,xx = np.mgrid[-25:25:1,-25:25:1]
airymod = modeling.models.AiryDisk2D(amplitude=1, x_0=0, y_0=0, radius=fwhm/pixscale)(yy,xx)

max_unsaturated_rate = saturation_limit / fiducial_integration_time * (airymod.max() / airymod.sum()) * throughput


rn_pess = readnoise_pessimistic/fiducial_integration_time
# Poisson noise
darkn_pess = ((dark_rate_pessimistic * fiducial_integration_time)**0.5).value * u.ph / fiducial_integration_time
dark_rn_pess = (((rn_pess + darkn_pess) * (e_paa/u.ph) /
                 collecting_area / nu_paa).to(u.mJy) /
                pixscale**2).to(u.MJy/u.sr) / throughput

rn_opt = readnoise_optimistic/fiducial_integration_time
darkn_opt = ((dark_rate_optimistic * fiducial_integration_time * u.ph)**0.5).value * u.ph / fiducial_integration_time
dark_rn_opt = (((rn_opt + darkn_opt) * (e_paa/u.ph) /
                collecting_area / nu_paa).to(u.mJy) /
               pixscale**2).to(u.MJy/u.sr) / throughput

miris_sb_sens = 0.77*u.mJy/(52*u.arcsec)**2
miris_fwhm = 52*u.arcsec

meergal_fwhm = 0.8*u.arcsec
meergal_sb_sens = 40*u.uJy/(meergal_fwhm**2 / (8*np.log(2)) * np.pi)

thor_fwhm = 10*u.arcsec
thor_sb_sens = 500*u.uJy/(thor_fwhm**2 / (8*np.log(2)) * np.pi)

vphas_limiting_mag = 20

supercosmos_sb_sens = (5*u.Rayleigh / nu_halpha * (nu_halpha.to(u.erg, u.spectral()) / u.ph)).to(u.MJy/u.sr)
supercosmos_fwhm = 3*u.arcsec

if __name__ == "__main__":
    print(f"Optimistic sensitivity:\ndark_rn_opt={dark_rn_opt}\nsb_sensitivity_MJySr={sb_sensitivity_MJySr}")
