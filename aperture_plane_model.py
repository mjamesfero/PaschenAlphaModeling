import numpy as np
import poppy
from astropy import units as u
import pylab as pl

primary_mirror_diameter = 250.1081*u.mm
secondary_mirror_diameter = 40.68*u.mm
pixscale = 0.806*u.arcsec
wavelength=1.8756*u.um

def poppy_model():


    # piston, tilt x j=2, tilt y j=3, focus j=4, astigmatism 45 j=5, astigmatism 0 j=6
    # values are RMS in nm
    wfe_budget = [0, 100, 100, 50, 36, 36]

    def generate_coefficients(wfe_budget):
        coefficients = []
        for term in wfe_budget:
            coefficients.append(
                # convert nm to meters, get value in range
                np.random.uniform(low=-1e-9 * term, high=1e-9 * term)
            )
        return coefficients

    possible_coefficients = [generate_coefficients(wfe_budget) for i in range(5)]

    results = []

    fig =pl.figure(1, figsize=(18, 5))
    fig.clf()

    for ii,coefficient_set in enumerate(possible_coefficients):
        print(coefficient_set)

        osys = poppy.OpticalSystem()
        ap = poppy.CircularAperture(radius=primary_mirror_diameter.to(u.m).value)    # pupil radius in meters
        sec = poppy.SecondaryObscuration(secondary_radius=secondary_mirror_diameter.to(u.m).value, n_supports=0)
        optic = poppy.CompoundAnalyticOptic(opticslist=[ap, sec], name='PASHION')

        zwfe = poppy.ZernikeWFE(
            coefficients=coefficient_set,
            radius=primary_mirror_diameter.to(u.m).value
        )
        osys.add_pupil(optic)
        osys.add_pupil(zwfe)
        osys.add_detector(pixelscale=pixscale.to(u.arcsec).value, fov_arcsec=60.0)  # image plane coordinates in arcseconds

        psf, interm = osys.calc_psf(wavelength.to(u.m).value, return_intermediates=True)                            # wavelength in meters
        results.append(psf)

        ax = pl.subplot(2, 5, 1+ii)
        poppy.display_psf(psf, ax=ax, title='PASHION psf')
        ax = pl.subplot(2, 5, 6+ii)
        interm[1].display(what='phase', ax=ax)

    pl.ion()
    pl.show()

    return psf, osys




def psf(xx, yy, xcen=0*u.mm, ycen=0*u.mm, smear_beam_fwhm=11*u.arcsec,
        smear_beam_amplitude=0.05, pixscale=0.806*u.arcsec, pixsize=18*u.um,
        wavelength=1.8756*u.um
       ):
    """
    """
    # rr2 = radius^2 in aperture plane
    rr2 = ((xx-xcen)**2 + (yy-ycen)**2)

    aperture = (rr2 > (secondary_mirror_diameter/2)**2) & (rr2 < (primary_mirror_diameter/2)**2)

    #focal_length = pixsize / (pixscale.to(u.radian).value)

    # generic eqn
    # ft(e^-ax^2) = sqrt(pi/a) e^-pi^2 k^2 / a
    # smear_beam = amp * e^-(x0^2 / 2 sigma^2)
    sb_sigma = smear_beam_fwhm / (8*np.log(2))**0.5
    sb_sigma_ft = sb_sigma.to(u.rad) / wavelength
    # assume the amplitude is the same in the focal plane (this is not a justified assumption)
    #smear_beama = smear_beam_amplitude * np.exp(-kk2 / (2 * sb_sigma**2))
    smear_beam_ft = smear_beam_amplitude * np.exp((-rr2 / 2 * sb_sigma_ft**2).to(u.dimensionless_unscaled, u.dimensionless_angles()))

    pure_psf = np.abs(np.fft.fftshift(np.fft.fft2(aperture)**2))
    smear_psf = np.abs(np.fft.fftshift(np.fft.fft2(aperture * smear_beam_ft)**2))

    size = xx.shape[0]
    fig = pl.figure(1)
    fig.clf()
    ax1 = pl.subplot(2, 2, 1)
    ax1.imshow(aperture)
    ax2 = pl.subplot(2, 2, 2)
    ax2.imshow(aperture*smear_beam_ft)
    ax3 = pl.subplot(2, 2, 3)
    ax3.imshow(pure_psf, norm=visualization.simple_norm(pure_psf, stretch='log'))
    ax3.axis((size/2-20,size/2+20,size/2-20,size/2+20))
    ax4 = pl.subplot(2, 2, 4)
    ax4.imshow(smear_psf, norm=visualization.simple_norm(smear_psf, stretch='log'))
    ax4.axis((size/2-20,size/2+20,size/2-20,size/2+20))


    fig2 = pl.figure(2)
    fig2.clf()
    fig2.gca().plot(pure_psf[size//2,:] / pure_psf.max(), label='pure')
    fig2.gca().plot(smear_psf[size//2,:] / smear_psf.max(), label='smear')
    pl.legend(loc='best')

if __name__ == "__main__":
    pop_psf, osys = poppy_model()
