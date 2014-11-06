"""

Use bias model from Cole ?


Need a model to fit to dN/dz as a function of r_p

Use Zhu model as starting point.


f1h and f2h are free parameters. f1h is a function of M.


sigma_g(rp|M) = f1h * sigma1h(rp|M) + f2h * sigma2h(rp|M)


one halo term sigma1h
---------------------

rho_m(r|M) = rho_s / ( (r/r_s)**gamma * (1 + r/r_s)**(3-gamma) )

where r_s = r_vir / conc
and gamma = 1,

r_vir is determined from the mass via

M = 4*pi/3 * rho_bar_m * delta_vir * r_vir**3

where rho_bar_m is the mean matter density and delta_vir is the
critical overdensity for virialization, where we use the fitting
formula from Bryan and Norman.

conc = c0 / (1+z) * (M/Mstar)**-beta

c0 = 9, beta=0.13, Mstar = 10**12.7 * u.m_sun

then rho_s is found via

M = 4 * pi * rho_s * r_vir**3 / conc**3 * (log(1+conc) - conc / (1 + conc))

Finally to find the surface density, integrate rho_m along one direction.

"""

from math import pi, log, hypot, log10, sin, cos
from astropy.cosmology import FlatLambdaCDM, Planck13, WMAP7
from barak.virial import deltavir
import astropy.units as u
import numpy as np

FIGSIZE = 5,5

#cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
cosmo = WMAP7

mass = 10**13.0 * u.M_sun

redshift = 0.6

Mstar = 10**12.7 * u.M_sun

def concentration(m_on_mstar, redshift, c0=9., beta=0.13):
    """ The concentration parameter for a dark matter halo.
    """
    return c0 / (1 + redshift) * m_on_mstar**-beta

def rho_nfw(r_on_rs, gamma=1):
    """
    r_on_rs is r / rs, where rs = r_vir / concentration

    You must multiply this by the scale density, rho_s, to get the profile!
    """
    return 1 / (r_on_rs**gamma * (1 + r_on_rs)**(3-gamma))

print 'Inputs:'
print '  Mass {:g}'.format(mass)
print '  M* {:g}'.format(Mstar)
print '  Redshift {}'.format(redshift)
print '  Cosmology {}'.format(cosmo.name)
print ''

Om = cosmo.Om(redshift)

# matter density
rho_bar_m = Om * cosmo.critical_density(redshift)

delta_vir = deltavir(redshift, cosmo=cosmo)
# rearrange M = 4*pi/3. * rho_bar_m * delta_vir * r_vir**3

r_vir = (mass / (4*pi/3. * rho_bar_m * delta_vir))**(1/3.)
print 'Virial radius {:g}'.format(r_vir.to(u.kpc))

# find the scale density rho_s
    
# rearrange M = 4*pi * rho_s * r_vir**3 / conc**3  * (log(1+conc) - conc / (1 + conc))

m_on_mstar = (mass / Mstar).to(u.dimensionless_unscaled).value
C = concentration(m_on_mstar, redshift)
rs = r_vir / C
print 'mass/M* {:g}'.format(m_on_mstar)
print 'concentration {:g}'.format(C)

rho_s = C**3 * mass / (4*pi * r_vir**3 * (log(1+C) - C/(1 + C))) 

print 'Scale density {:g}'.format(rho_s.to(u.g / u.cm**3))

if 1:
    rvals = np.logspace(-1, 4) * u.kpc
    R = (rvals / rs).to(u.dimensionless_unscaled).value    
    rho_m = rho_s * rho_nfw(R)
    # test plot of the density profile
    from barak.plot import make_log_xlabels, make_log_ylabels
    fig = plt.figure(1, figsize=FIGSIZE)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.log10(rvals.to(u.kpc).value), np.log10(rho_m.to(u.M_sun/u.kpc**3).value))
    ax.set_xlabel(r'$r$ (physical kpc)')
    ax.set_ylabel(r'$\log_{10} \rho(r)\ (M_\odot/\mathrm{kpc}^3)$')
    ax.set_xlim(-0.9,3.9)
    #make_log_ylabels(ax)
    make_log_xlabels(ax)
    fig.savefig('check_rho.png', bbox_inches='tight')
    plt.show()

# one halo term

if 1:

    # Integrate the NFW profile in the line-of-sight direction give a
    # surface density profile for a series of r_p values.
    
    from scipy.integrate import quad

    RS = rs.to(u.kpc).value

    def integrand(s, rp):
        """rp and s must be in units of physical kpc """
        R = hypot(s, rp) / RS
        return rho_nfw(R)

    # rp must be in units of physical kpc
    rp_vals = np.logspace(-1, 4)
    const = rho_s.to(u.g/u.kpc**3).value
    sigma_p = []
    for rp in rp_vals:
        sigma_p.append(quad(integrand, -np.inf, np.inf, args=(rp,))[0])

    sigma_p = np.array(sigma_p) * const * u.g / u.kpc**2

    if 1:
        # test plot
        fig = plt.figure(2, figsize=FIGSIZE)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(np.log10(rp_vals), np.log10(sigma_p.to(u.M_sun/u.kpc**2).value))
        ax.set_xlabel(r'Impact parameter $r_p$(physical kpc)')
        ax.set_ylabel(r'$\log_{10} \Sigma(r_\mathrm{p})\  (M_\odot/\mathrm{kpc}^2)$')
        ax.set_xlim(-0.9,3.9)
        #make_log_ylabels(ax)
        make_log_xlabels(ax)
        fig.savefig('check_sigma.png', bbox_inches='tight')
        plt.show()

# two halo term
"""
sigma_2h = rho_bar_m * int -inf to +inf   xi_hm (sqt(rp**2 + s**2)) ds
"""

if 1:
    from cosmolopy.perturbation import fgrowth, w_tophat, power_spectrum
    from cosmology.parameters import WMAP7_BAO_H0_mean

    cosmo1 = WMAP7_BAO_H0_mean(flat=1)
    
    # overdensity threshold for spherical collapse
    delta_c = 3 / 20. * (12 * pi)**(2/3.) * \
              (1 + 0.013 * log10(cosmo.Om(redshift)))

    growth_factor = fgrowth(redshift, cosmo.Om0)

    """
    W(x) = 3*(sin(x) - x*cos(x)) / x**3 
    
    sigma**2 = int_0^inf k**3 * Plin(k) / (k * 2 * pi**2) * W  * dk
    """
    tophat_radius = (3 * mass / (4 * pi * rho_bar_m)**(1/3.)).to(u.Mpc).value
    def sigma_integrand(k):
        # k must be in inverse Mpc
        return 1/k * k**3 * power_spectrum(k, redshift) / (2*pi**2) \
               * w_tophat(k, tophat_radius)

    # requires sigma
    nu = delta_c / growth_factor / sigma

    # bias
    a = 1 / sqrt(2)
    b = 0.35
    c = 0.8
    an2 = a*nu**2

    # A17 requires nu
    bias = 1 + 1 / (sqrt(a) * delta_c) * \
           (sqrt(a)*an2 + sqrt(a)*b*an2**(1-c) - \
            an2**c / (an2**c + b*(1-c)*(1 - c/2.)))

    # A19 requires nu. Find A using int_0^inf fnu dnu = 1. They find 0.129.
    a = 0.707
    p = 0.3
    an2 = a*nu**2
    fnu = A * sqrt(2 * a * nu**2 / pi) * (1 + an2)**-p * exp(-0.5*an2) / nu

    u_kv = rho_nfw * rho_s

    
