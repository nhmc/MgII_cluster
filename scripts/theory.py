"""
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

from math import pi, log, hypot
from astropy.cosmology import FlatLambdaCDM, Planck13
from barak.virial import deltavir
import astropy.units as u
import numpy as np

cosmo = Planck13

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

# one halo term
if 1:
    delta_vir = deltavir(redshift, cosmo=cosmo)
    # matter density
    rho_bar_m = cosmo.critical_density(redshift) * cosmo.Om(redshift)
    
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
        rvals = np.logspace(-1, 3) * u.kpc
        R = (rvals / rs).to(u.dimensionless_unscaled).value    
        rho_m = rho_s * rho_nfw(R)
        # test plot of the density profile
        from barak.plot import make_log_xlabels, make_log_ylabels
        fig = plt.figure(1)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(np.log10(rvals.to(u.kpc).value), np.log10(rho_m.to(u.M_sun/u.kpc**3).value))
        ax.set_xlabel('kpc (physical)')
        ax.set_ylabel('density (M_sun/kpc^3)')
        make_log_ylabels(ax)
        make_log_xlabels(ax)
        plt.show()

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
        fig = plt.figure(2)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(np.log10(rp_vals), np.log10(sigma_p.to(u.M_sun/u.kpc**2).value))
        ax.set_xlabel('impact parameter (physical kpc)')
        ax.set_ylabel(r'$\rho_\mathrm{p}$ (M_sun/kpc^2)')
        make_log_ylabels(ax)
        make_log_xlabels(ax)
        plt.show()

# two halo term 
