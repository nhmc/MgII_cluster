"""
This script will generate the model to fit to dN/dz for MgII as a
function of impact parameter.

First aim is to copy the model from Zhu et al. 2014


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
from __future__ import division, print_function, unicode_literals

from math import pi, log, hypot, log10, sin, cos, sqrt
from astropy.cosmology import FlatLambdaCDM, Planck13, WMAP7
import astropy.units as u
import numpy as np

import sympy.mpmath as mp

import time

from barak.plot import make_log_xlabels, make_log_ylabels
from barak.virial import deltavir

from scipy.integrate import quad

from cosmolopy.perturbation import fgrowth, w_tophat, power_spectrum
from cosmolopy.parameters import WMAP7_BAO_H0_mean


FIGSIZE = 5,5


def concentration(m_on_mstar, redshift, c0=9., beta=0.13):
    """ Find the concentration parameter for a dark matter halo.

    At first glance this seems insensitive to m/m^* (which is to the
    exponent of -beta ~ -0.13). But the mass range is large (factors
    of 10-100), so the term can still be important relative to the
    redshift term (redshift changes by factor of a few).
    """
    return c0 / (1. + redshift) * m_on_mstar**-beta

def rho_nfw(r_on_rs, gamma=1):
    """ Find the NFW density profile at a given radius. You must
    multiply the return value by the scale density, rho_s, to get the
    actual profile!
    
    r_on_rs is r / rs, where rs = r_vir / concentration

    r is comoving.

    """
    return 1 / (r_on_rs**gamma * (1 + r_on_rs)**(3-gamma))


def M_params(mass, rho_bar_m, delta_vir, Mstar, redshift):
    """ Calculate all parameters which depend on the halo mass.

    Returns a dictionary with all the derived parameters
    """

    # rearrange M = 4*pi/3. * rho_bar_m * delta_vir * r_vir**3
    r_vir = (mass / (4*pi/3. * rho_bar_m * delta_vir))**(1/3.)

    # find the scale density rho_s


    # rearrange M = 4*pi * rho_s * r_vir**3 / conc**3  * (log(1+conc) - conc / (1 + conc))

    M_on_mstar = (mass / Mstar).to(u.dimensionless_unscaled).value
    C = concentration(M_on_mstar, redshift)
    rs = r_vir / C

    rho_s = C**3 * mass / (4*pi * r_vir**3 * (log(1+C) - C/(1 + C))) 

    # tophat_radius (used for 2-halo term)

    return dict(rvir=r_vir, M_on_mstar=M_on_mstar, C=C, rs=rs, rho_s=rho_s)


# one halo term
def Sigma_1h(rp_vals, M):
    """
    Integrate the NFW profile in the line-of-sight direction give a
    surface density profile over a range of impact parameters.

    Parameters
    ----------
    rp_vals : array_like
      list of comoving impact parameters

    M : dict
      Dictionary of parameters that depend on the halo mass. Must
      contain keys 'rs' (with units) and 'rho_s' (also with units),
      both comoving. See M_params().

    Returns
    -------
    sigma_p : shape (N,) 
      Surface density profile

    """
    RS = M['rs'].to(u.kpc).value

    def integrand(s, rp):
        """rp and s must be in units of physical kpc """
        R = hypot(s, rp) / RS
        return rho_nfw(R)

    # rp must be in units of physical kpc
    const = M['rho_s'].to(u.g/u.kpc**3).value
    sigma_p = []
    for rp in rp_vals:
        sigma_p.append(quad(integrand, -np.inf, np.inf, args=(rp,))[0])

    sigma_p = np.array(sigma_p) * const * u.g / u.kpc**2
    return sigma_p


def calc_bias(delta_c, nu_M):
    """ Find the bias from the overdensity threshold for collapse

    A17 from Zhu et al.
    """ 
    a = 1 / sqrt(2)
    b = 0.35
    c = 0.8
    an2 = a*nu_M**2
    
    return 1 + 1 / (sqrt(a) * delta_c) * \
           (sqrt(a)*an2 + sqrt(a)*b*an2**(1-c) - \
            an2**c / (an2**c + b*(1-c)*(1 - c/2.)))


def calc_fnu(nu_M, a=0.707, p=0.3, A=1):
    """ Find f(nu) using equation A19 from Zhu et al.

    A19 is used to get the mass function, f(nu). Find A using
    int_0^inf fnu dnu = 1. They find 0.129.  We find A=3.1??

    # need to integrate fnu from 0 -> inf and set equal to 1, which sets A.
    res2,_ = quad(fnu, 0, np.inf)
    print res2
    """
    # normalization constant is A, assume 1 for now.
    
    an2 = a*nu_M**2
    return A*np.sqrt(2 * an2 / pi) * (1 + an2**-p) * np.exp(-0.5*an2) / nu_M


def Z_params(z, cosmo):
    """ Quantities which depend on the redshift and cosmology."""
    Z = {}
    Z['rho_bar_m'] = cosmo.Om(z) * cosmo.critical_density(z)
    Z['delta_vir'] = deltavir(z, cosmo=cosmo)

    # A14, overdensity threshold for spherical collapse
    Z['delta_c'] = 3 / 20. * (12 * pi)**(2/3.) * \
                   (1 + 0.013 * log10(cosmo.Om(REDSHIFT)))
    Z['growth_factor'] = fgrowth(REDSHIFT, cosmo.Om0)
    return Z


def calc_sigma(MASS, Z, REDSHIFT, COSMO1, debug=False):
    # A15, finds sigma 
    """
    W(x) = 3*(sin(x) - x*cos(x)) / x**3 
    
    sigma**2 = int_0^inf k**3 * Plin(k) / (k * 2 * pi**2) * W  * dk

    where Plin is the matter power spectrum from Eisenstein and Hu.

    """
    # Need to loop over mass and k here
    
    tophat_radius_Mpc = ((3 * MASS / (4 * pi * Z['rho_bar_m']))**(1/3.)).to(u.Mpc).value
    def sigma_integrand(k):
        # k must be in inverse Mpc

        # should we use the REDSHIFT here? It's currently set to zero.
        
        return k**2 * power_spectrum(float(k), 0, **COSMO1) / (2*pi**2) \
               * w_tophat(float(k), float(tophat_radius_Mpc))**2

    def sigma_integrand_ufunc(k):
        # Same function as sigma_integrand, but works on arrays of k, not just scalars.
        # Useful for plotting purposes.
        # k must be in inverse Mpc
        return k**2 * power_spectrum(k, REDSHIFT, **COSMO1) / (2*pi**2) \
               * w_tophat(k, tophat_radius_Mpc)**2

    if 0:
        # plot the sigma_integrand function using the _ufunc version.  It's a bastard to
        # numerically integrate.
        fig = plt.figure()
        fig.clf()
        ax = plt.gca()
        kvals = np.logspace(-2, 2, 1e5)
        ax.loglog(kvals, sigma_integrand_ufunc(kvals))
        plt.show()

    if debug:
        
        print('Integrating over all k 0 -> infinity to find sigma_M')

        #res  = quad(sigma_integrand, 0, np.inf)
        t1 = time.time()
        res = mp.quad(sigma_integrand, (0, mp.inf))
        t2 = time.time()
        print('Done using mp.quad in', t2 - t1, 's')
        # this is ~3 times slower than mp.quad.
        #res2,_ = quad(sigma_integrand, 0, np.inf)
        #print 'quad', time.time() - t2, 's'
        #this takes ages, but gives pretty much the same answer (to within 1e-6) as mp.quad
        #res2 = mp.quadosc(sigma_integrand, (0, mp.inf), omega=tophat_radius)
        #print 'quadosc done', (t2 - time.time())/60., 's'
        #print res, res2
    sigma_M = sqrt(res)

    return sigma_M


def Phm_integrand(kval, mval):
    """ Find the bit which goes inside the integral in A12

    uses global variables Z, REDSHIFT, COSMO1

    integrate using this function to evaluate A12 in Zhu et al.
    """
    mval = mval * u.M_sun

    # now find sigma (this is pretty slow)
    sigma_M = calc_sigma(mval, Z, REDSHIFT, COSMO1)

    # A13
    nu_M = Z['delta_c'] / Z['growth_factor'] / sigma_M


    # find new parameters for this halo mass
    mpar = M_params(mval, Z['rho_bar_m'], Z['delta_vir'], MSTAR, REDSHIFT)
    
    RS_M = mpar['rs'].to(u.Mpc).value
    # find fnu

    def u_knu_integrand(r):
        r = float(r)
        x = kval * r
        return rho_nfw(r/RS_M) * 4 * pi * r**2 * sin(x)/x

    if 0:
        # plot the integrand
        plt.figure()
        rvals = 10**np.linspace(-10, 10, 50000)
        uknu = [u_knu_integrand(r) for r in rvals]
        plt.plot(np.log10(rvals), np.log10(uknu))
        plt.show()
    #mp.quadosc(u_knu_integrand, (0, mp.inf), omega=K)

    # this is 3 times faster, but less accurate.
    #u_nu_temp = quad(u_knu_integrand, 0, np.inf)
    u_nu = float(mp.quad(u_knu_integrand, (0, mp.inf)))

    # bias, A17
    bias = calc_bias(Z['delta_c'], nu_M)
    
    # fnu, A19
    fnu = calc_fnu(nu_M)
    
    integrand = u_nu * bias * fnu
    
    print('    u_nu {:g} bias {:g} fnu {:g}'.format(u_nu, bias, fnu))
    return integrand


if 1:

    # note the default is no baryonic effects in the power spectrum!
    #
    # These are the values from Zhu, with extras from
    # WMAP7_BAO_H0_mean(flat=1)

    COSMO1 = {'N_nu': 0,
              'Y_He': 0.24,
              'baryonic_effects': False,
              'h': 0.7,
              'n': 0.96,
              'omega_M_0': 0.3,
              'omega_b_0': 0.0456,
              'omega_k_0': 0.0,
              'omega_lambda_0': 0.7,
              'omega_n_0': 0.0,
              'sigma_8': 0.8,
              't_0': 13.75,
              'tau': 0.087,
              'w': -1.0,
              'z_reion': 10.4}

    cosmo = FlatLambdaCDM(H0=COSMO1['h']*100,
                          Om0=COSMO1['omega_M_0'])


    ############################################
    # These are fitting parameters we can vary
    ############################################
    
    # the halo mass
    MASS = 10**14.5 * u.M_sun
    
    # redshift
    REDSHIFT = float(0.6)
    
    #############################################
    #  'Fixed' parameters
    #############################################
    
    # the fiducial mass for the galaxy luminosity function.
    MSTAR = 10**12.7 * u.M_sun
    
    print('Inputs:')
    print('  Mass {:g}'.format(MASS))
    print('  M* {:g}'.format(MSTAR))
    print('  Redshift {}'.format(REDSHIFT))
    print('  Cosmology {}'.format(repr(cosmo)))
    print('')
    
if 1:
    ################################################################
    # Setup and general quantities needed for 1- and 2-halo terms
    ################################################################
    Z = Z_params(REDSHIFT, cosmo)
    M = M_params(MASS, Z['rho_bar_m'], Z['delta_vir'], MSTAR, REDSHIFT)

    print('Virial radius {:g}'.format(M['rvir'].to(u.kpc)))
    print('mass/M* {:g}'.format(M['M_on_mstar']))
    print('concentration {:g}'.format(M['C']))
    print('Scale density {:g}'.format(M['rho_s'].to(u.g / u.cm**3)))

if 0:

    ########################################
    # One halo term
    ########################################
    
    # test plot of the density profile

    # note comoving
    rvals = np.logspace(-1, 4) * u.kpc
    R_M = (rvals / M['rs']).to(u.dimensionless_unscaled).value    
    rho_m_M = M['rho_s'] * rho_nfw(R_M)

    fig = plt.figure(1, figsize=FIGSIZE)
    fig.clf()
    ax = fig.add_subplot(111)
    ax.plot(np.log10(rvals.to(u.kpc).value),
            np.log10(rho_m_M.to(u.M_sun/u.kpc**3).value))
    ax.set_xlabel(r'$r$ (physical kpc)')
    ax.set_ylabel(r'$\log_{10}\,\rho(r)\ (M_\odot/\mathrm{kpc}^3)$')
    ax.set_xlim(-0.9,3.9)
    #make_log_ylabels(ax)
    make_log_xlabels(ax)
    fig.savefig('check_rho.png', bbox_inches='tight')
    plt.show()

    
if 0:
    # test plot of the surface density for a NFW profile.
    rp_vals = np.logspace(-1, 4)
    xcorr1h = Sigma_1h(rp_vals, M).to(u.M_sun/u.kpc**2)
    if 1:
        # test plot
        fig = plt.figure(2, figsize=FIGSIZE)
        fig.clf()
        ax = fig.add_subplot(111)
        ax.plot(np.log10(rp_vals), np.log10(xcorr1h.value))
        ax.set_xlabel(r'Impact parameter $r_p$(physical kpc)')
        ax.set_ylabel(r'$\log_{10}\,\Sigma(r_\mathrm{p})\  (M_\odot/\mathrm{kpc}^2)$')
        ax.set_xlim(-0.9,3.9)
        #make_log_ylabels(ax)
        make_log_xlabels(ax)
        fig.savefig('check_sigma.png', bbox_inches='tight')
        plt.show()

if 1:
    ###################################
    # two halo term
    ###################################
    """
    sigma_2h = rho_bar_m * int -inf to +inf   xi_hm (sqt(rp**2 + s**2)) ds
    """
    
    # Calculate this on a big array of k. Just one halo mass for now.

    # M goes from M_min to M_max ()

    # range of halo masses (need to integrate the whole of the below over these)
    num_M = 20
    num_k = 10

    logMlo = 3
    logMhi = 17
    logMvals = np.linspace(3, 17.01, num_M)
    # distances from 100 kpc to 100 Mpc
    logkvals = np.linspace(-2, 1, num_k) 

    t1 = time.time()
    out = np.zeros((num_k, num_M))
    for i in range(num_k):
        kval = 10**logkvals[i]
        print('{} of {}, k={:g}'.format(i+1, num_k, kval))
        for j in range(num_M):
            mval = 10**logMvals[j]
            print('  {} of {}, M={:g}'.format(j+1, num_M, mval))
            res = Phm_integrand(kval, mval)
            print(res)
            out[i,j] = res

        # save as we go
        with open('temp.npz', 'w') as fh:
            print('Updating temp.npz')
            np.savez(fh, out=out, logk=logkvals, logM=logMvals)


    t2 = time.time()
    print 'Total time elapsed {} min'.format((t2 - t1) / 60)
