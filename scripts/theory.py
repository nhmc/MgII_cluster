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

import time, os

from barak.plot import make_log_xlabels, make_log_ylabels
from barak.virial import deltavir

from scipy.integrate import quad

from cosmolopy.perturbation import fgrowth, w_tophat, power_spectrum
from cosmolopy.parameters import WMAP7_BAO_H0_mean

from barak.io import loadobj, saveobj


SIGMA_CACHE = {}
if os.path.exists('SIGMA_CACHE.sav'):
    SIGMA_CACHE = loadobj('SIGMA_CACHE.sav')
    print('Found', len(SIGMA_CACHE), 'cached sigma values in save file')

FIGSIZE = 5,5

PLOT = 1

def find_roots(darr):
    """ Approximate root-finding. Cannot find multiple roots that are
    closer together than the array spacing.
    """
    sign = np.sign(darr)
    c0 = np.zeros(len(darr), dtype=bool)
    # The offset isn't quite right here, but it doesn't change much
    c0[1:] = np.abs(sign[1:] - sign[:-1]) == 2
    return c0


def find_zeros_tophat():
    """ Find the zeros of the fourier transform of a tophat function.

    (i.e. the x values satisfying

     sin(x) - x * cos(x) == 0

     for 0 < x < 1e5)

    The first ~30000 zeros (for integration purposes).
    """
    x = np.arange(0, 1e5, 1e-2)

    def f(x):
        return (np.sin(x) - x * np.cos(x)) / x 

    y = f(x)
    c0 = find_roots(y)


    #Now find roots with better accuracy

    ind = np.flatnonzero(c0)
    
    from scipy import optimize
    xroots = []
    xroots.append(optimize.brentq(f, 1, x[ind[0]+1]))
    i = 0
    while i < len(ind) - 1:
        xroots.append(optimize.brentq(f, x[ind[i]+1], x[ind[i+1]+1]))
        i += 1

    xroots = np.array(xroots)

    np.savez('zeros.npz', xroots=xroots)

    #plt.plot(x, y)
    #plt.plot(xroots, f(xroots), 'ro')

    
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

    """
    return 1 / (r_on_rs**gamma * (1 + r_on_rs)**(3-gamma))


def powerspec(y, dt, fig=None):
    """ Find the power spectrum using an FFT. 
    
    The samples y must be equally spaced with spacing dt. It is
    assumed y is real.

    return the coords in fourier transform space (e.g. 1/t) and the power
    spectrum values, 2 * abs(fft)**2
    """
    fft = np.fft.fft(y)
    nu = np.fft.fftfreq(len(y), dt)
    pspec = (2*np.abs(fft)**2)[:len(y)//2 + 1]
    pnu = np.abs(nu)[:len(y)//2 + 1]
    if fig is not None:
        import pylab as pl
        ax = fig.add_subplot(211)
        ax.plot(np.arange(len(y))*dt, y)
        ax.set_ylabel('$y$')
        ax.set_xlabel('T')
        ax = fig.add_subplot(212)

        ax.semilogy(pnu, pspec)
        ax.set_ylabel(r'$2*abs(fft(y))^2$')
        ax.set_xlabel('f = 1/T')

    return pnu, pspec


def M_params(mass, Mstar, redshift, z_params, cosmo1):
    """ Calculate all parameters which depend on the halo mass.

    Note that some also depend on redshift.

    Returns a dictionary with all the derived parameters
    """

    # rearrange M = 4*pi/3. * rho_bar_m * delta_vir * r_vir**3

    # Note that this is a physical (not comoving) distance.
    r_vir = (mass / (4*pi/3. * z_params['rho_bar_m'] *
                     z_params['delta_vir']))**(1/3.)

    # find the scale density rho_s

    # rearrange M = 4*pi * rho_s * r_vir**3 / conc**3  * (log(1+conc) - conc / (1 + conc))

    M_on_mstar = (mass / Mstar).to(u.dimensionless_unscaled).value
    C = concentration(M_on_mstar, redshift)
    rs = r_vir / C

    rho_s = C**3 * mass / (4*pi * r_vir**3 * (log(1+C) - C/(1 + C))) 

    sigma_M = calc_sigma(mass, z_params, redshift, cosmo1)
    nu_M = z_params['delta_c'] / z_params['growth_factor'] / sigma_M
    bM = calc_bias(Z['delta_c'], nu_M)

    # tophat_radius (used for 2-halo term)

    return dict(rvir=r_vir, M_on_mstar=M_on_mstar, C=C, rs=rs, rho_s=rho_s,
                sigma=sigma_M, nu=nu_M, bias=bM)


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


def calc_fnu(nu_M, a=0.707, p=0.3, A=0.3221836349595475):
    """ Find f(nu) using equation A19 from Zhu et al.

    A19 is used to get the mass function, f(nu). Find A using
    int_0^inf fnu dnu = 1. They find 0.129.  We find A=0.322? Could be
    cosmology difference?

    Note that this differs from equation (10) in Sheth and Tormen,
    which suggests it should be sqrt(0.5 * an2 / pi). Altering this
    still doesn't result in A=0.129, though.

    # need to integrate fnu from 0 -> inf and set equal to 1, which sets A.
    res2,_ = scipy.integrate.quad(calc_fnu, 0, np.inf)
    print res2
    """
    # normalization constant is A, assume 1 for now.
    
    an2 = a*nu_M**2
    return A*np.sqrt(2 * an2 / pi) * (1 + an2**-p) * np.exp(-0.5*an2) / nu_M

def calc_fnu_mp(nu_M, a=0.707, p=0.3, A=1):
    """ Find f(nu) using equation A19 from Zhu et al.

    A19 is used to get the mass function, f(nu). Find A using
    int_0^inf fnu dnu = 1. They find 0.129.  We find A=0.322?

    # need to integrate fnu from 0 -> inf and set equal to 1, which sets A.
    res2,_ = scipy.integrate.quad(calc_fnu, 0, np.inf)
    print res2
    """
    # normalization constant is A, assume 1 for now.
    
    an2 = a*nu_M
    return A*mp.sqrt(2 * an2 / pi) * (1 + an2**-p) * mp.exp(-0.5*an2)


def Z_params(z, cosmo):
    """ Quantities which depend on only the redshift and cosmology.

    These *do not* depend on the halo mass.
    """
    Z = {}
    Z['rho_bar_m'] = cosmo.Om(z) * cosmo.critical_density(z)
    Z['delta_vir'] = deltavir(z, cosmo=cosmo)

    # A14, overdensity threshold for spherical collapse
    Z['delta_c'] = 3 / 20. * (12 * pi)**(2/3.) * \
                   (1 + 0.013 * log10(cosmo.Om(z)))
    Z['growth_factor'] = fgrowth(z, cosmo.Om0)
    return Z


def sigma_integrand(k, MASS, Z, COSMO1):
    # k must be in inverse Mpc
    
    # should we use the REDSHIFT here? It's currently set to
    # zero. Setting z=0.6 simply changes the normalisation by a
    # factor of ~2. The description in Zhu et al., just before
    # A15, says that it is the present-day value (i.e. z=0), so
    # that's what we're using.
    tophat_radius_Mpc = ((3 * MASS / (4 * pi * Z['rho_bar_m']))**(1/3.)).to(u.Mpc).value
    return k**2 * power_spectrum(float(k), 0, **COSMO1) / (2*pi**2) \
           * w_tophat(float(k), float(tophat_radius_Mpc))**2

def sigma_integrand_ufunc(k, MASS, Z, COSMO1):
    # Same function as sigma_integrand, but works on arrays of k, not just scalars.
    # Useful for plotting purposes.
    # k must be in inverse Mpc
    tophat_radius_Mpc = ((3 * MASS / (4 * pi * Z['rho_bar_m']))**(1/3.)).to(u.Mpc).value
    return k**2 * power_spectrum(k, 0, **COSMO1) / (2*pi**2) \
           * w_tophat(k, tophat_radius_Mpc)**2


def calc_sigma(MASS, Z, REDSHIFT, COSMO1, debug=False):
    # A15, finds sigma 
    """
    W(x) = 3*(sin(x) - x*cos(x)) / x**3 
    
    sigma**2 = int_0^inf k**3 * Plin(k) / (k * 2 * pi**2) * W  * dk

    where Plin is the matter power spectrum from Eisenstein and Hu.

    """
    # Need to loop over mass and k here
    key_cosmo = tuple(val for val in sorted(COSMO1.keys()))
    key = (MASS.value, REDSHIFT, key_cosmo)

    if key in SIGMA_CACHE:
        print('    Using cached value for inputs')
        return SIGMA_CACHE[key]
    
    tophat_radius_Mpc = ((3 * MASS / (4 * pi * Z['rho_bar_m']))**(1/3.)).to(u.Mpc).value
    def sigma_integrand(k):
        # k must be in inverse Mpc

        # should we use the REDSHIFT here? It's currently set to
        # zero. Setting z=0.6 simply changes the normalisation by a
        # factor of ~2. The description in Zhu et al., just before
        # A15, says that it is the present-day value (i.e. z=0), so
        # that's what we're using.
         
        return k**2 * power_spectrum(float(k), 0, **COSMO1) / (2*pi**2) \
               * w_tophat(float(k), float(tophat_radius_Mpc))**2

    # def sigma_integrand_ufunc(k):
    #     # Same function as sigma_integrand, but works on arrays of k, not just scalars.
    #     # Useful for plotting purposes.
    #     # k must be in inverse Mpc
    #     return k**2 * power_spectrum(k, 0, **COSMO1) / (2*pi**2) \
    #            * w_tophat(k, tophat_radius_Mpc)**2

    if 0:
        # plot the sigma_integrand function using the _ufunc version.
        # It's a bastard to numerically integrate.
        fig = plt.figure()
        fig.clf()
        ax = plt.gca()
        kvals = np.logspace(-2, 2, 1e5)
        ax.loglog(kvals, sigma_integrand_ufunc(kvals))
        plt.show()

    if debug:
        print('Integrating over all k 0 -> infinity to find sigma_M')

        #res  = quad(sigma_integrand, 0, np.inf)

    xroots = np.load('zeros.npz')['xroots']
    kroots = xroots / tophat_radius_Mpc
    kroots = np.array([0] + list(kroots.tolist()))

    t1 = time.time()
    res = 0
    resall = []
    for i,(k0,k1) in enumerate(zip(kroots[:-1], kroots[1:])):
        if i > 500:
            break
        out = quad(sigma_integrand, k0, k1)[0]
        res += out
        resall.append(out)
        #print(i,len(kroots))

    #res = mp.quad(sigma_integrand, (0, mp.inf))
    t2 = time.time()

    if debug:
        print('Done using mp.quad in', t2 - t1, 's')

    # this is ~3 times slower than mp.quad.
    #res2,_ = quad(sigma_integrand, 0, np.inf)
    #print 'quad', time.time() - t2, 's'
    #this takes ages, but gives pretty much the same answer (to within 1e-6) as mp.quad
    #res2 = mp.quadosc(sigma_integrand, (0, mp.inf), omega=tophat_radius)
    #print 'quadosc done', (t2 - time.time())/60., 's'
    #print res, res2

    sigma_M = sqrt(res)

    SIGMA_CACHE[key] = sigma_M

    return sigma_M


def Phm_integrand(kval, mval, debug=0):
    """ Find the bit which goes inside the integral in A12

    uses global variables Z, REDSHIFT, COSMO1

    integrate using this function to evaluate A12 in Zhu et al.
    """
    mval = mval * u.M_sun

    # find new parameters for this halo mass. This also calculates
    # sigma (which is pretty slow) and nu_M
    mpar = M_params(mval, MSTAR, REDSHIFT, Z, COSMO1)

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

    # This effectively integrates to r = infinity. But maybe should
    # just integrate to the virial radius instead? (following eqan
    # (80) in Cooray and Sheth 2002)

    u_nu = 0
    # testing comparing to mp.quadosc suggests 10000 give a fractional
    # error < 1e-5 (i.e. 0.001%). It is about a factor of 10 faster.
    per = np.pi / kval
    for i in xrange(10000):
        if (i+1) * per > mpar['rvir']:
        # only integrate to the virial radius, following Cooray and Sheth.
            break
        u_nu += quad(u_knu_integrand, i * per, (i+1) * per)[0]

    # this is 3 times faster, but less accurate.
    #u_nu_temp = quad(u_knu_integrand, 0, np.inf)
    #u_nu = float(mp.quad(u_knu_integrand, (0, mp.inf)))

    # fnu, A19
    fnu = calc_fnu(mpar['nu'])

    integrand = u_nu * mpar['bias'] * fnu

    if debug:
        print('    u_nu {:g} bias {:g} fnu {:g}'.format(u_nu, mpar['bias'], fnu))
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
    
    ################################################################
    # Setup and general quantities needed for 1- and 2-halo terms
    ################################################################
    Z = Z_params(REDSHIFT, cosmo)
    M = M_params(MASS, MSTAR, REDSHIFT, Z, COSMO1)

    print('Virial radius {:g}'.format(M['rvir'].to(u.kpc)))
    print('mass/M* {:g}'.format(M['M_on_mstar']))
    print('concentration {:g}'.format(M['C']))
    print('Scale density {:g}'.format(M['rho_s'].to(u.g / u.cm**3)))


    ########################################
    # One halo term
    ########################################
    
    # test plot of the density profile

    rvals = np.logspace(-1, 4) * u.kpc
    R_M = (rvals / M['rs']).to(u.dimensionless_unscaled).value    
    rho_m_M = M['rho_s'] * rho_nfw(R_M)

    if PLOT:
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
    
    
    if PLOT:
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
    
    ###################################
    # two halo term
    ###################################
    """
    sigma_2h = rho_bar_m * int -inf to +inf   xi_hm (sqt(rp**2 + s**2)) ds
    """

    # Calculate this on a big array of k. Just one halo mass for now.

    try:
        saved = np.load('temp.npz')
    except IOError:
        saved = None
    
    # M goes from M_min to M_max ()

    # range of halo masses (need to integrate the whole of the below
    # over these)
    num_M = 50

    logMlo = 3
    logMhi = 17
    logMvals = np.linspace(logMlo, logMhi + .01, num_M)

    # distances from 100 kpc (logk = 1) to 100 Mpc (logk = -2),
    # comoving (?)
    num_k = 100
    logkvals = np.linspace(-2, 1, num_k)

    if saved is not None and \
           len(logkvals) == len(saved['logk']) and \
           len(logMvals) == len(saved['logM']) and \
           np.allclose(logkvals, saved['logk']) and \
           np.allclose(logMvals, saved['logM']):
        print('Using saved results')
        out = saved['out']
    else:
        # regenerate the results
        t1 = time.time()
    
        out = np.zeros((num_k, num_M))
        for i in range(num_k):
            kval = 10**logkvals[i]
            s = '{} of {}, k={:g}'.format(i+1, num_k, kval)
            for j in range(num_M):
                mval = 10**logMvals[j]
                print(s + '  {} of {}, M={:g}'.format(j+1, num_M, mval))
                res = Phm_integrand(kval, mval)
                #print(res)
                out[i,j] = res
    
            # save as we go
            #with open('temp.npz', 'w') as fh:
            #    print('Updating temp.npz')
            #    np.savez(fh, out=out, logk=logkvals, logM=logMvals)

        t2 = time.time()
        print('Total time elapsed {} min'.format((t2 - t1) / 60))

    saveobj('SIGMA_CACHE.sav', SIGMA_CACHE, overwrite=1)
    
    Phm_term = np.trapz(out, x=10**logMvals, axis=-1)

    Phm = Phm_term * M['bias'] * power_spectrum(10**logkvals, 0, **COSMO1)

    kvals = 10**logkvals

    # now take the fourier transform of Phm to get the correlation function.

    # xi(r) = ind dk^3 / (2 * pi)^3 * P(k) * exp(i * k dot r)

    # xi(r) = ind dk / (2 * pi) * P(k) * exp(i * k * r)

    # Note that exp(i * k * r) = cos(kr) + i * sin(kr). We only want
    # the real part.

    # where k and r are vectors

    # note rvals should not be outside limits corresponsing to
    # largest and smallest k value.
    rvals = np.logspace(-1, 2, 50)
    xi_real = []
    xi_imag = []
    for r in rvals:
        #print(r)
        # needed for numerical integration to deal with oscillations
        half_period = pi / r
        def f(k):
            return np.interp(k, kvals, Phm, right=0) * cos(k * r) / (2 * pi) 

        def f_imag(k):
            return np.interp(k, kvals, Phm, right=0) * sin(k * r) / (2 * pi) 

        i = 0
        tot_real = 0.
        tot_imag = 0.
        while f(i * half_period) > 0:
            tot_real += quad(f, i * half_period, (i+1) * half_period)[0]
            tot_imag += quad(f_imag, i * half_period, (i+1) * half_period)[0]
            i += 1

        xi_real.append(tot_real)
        xi_imag.append(tot_imag)


    # xi_real is the two-halo correlation function. Need to integrate
    # this along the redshift direction.

    rp_vals = np.logspace(-1, 2, 50)

    # rp must be in units of physical kpc
    xi2h_rp = []
    for rp in rp_vals:
        def integrand(s):
            """rp and s must have the same units """
            R = hypot(s, rp)
            return np.interp(R, rvals, xi_real)

        xi2h_rp.append(quad(integrand, -np.inf, np.inf)[0])


    # make a plot of the 1 halo and two halo results. The
    # normalisations for each term are arbitrary for now

    # convert to proper (physical) coordinates.
    rp_vals1 = rp_vals / (1 + REDSHIFT)
    xi1h_rp = Sigma_1h(rp_vals1, M)
    plt.figure()
    ax = plt.gca()
    ax.plot(np.log10(rp_vals1), np.log10(xi1h_rp.value) - 21.4)
    ax.plot(np.log10(rp_vals1), np.log10(xi2h_rp))
    from barak.plot import make_log_xlabels, puttext
    puttext(0.9, 0.9, 'log10(mass/m_sun) {:.3g}'.format(np.log10(MASS.value)),
            ax, ha='right')
    make_log_xlabels(plt.gca())
    ax.set_xlabel('log10 r_p (Mpc)')
    ax.set_ylabel('log10 xi')
    plt.show()

