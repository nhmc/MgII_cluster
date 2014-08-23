from __future__ import division
from barak.coord import ang_sep
from barak.utilities import between, indgroupby, flatten, Bins
from barak.plot import errplot, puttext, make_log_xlabels, \
     make_log_ylabels
from barak.io import loadobj,saveobj
from astropy.table import Table
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, Planck13
import os
import numpy as np
import matplotlib.pyplot as plt

#cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
cosmo = Planck13
CALC = 1
PLOTRES = 1
DEBUG = False

WMIN_MGII_ZSEARCH = 3800.
WMAX_MGII_ZSEARCH = 9100.
MgIICAT = 'Zhu'
#MgIICAT = 'Britt'
CLUSCAT = 'Redmapper'
#CLUSCAT = 'GMBCG'

ZMIN_CLUS = 0.33

# a hit is an absorber within +/- CLUS_ZERR of a cluster redshift
#CLUS_ZERR = 0.03
CLUS_ZERR = 'erphotz'
# minimum richness (S_CLUSTER for GMBCG)
#MAXRICH = 1000
#MINRICH = 10

log10MINMASS = 13.6
log10MAXMASS = 20.0

# 10^13.6, 10^13.8, 10^14, 10^14.2, 10^14.4, 10^14.4 and above have all
# roughly equal numebrs of clusters.

# stop MgII z search path this many km/s of QSO MgII emission (negative means bluewards)
DV_MAX_ZSEARCH = -5000.
# start MgII z search path this many km/s if QSO Lya emission (positive means redwards)
DV_MIN_ZSEARCH = 3000.

run_id = 'mass_{}-{}_{}'.format(log10MINMASS, log10MAXMASS, CLUS_ZERR)

# run_id = '{}_{}_rich_{}-{}_{}_{}_{}'.format(
#     MgIICAT, CLUSCAT, MINRICH, MAXRICH, CLUS_ZERR, WMIN_MGII_ZSEARCH, WMAX_MGII_ZSEARCH)    


c_kms = 299792.458         # speed of light km/s, exact
wlya = 1215.6701
w2796 = 2796.3542699
w2803 = 2803.5314853

print run_id

if not os.path.exists(run_id):
    print 'creating directory', run_id
    os.mkdir(run_id)

prefix = '/Users/ncrighton/Projects/qso_clusters/'
#prefix = '/media/ntejos/disk1/catalogs/'

def add_columns(table, names, arr, indexes=None):
    from astropy.table import Column
    if isinstance(names, basestring):
        names = [n.strip() for n in names.split(',')]
    assert len(names) == len(arr)
    columns = [Column(name=names[i], data=arr[i]) for i in range(len(names))]
    table.add_columns(columns, indexes=indexes)

def match_clus_qso(clus, qso, filename=None, rho=20):
    """find all QSO - cluster pairs with impact params < rho.

    clus['ra'], clus['dec'], clus['z'], clus['id'], clus['rlos']
    rlos is the comoving line of sight distance to each cluster
    qso['ra'], qso['dec'], qso['zmin_MgII'], qso['zmax_MgII'], qso['qid']

    rho is in same units as rlos (default assumes Mpc), and is a proper
    distance
    """
    print 'matching'

    iqso = []
    iclus = []
    zclus = []
    zclus_er = []
    seps_deg = []
    seps_Mpc = []

    #fig9 = plt.figure(9)
    #ax = pl.gca()
    #ax.cla()
    for i in xrange(len(clus)):
        # find impact parameter in degrees corresponding to maximum rho in proper Mpc
        # note rho is a proper distance
        scalefac = 1. / (1. + clus['z'][i])
        comoving_sep = rho / scalefac
        if not i % 5000: print i, 'of', len(clus), 'maximum rho comoving',\
           comoving_sep, 'Mpc'
        angsep_deg = comoving_sep / clus['rlos'][i] * 180. / np.pi
        # throw away everything a long way away
        c0 = between(qso['dec'], clus['dec'][i] -  angsep_deg,
                     clus['dec'][i] + angsep_deg)
        qra = qso['ra'][c0]
        qdec = qso['dec'][c0]
        qzmin = qso['zmin_mg2'][c0]
        qzmax = qso['zmax_mg2'][c0]
        seps = ang_sep(clus['ra'][i], clus['dec'][i], qra, qdec)
        # within rho
        close_angsep = seps < angsep_deg
        # with a smaller redshift than the b/g qso but large enough
        # redshift that we could detect MgII
        close_z = between(clus['z'][i], qzmin, qzmax)
        c1 = close_angsep & close_z
        num = c1.sum()
        #import pdb; pdb.set_trace()
        iqso.extend(qso['qid'][c0][c1])
        iclus.extend([clus['id'][i]] * num)
        zclus.extend([clus['z'][i]] * num)
        zclus_er.extend([clus['zer'][i]] * num)
        seps_deg.extend(seps[c1])
        seps_Mpc.extend(np.pi / 180 * clus['rlos'][i] * seps[c1])

    seps_Mpc = np.array(seps_Mpc)
    seps_Mpc_prop = seps_Mpc / (1 + np.array(zclus))
    logsepMpc_prop = np.log10(seps_Mpc_prop)
    logsepMpc_com = np.log10(seps_Mpc)
    names = ('qid cid sepdeg sepMpc_com sepMpc_prop '
             'logsepMpc_com logsepMpc_prop cz cz_er').split()
    qnear = np.rec.fromarrays(
        [iqso, iclus, seps_deg, seps_Mpc, seps_Mpc_prop,
         logsepMpc_com, logsepMpc_prop, zclus, zclus_er],
        names=names)

    qnear = Table(qnear)
    if filename is not None:
        qnear.write(filename)

    return qnear


def find_dndz_vs_rho(rho, mg2, iMgII_from_id, ewrestmin, ewrestmax):
    """ Find dNdz as a function of rho for absorbers with rest ew >
    ewrestmin given a list of results in rho."""
    res = []
    for irho,r in enumerate(rho):
        #print 'bin', irho
        # absorbers in this bin
        i = np.array([iMgII_from_id[ind] for ind in r['abid']], dtype=int)
        mg2_ = mg2[i]
        c0 = between(mg2_['Wr'], ewrestmin, ewrestmax)
        l = c0.sum()
        res.append((l / r['zpathtot'], np.sqrt(l) / r['zpathtot'], l))

    dNdz, dNdz_err, nabs = zip(*res)
    print nabs
    return dNdz, dNdz_err, nabs

def m200_from_richness(lam):
    """ Find m200 in solar masses from the richness lambda in the
    redmapper catalogue. """
    return 10**14 * (lam/60.)**1.08 * np.exp(1.72)

def read_redmapper():
    d = fits.getdata(prefix + 'clusters/redmapper/'
                     'dr8_run_redmapper_v5.10_lgt5_catalog.fits')
    #d = fits.getdata(prefix + 'clusters/redmapper/DR8/'
    #                 'dr8_run_redmapper_v5.10_lgt5_catalog.fit')

    z = d['Z_LAMBDA']
    c0 = d['BCG_SPEC_Z'] != -1 
    z[c0] = d['BCG_SPEC_Z'][c0]
    zer = d['Z_LAMBDA_E']
    if CLUS_ZERR == 'erphotz':
        zer[c0] = 0.001
    elif isinstance(CLUS_ZERR, float):
        zer[:] = CLUS_ZERR
    else:
        raise ValueError

    # 0.005 corresponds to a velocity dispersion of 937 km/s at z=0.6 
    zer = np.where(zer < 0.005, 0.005, zer)

    if os.path.exists('dc_redmapper.sav'):
        rlos = loadobj('dc_redmapper.sav')
        assert len(rlos) == len(d)
    else:
        # this takes about 5 min to run
        print 'calculating comoving distances'
        rlos = cosmo.comoving_distance(z)
        saveobj('dc_redmapper.sav', rlos)

    # in solar masses, conversion from Rykoff 2013 appendix B.
    m200 = m200_from_richness(d['LAMBDA_CHISQ'])

    d1 = np.rec.fromarrays([d.RA, d.DEC, z, zer,
                            d.LAMBDA_CHISQ, d.MEM_MATCH_ID, rlos.value, m200],
                           names='ra,dec,z,zer,richness,id,rlos,m200')
    d2 = d1[d1.z > ZMIN_CLUS]
    d3 = d2[between(np.log10(d2['m200']), log10MINMASS, log10MAXMASS)]
    #d3 = d2[between(d2['richness'], MINRICH, MAXRICH)]

    iclus_from_id = {idval:i for i,idval in enumerate(d3.id)}
    return d3, iclus_from_id


def get_MgII_zsearch_lim(zqso):
    """ Find the MgII search limits for an array of QSO redshifts.
    """
    zmin_lya = (1 + zqso) * (1 + DV_MIN_ZSEARCH/c_kms) * wlya / w2796 - 1
    zmin_blue_cutoff = WMIN_MGII_ZSEARCH / w2796 - 1
    zmin = zmin_lya.clip(zmin_blue_cutoff, 1000)

    zmax_zqso = (1 + zqso) * (1 + DV_MAX_ZSEARCH/c_kms) - 1
    zmax_red_cutoff = WMAX_MGII_ZSEARCH / w2803 - 1.
    zmax = zmax_zqso.clip(0, zmax_red_cutoff)

    cond = zmax > zmin
    zmax[~cond] = zmin[~cond] 
    return zmin, zmax


def read_zhu():
    MgII = fits.getdata(prefix + '/MgII/Expanded_SDSS_DR7_107.fits')
    qso0 = fits.getdata(prefix + '/MgII/QSObased_Expanded_SDSS_DR7_107.fits')

    #MgII = fits.getdata(prefix + '/MgII/JHU-SDSS/Expanded_SDSS_DR7_107.fits')
    #qso0 = fits.getdata(prefix + '/MgII/JHU-SDSS/QSObased_Expanded_SDSS_DR7_107.fits')

    # find the min & max redshift for MgII search path
    qso_zmin, qso_zmax = get_MgII_zsearch_lim(qso0['ZQSO'])

    # remove qsos with tiny z search paths
    cond = (qso_zmax - qso_zmin) > 0.05

    cond &= qso_zmin < 0.9
    
    qso = qso0[cond]
    qso_zmin = qso_zmin[cond]
    qso_zmax = qso_zmax[cond]

    # add in DR9 too later? (not as many as DR7). Need to check there
    # is no overlap in QSOs first.
    arr = [qso['RA'], qso['DEC'], qso['ZQSO'], qso_zmin, qso_zmax, qso['INDEX_QSO']]
    qso1 = np.rec.fromarrays(arr, names='ra,dec,z,zmin_mg2,zmax_mg2, qid')
    arr = [MgII.ZABS, MgII.REW_MGII_2796, MgII.INDEX_QSO, np.arange(len(MgII))]
    MgII1 = np.rec.fromarrays(arr, names='z,Wr,qid,abid')
    MgII1.sort(order='qid')

    iMgII_from_id = {ind:i for i,ind in enumerate(MgII1['abid'])}
    iqso_from_id = {ind:i for i,ind in enumerate(qso1['qid'])}

    return dict(MgII=MgII1, qso=qso1), iqso_from_id, iMgII_from_id

def plot_hist(run_id, clus, MgII, title):
    zbin = Bins(np.arange(-0.1, 1.2, 0.025))
    fig = plt.figure(1, figsize=(4.5,4.5))
    fig.clf()
    ax = plt.gca()
    vals,_ = np.histogram(clus.z, bins=zbin.edges)
    y = np.where(vals == 0, -0.1, np.log10(vals))
    ax.plot(zbin.cen, y, 'r', lw=2, ls='steps-mid',
            label='clusters (n=%i)' % len(clus), zorder=10)
    vals,_ = np.histogram(MgII['z'], bins=zbin.edges)
    label = 'MgII (n={})'.format(len(MgII))
    y = np.where(vals == 0, -0.1, np.log10(vals))
    ax.plot(zbin.cen, y, 'b',lw=2, ls='steps-mid', label=label)
    ax.set_xlabel('$\mathrm{Redshift}$')
    ax.set_ylabel('$\log_{10}(\mathrm{Number})$')
    ax.set_xlim(0.25, 0.9)
    plt.legend(frameon=0, fontsize=8)
    plt.title(title)
    plt.savefig(run_id + '/zhist.png', dpi=200)



if 1:
    clus, iclus_from_id = read_redmapper()
    ab, iqso_from_id, iMgII_from_id = read_zhu()

if 0:
    # Compare the cluster and MgII positions
    plt.figure()
    plt.plot(clus.ra, clus.dec, 'x')
    plt.plot(ab['qso'].ra, ab['qso'].dec, '+', ms=8, mew=2)
    plt.show()

    # they overlap nicely

if 0:
    # plot the equivalent width distribution for MgII
    plt.figure()
    plt.hist(ab['MgII'].Wr, log=True, bins=np.arange(0, 20, 0.1))

if CALC:
    qso = ab['qso']
    # find qso sightlines that are within 10 proper Mpc of a foreground cluster.

    if os.path.exists(run_id + '/qso_cluster_pairs.fits'):
        pairs0 = Table.read(run_id + '/qso_cluster_pairs.fits')
    else:
        # takes about 10 min to run.
        pairs0 = match_clus_qso(clus, qso, 
            filename=run_id + '/qso_cluster_pairs.fits')

    # assign a unique identifier to each pair. modifies pairs in place.
    add_columns(pairs0, ['pid'], [np.arange(len(pairs0))])


if 0:
    # Plot the redshift distributions for clusters and MgII
    zbin = Bins(np.arange(-0.1, 1.2, 0.1))
    fig = plt.figure(1)
    fig.clf()
    ax = plt.gca()
    vals,_ = np.histogram(clus.z, bins=zbin.edges)
    ax.plot(zbin.cen, vals/10., 'm', lw=2, ls='steps-mid',
            label='redmapper (n=%i)' % len(clus), zorder=10)
    vals,_ = np.histogram(britt.z_abs[britt.z_abs > 0], bins=zbin.edges)
    ax.plot(zbin.cen, vals, 'g',lw=2, ls='steps-mid', label='Britt MgII')
    vals,_ = np.histogram(ab['MgII'].z, bins=zbin.edges)
    ax.plot(zbin.cen, vals, 'k',lw=2,
            drawstyle='steps-mid', label='Zhu dr7 MgII exp')

    ax.set_xlabel('Redshift')
    #ax.set_ylim(-0.2, 1.8)
    plt.legend(frameon=0, fontsize=8)
    plt.show()

if PLOTRES:
    plot_hist(run_id, clus, ab['MgII'], run_id)

if CALC:
    cids = clus['id']
    pairs = pairs0[np.in1d(pairs0['cid'], cids)]

    # for each qso-cluster pair find any absorbers with impact par <
    # 1 Mpc within some z range of the cluster.

    # for rho < 1
    # z path length within 1Mpc of cluster per pair
    # absorber id for a cluster-absorber pair
    # pair ids where the cluster in the pair is near an absorber
    # cluster id
    # qso id
    # total zpath over all pairs

    #LOGBINS = False
    #rbin = Bins(np.arange(0, 11, 1))

    LOGBINS = True
    rbin = Bins(np.arange(-2, 1.21, 0.2))

    outname = run_id + '/rho_dNdz_clus.sav'
    if os.path.exists(outname):
        print 'Reading results from', outname
        rho = loadobj(outname)
    else:
        rho = [dict(zpathlim=[], abid=[], pid=[], cid=[], qid=[], zpathtot=0) for
               i in range(len(rbin.cen))]
    
        # find tot zpath (including both field and cluster paths up to
        # z=1, only towards sightlines with a nearby cluster though) also?
    
        print 'Calculating MgII hits close to clusters, and the total z path length'
    
        if DEBUG:
            fig4 = plt.figure(4, figsize=(6,6))
            ax = fig4.add_subplot(111)

        for i,(qid,ind) in enumerate(indgroupby(pairs, 'qid')):
            if not i % 2000:
                print i
            # for every cluster near this qso, check zpath, and add to
            # absorbers if necessary
            q = qso[iqso_from_id[qid]]
            zmin_mg2 = q['zmin_mg2']
            zmax_mg2 = q['zmax_mg2']
    
            if DEBUG:
                cl = clus[np.in1d(clus['id'], pairs[ind]['cid'])]
                ax.cla()
                ax.plot(cl['ra'], cl['dec'], 'b+', ms=8, mew=2)
                ax.plot(q['ra'], q['dec'], 'rx', ms=6, mew=2)
                plt.show()
                raw_input('zqso {:.3g} nclus {}'.format(q['z'], len(ind)))
            
            # find all absorbers in this qso
            i0 = ab['MgII']['qid'].searchsorted(qid)
            i1 = ab['MgII']['qid'].searchsorted(qid, side='right')
            mg2 = ab['MgII'][i0:i1]
            # get the MgII detection limits from the first absorber entry
    
            if DEBUG:
                print '{}: qso ID {}, {} MgII, {}, zmin {:.2f}, zmax {:.2f}, zqso {:.2f}'.format(
                    i+1, qid, len(mg2), mg2['z'], zmin_mg2, zmax_mg2, q['z'])
                print 'f/g clus', len(ind)
            #raw_input('  About to loop of pairs for this sightline...')
            # for each cluster near this sightline
            for p in pairs[ind]:
                if DEBUG:
                    print '    pair cluster z %.3f, sep Mpc %.2f' % (p['cz'], p['sepMpc_prop'])
                # check MgII detection range overlaps with cluster z
                # if not, skip to next cluster
                if not between(p['cz'], zmin_mg2, zmax_mg2):
                    if DEBUG:
                        print '    cluster outside MgII region'
                    continue
                # redshift uncertainty in cluster                    
                zmin = max(p['cz'] - p['cz_er'], zmin_mg2)
                zmax = min(p['cz'] + p['cz_er'], zmax_mg2)
                #assert zmax > zmin
    
                close_z = between(mg2['z'], zmin, zmax)
                closeids = mg2['abid'][close_z]
                nabs = len(closeids)
    
                if DEBUG:
                    print '    nMgII', nabs
                    print '    MgII close', mg2[close_z]['z']
                    print '    ic={:i}, zmin={.3f}, zmax={.3f}'.format(p['ic'], zmin, zmax)
                    raw_input('We have a nearby absorber!')

                if LOGBINS:
                    ibin = int((p['logsepMpc_prop'] - rbin.edges[0]) /
                               rbin.width[0])
                else:
                    ibin = int(p['sepMpc_prop'] / rbin.width[0])
                rho[ibin]['zpathlim'].append((zmin, zmax))
                rho[ibin]['abid'].append(closeids)
                rho[ibin]['cid'].append(p['cid'])
                rho[ibin]['pid'].append(p['pid'])
                rho[ibin]['qid'].append(p['qid'])
    
        # count the total redshift path per bin, and the total bumber 
        for i in range(len(rho)):
            zpathlim = np.array(rho[i]['zpathlim'])
            if len(zpathlim) == 0:
                rho[i]['zpathtot'] = 0
            else:
                rho[i]['zpathtot'] = (zpathlim[:,1] - zpathlim[:,0]).sum()
            # ids of absorbers matching clusters
            rho[i]['abid'] = list(flatten(rho[i]['abid']))
    
        print 'Saving to', outname
        saveobj(outname, rho, overwrite=1)


if PLOTRES:

    outname = run_id + '/rho_dNdz_clus.sav'
    #outname = 'rho_dNdz_clus_s_lt_10.sav'
    fig3 = plt.figure(3, figsize=(7.5,7.5))
    fig3.subplots_adjust(left=0.16)
    fig3.clf()
    ax = plt.gca()
    ax.set_title(run_id)

    if 0:
        ewbins = Bins([0.5, 0.7, 1.0, 1.5, 5.0])
        labels = ('0.4 < Wr$_{2796}$ < 0.7', '0.7 < Wr$_{2796}$ < 1.0',
                  '1.0 < Wr$_{2796}$ < 1.5',
                  '1.5 < Wr$_{2796}$ < 5')
        colors = 'gbmr'
        symbols = 'soo^'
        offsets = [-0.075, -0.025, 0.025, 0.075]
    elif 0:
        ewbins = Bins([0.0, 0.6, 1.0, 1.5, 5.0])
        labels = ('0.3 < Wr$_{2796}$ < 0.6',
                  '0.6 < Wr$_{2796}$ < 1.0',
                  '1.0 < Wr$_{2796}$ < 1.5',
                  '1.5 < Wr$_{2796}$ < 5')
        colors = 'gbmr'
        symbols = 'soo^'
        offsets = [-0.075, -0.025, 0.025, 0.075]
    else:
        ewbins = Bins([0.6, 5.0])
        labels = ['0.6 < Wr$_{2796}$ < 5']
        colors = 'g'
        symbols = 'o'
        offsets = [0]

    for i in range(len(labels)):
        dNdz, dNdz_er, n = find_dndz_vs_rho(
            rho, ab['MgII'], iMgII_from_id,
            ewbins.edges[i], ewbins.edges[i+1])
        errplot(np.log10(rbin.cen + offsets[i]), dNdz, dNdz_er, ax=ax,
                fmt=colors[i]+symbols[i], label=labels[i])
        for j in range(len(n)):
            puttext(rbin.cen[j], 0.03 + i*0.03, n[j], ax, color=colors[i],
                    fontsize=10, xcoord='data')
    
    ax.legend(frameon=0)
    ax.set_xlabel('Cluster-absorber impact par. (proper Mpc)')
    ax.set_ylabel(r'$dN/dz\ (MgII)$')
    #ax.minorticks_on()
    ax.set_ylim(-0.09, 0.85)
    ax.set_xlim(-0.5, 10.5)
    make_log_xlabels(ax)
    make_log_ylabels(ax)
    fig3.savefig(run_id + '/dNdz_vs_rho.png')
    plt.show()
