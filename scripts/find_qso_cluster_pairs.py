from __future__ import division
from barak.coord import ang_sep, match as match_radec
from barak.utilities import between, indgroupby, flatten, Bins
from barak.plot import errplot, puttext, make_log_xlabels, \
     make_log_ylabels
from barak.io import loadobj,saveobj
from astropy.table import Table, Column
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM, Planck13
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.mlab import rec_append_fields


#cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
cosmo = Planck13
CALC = 1
PLOTRES = 1
DEBUG = False

WMIN_MGII_ZSEARCH = 3800.
WMAX_MGII_ZSEARCH = 9100.

ZMIN_CLUS = 0.33

# a hit is an absorber within +/- CLUS_ZERR of a cluster redshift
#CLUS_ZERR = 0.03
CLUS_ZERR = 'erphotz'

# minimum richness
#MAXRICH = 1000
#MINRICH = 10

log10MINMASS = 13.6
log10MAXMASS = 20.0

# 10^13.6, 10^13.8, 10^14, 10^14.2, 10^14.4, 10^14.4 and above have all
# roughly equal numebrs of clusters.

# stop MgII z search path this many km/s from QSO MgII emission
# (negative means bluewards)
DV_MAX_ZSEARCH = -5000.
# start MgII z search path this many km/s from QSO Lya emission
# (positive means redwards)
DV_MIN_ZSEARCH = 3000.

# maximum rho for matching qso-cluster pairs in proper Mpc 
MAX_RHO_PROP_MPC = 26.

run_id = 'mass_{}-{}_{}_rho_{}'.format(
    log10MINMASS, log10MAXMASS, CLUS_ZERR, MAX_RHO_PROP_MPC)

c_kms = 299792.458         # speed of light km/s, exact
wlya = 1215.6701
w2796 = 2796.3542699
w2803 = 2803.5314853

print run_id

if not os.path.exists(run_id):
    print 'Creating directory', run_id
    os.mkdir(run_id)

prefix = '/Users/ncrighton/Projects/qso_clusters/'
#prefix = '/media/ntejos/disk1/catalogs/'

def plot_rho_QSO_prop(fig,rho,ab,iqso_from_id,qso_props=None):
    """For a given well defined rho dictionary, it plots different QSO
    properties with the aim to test fo systematics.

    """
    global rbin
    if qso_props is None:
        qso_props=['z','snr','nabs','RMAG','IMAG','ZMAG']
    
    for i,qso_prop in enumerate(qso_props):
        mean = []
        err = []
        median = []
        n = []
        for r in rho:
            # QSOs in this bin
            qids = np.array([iqso_from_id[ind] for ind in r['qid']], dtype=int)
            prop = ab['qso'][qso_prop][qids]
            mean += [np.mean(prop)]
            err  += [np.std(prop)]
            median += [np.median(prop)]
            n  += [len(qids)]
        ax = fig.add_subplot(1,len(qso_props),i)
        ax.errorbar(rbin.cen,mean,yerr=err,capsize=0,fmt='-o',color='k')
        ax.plot(rbin.cen,median,'-r',lw=3)
        for j in range(len(n)):
            puttext(rbin.cen[j], 0.1 , str(n[j]), ax, color='k',
                    fontsize=10, xcoord='data', ha='center')
        ax.set_xlabel('rho')
        ax.set_ylabel(qso_prop)


def append_QSO_props(ab,qso_props=None):
    """For each QSO in ab['qso'], it finds the corresponding one to the
    catalog of Scheider et al. for DR7 QSOs and append properties to
    ab['qso'].

    """
    #Tables work better for me to append
    qso_orig = Table(ab['qso'])

    qsos_dr7 = fits.getdata('/media/ntejos/disk1/catalogs/qso_sdss_dr7/dr7qso.fit') #Scheider et al.
    if qso_props is None:
        props = ['RMAG','IMAG','ZMAG'] #subsample of properties in qsos_dr7
    
    #find the matched indices
    matches = match_radec(qso_orig['ra'],qso_orig['dec'],qsos_dr7['RA'],qsos_dr7['DEC'],2.)
    
    #append properties to the original ab['qso']
    for prop in props:
        aux_prop = qsos_dr7[prop][matches['ind']] # matched property
        aux_prop = np.where(matches['ind']==-1,-1,aux_prop) #replace unmatched values with -1
        qso_orig.add_column(Column(data=aux_prop,name=prop))
    return qso_orig



def plot_hist(run_id, clus, MgII, title):
    """ Plot cluster and MgII redshift histograms. """
    zbin = Bins(np.arange(-0.1, 1.2, 0.025))
    fig = plt.figure(1, figsize=(4.5,4.5))
    fig.clf()
    ax = plt.gca()
    vals,_ = np.histogram(clus.z, bins=zbin.edges)
    y = np.where(vals == 0, -10, np.log10(vals))
    ax.plot(zbin.cen, y, 'r', lw=2, ls='steps-mid',
            label='clusters (n=%i)' % len(clus), zorder=10)
    vals,_ = np.histogram(MgII['z'], bins=zbin.edges)
    label = 'MgII (n={})'.format(len(MgII))
    y = np.where(vals == 0, -10, np.log10(vals))
    ax.plot(zbin.cen, y, 'b',lw=2, ls='steps-mid', label=label)
    ax.set_xlabel('$\mathrm{Redshift}$')
    ax.set_ylabel('$\mathrm{Number}$')
    y0,y1 = ax.get_ylim()
    ax.set_ylim(-0.8, y1)
    ax.set_xlim(0.25, 1.05)
    make_log_ylabels(ax)
    ax.legend(frameon=0, fontsize=8)
    ax.set_title(title)
    plt.savefig(run_id + '/zhist.png', dpi=200, bbox_inches='tight')

def match_clus_qso(clus, qso, filename=None, rho=MAX_RHO_PROP_MPC):
    """find all QSO - cluster pairs with impact params < rho.

    Need fields:
    
    clus['ra'], clus['dec'], clus['z'], clus['id'], clus['rlos']

    where rlos is the comoving line of sight distance to each cluster

    qso['ra'], qso['dec'], qso['zmin_MgII'], qso['zmax_MgII'], qso['qid']

    rho is in same units as rlos (default assumes Mpc), and is a proper
    distance
    """
    print 'Matching'

    iqso = []
    iclus = []
    zclus = []
    zclus_er = []
    seps_deg = []
    seps_Mpc = []

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

    if filename is not None:
        Table(qnear).write(filename)

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

    dNdz, dNdz_err, nabs = map(np.array, zip(*res))
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
    arr = [qso['RA'], qso['DEC'], qso['ZQSO'], qso_zmin, qso_zmax,
           qso['INDEX_QSO']]
    qso1 = np.rec.fromarrays(arr, names='ra,dec,z,zmin_mg2,zmax_mg2, qid')
    arr = [MgII.ZABS, MgII.REW_MGII_2796, MgII.INDEX_QSO, np.arange(len(MgII))]
    MgII1 = np.rec.fromarrays(arr, names='z,Wr,qid,abid')
    MgII1.sort(order='qid')

    iMgII_from_id = {ind:i for i,ind in enumerate(MgII1['abid'])}
    iqso_from_id = {ind:i for i,ind in enumerate(qso1['qid'])}

    return dict(MgII=MgII1, qso=qso1), iqso_from_id, iMgII_from_id


if CALC:
    print 'Reading cluster and MgII catalogues'
    clus, iclus_from_id = read_redmapper()
    ab, iqso_from_id, iMgII_from_id = read_zhu()

    qso = ab['qso']
    # find qso sightlines that are within 10 proper Mpc of a foreground cluster.

    if os.path.exists(run_id + '/qso_cluster_pairs.fits'):
        print 'Reading', run_id + '/qso_cluster_pairs.fit'
        pairs0 = fits.getdata(run_id + '/qso_cluster_pairs.fits')
    else:
        # takes about 10 min to run.
        pairs0 = match_clus_qso(clus, qso, 
            filename=run_id + '/qso_cluster_pairs.fits')

    # assign a unique identifier to each pair. modifies pairs in place.

    pairs0 = rec_append_fields(pairs0, ['pid'], [np.arange(len(pairs0))])

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
    rbin = Bins(np.arange(-1.4, 1.61, 0.2))

    outname = run_id + '/rho_dNdz_clus.sav'
    if os.path.exists(outname):
        print 'Reading', outname
        rho = loadobj(outname)
    else:
        rho = [dict(zpathlim=[], abid=[], pid=[], cid=[], qid=[],
                    Wr=[], Wre=[], zpathtot=0) for i in xrange(len(rbin.cen))]
    
        # find tot zpath (including both field and cluster paths up to
        # z=1, only towards sightlines with a nearby cluster though) also?
    
        print 'Calculating MgII hits close to clusters, and the total z path length'
    
        if DEBUG:
            fig4 = plt.figure(4, figsize=(6,6))
            ax = fig4.add_subplot(111)

        print 'Looping over QSOs'
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
                cz = p['cz']
                if DEBUG:
                    print '    pair cluster z %.3f, sep Mpc %.2f' % (
                        p['cz'], p['sepMpc_prop'])
                # check MgII detection range overlaps with cluster z
                # if not, skip to next cluster
                if cz < zmin_mg2:
                    continue
                if cz > zmax_mg2:
                    continue

                # redshift uncertainty in cluster                    
                zmin = max(cz - p['cz_er'], zmin_mg2)
                zmax = min(cz + p['cz_er'], zmax_mg2)
                #assert zmax > zmin
    
                close_z = between(mg2['z'], zmin, zmax)
                closeids = mg2['abid'][close_z]
                nabs = len(closeids)


                if DEBUG:
                    print '    nMgII', nabs
                    print '    MgII close', mg2[close_z]['z']
                    print '    ic={:i}, zmin={.3f}, zmax={.3f}'.format(
                        p['ic'], zmin, zmax)
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
    fig3 = plt.figure(3, figsize=(7.5,7.5))
    #fig3.subplots_adjust(left=0.16)
    fig3.clf()
    ax = plt.gca()
    ax.set_title(run_id)

    ewbins = Bins([0.6, 5.0])
    labels = ['0.6 < Wr$_{2796}$ < 5']
    colors = 'g'
    symbols = 'o'
    offsets = [0]

    for i in range(len(labels)):
        dNdz, dNdz_er, n = find_dndz_vs_rho(
            rho, ab['MgII'], iMgII_from_id,
            ewbins.edges[i], ewbins.edges[i+1])
        y = np.log10(dNdz)
        ylo = np.log10(dNdz - dNdz_er)
        yhi = np.log10(dNdz + dNdz_er)
        errplot(rbin.cen + offsets[i], y, (ylo, yhi), ax=ax,
                fmt=colors[i]+symbols[i], label=labels[i])
        for j in range(len(n)):
            puttext(rbin.cen[j], 0.03 + i*0.03, str(n[j]), ax, color=colors[i],
                    fontsize=10, xcoord='data', ha='center')
    
    ax.legend(frameon=0)
    ax.set_xlabel('Cluster-absorber impact par. (proper Mpc)')
    ax.set_ylabel(r'$dN/dz\ (MgII)$')
    # skip last bin, where not all pairs are measured.
    ax.set_xlim(rbin.edges[0] - rbin.halfwidth[0],
                rbin.edges[-2] + rbin.halfwidth[-1])
    make_log_xlabels(ax)
    make_log_ylabels(ax)
    #fig3.savefig(run_id + '/dNdz_vs_rho.png')
    plt.show()


if 0:
    # check whether QSO properties are consistent across bins
    qso_orig = append_QSO_props(ab)
    ab['qso'] = qso_orig
    fig = plt.figure(figsize=(20,5))
    plot_rho_QSO_prop(fig, rho, ab, iqso_from_id)
