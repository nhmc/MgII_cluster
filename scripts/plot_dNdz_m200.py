"""
Make dN/dz plot comparing different mass cuts
"""

if 1:

    outnames = ['mass_13.6-14.0_erphotz_rho_26.0/rho_dNdz_clus.sav',
                'mass_14.0-14.4_erphotz_rho_26.0/rho_dNdz_clus.sav',
                'mass_14.4-20.0_erphotz_rho_26.0/rho_dNdz_clus.sav',]

    rhovals = []
    for n in outnames:
        print 'Reading', n
        rhovals.append(loadobj(n))

    fig3 = plt.figure(3, figsize=(7.5,7.5))
    #fig3.subplots_adjust(left=0.16)
    fig3.clf()
    ax = plt.gca()
    ax.set_title(run_id)

    ewbins = Bins([0.6, 5.0])
    labels = ['13.6-14', '14-14.4', '14.4+']
    colors = 'grb'
    symbols = 'os^'
    offsets = [0, 0, 0]

    for i in range(len(labels)):

        dNdz, dNdz_er, n = find_dndz_vs_rho(
            rhovals[i], ab['MgII'], iMgII_from_id,
            0.6, 6.0)
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
    #ax.set_xlim(rbin.edges[0] - rbin.halfwidth[0],
    #            rbin.edges[-2] + rbin.halfwidth[-1])

    ax.set_xlim(rbin.edges[5] - rbin.halfwidth[5],
                rbin.edges[-2] + rbin.halfwidth[-1])
    ax.set_ylim(-0.7, 0.8)
    make_log_xlabels(ax)
    make_log_ylabels(ax)
    #fig3.savefig('all_dNdz_vs_rho.png', dpi=200, bbox_inches='tight')
    plt.show()
