"""
Make a plot of different QSO properties as a function of rho
"""

if 0:
    #outnames = ['mass_13.6-14.0_erphotz_rho_26.0/rho_dNdz_clus.sav',
    #            'mass_14.0-14.4_erphotz_rho_26.0/rho_dNdz_clus.sav',
    #            'mass_14.4-20.0_erphotz_rho_26.0/rho_dNdz_clus.sav',]
    outnames = ['mass_13.6-20.0_erphotz_rho_46.0/rho_dNdz_clus.sav']


    rhovals = []
    for n in outnames:
        print 'Reading', n
        rhovals.append(loadobj(n))
        

fig = plt.figure(figsize=(20,5))
plot_rho_QSO_prop(fig, rho, ab, iqso_from_id)


