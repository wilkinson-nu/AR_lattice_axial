import numpy as np

## specific to 1603.03048 first flux iteration
#sign_convention = -1       ## stupid choice
sign_convention =  1       ## less stupid choice
MV_1603_03048_iter0  = 0.84      ## vector dipole mass
ga_1603_03048_iter0  = sign_convention *1.267 ## axial charge
tc_1603_03048_iter0  = 9*.14*.14 ## GeV^2
t0_1603_03048_iter0  =-0.28      ## GeV^2
xi_1603_03048_iter0  = 3.7058    ## mu_p -mu_n -1, magnetic couplings
#
MN_1603_03048_iter0  = 0.9389    ## GeV, appx nucleon mass
Mpi_1603_03048_iter0 = 0.14      ## GeV, appx pion mass
#
Mp_1603_03048_iter0  = 0.938272  ## GeV, proton mass
Mn_1603_03048_iter0  = 0.939565  ## GeV, neutron mass
me_1603_03048_iter0  = 0.000511  ## GeV, electron mass
mmu_1603_03048_iter0 = 0.1057    ## GeV, muon mass
#
cosThetaC_1603_03048_iter0 = 0.9743   ## Cos(Theta_C)
GFermi_1603_03048_iter0    = 1.166e-5 ## GeV^-2
mu_p_1603_03048_iter0      = 2.7928
mu_n_1603_03048_iter0      =-1.9130

params_1603_03048_iter0 = {}
params_1603_03048_iter0['MV'] = MV_1603_03048_iter0
params_1603_03048_iter0['ga'] = ga_1603_03048_iter0
params_1603_03048_iter0['tc'] = tc_1603_03048_iter0
params_1603_03048_iter0['t0'] = t0_1603_03048_iter0
params_1603_03048_iter0['xi'] = xi_1603_03048_iter0
params_1603_03048_iter0['MN'] = MN_1603_03048_iter0
params_1603_03048_iter0['Mpi'] = Mpi_1603_03048_iter0
params_1603_03048_iter0['me'] = me_1603_03048_iter0
params_1603_03048_iter0['mmu'] = mmu_1603_03048_iter0
params_1603_03048_iter0['GF'] = GFermi_1603_03048_iter0
params_1603_03048_iter0['cosThetaC'] = cosThetaC_1603_03048_iter0
params_1603_03048_iter0['mu_p'] = mu_p_1603_03048_iter0
params_1603_03048_iter0['mu_n'] = mu_n_1603_03048_iter0

params_iter0 = dict( params_1603_03048_iter0)

