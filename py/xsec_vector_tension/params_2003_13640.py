import numpy as np
import gvar as gv

## specific to 2003.13640
## used to infer intercept of GMx(0)
mu_p_2003_13640 = 2.7928473508
mu_n_2003_13640 =-1.91304272
#
Mpi_2003_13640  = 0.13957 ## GeV
Mpi2_2003_13640 = Mpi_2003_13640 *Mpi_2003_13640 ## GeV^2
#
tcV_2003_13640 = 4* Mpi2_2003_13640 ## GeV^2
tcS_2003_13640 = 9* Mpi2_2003_13640 ## GeV^2
t0V_2003_13640 =-0.21 ## GeV^2
t0S_2003_13640 =-0.28 ## GeV^2

## taken from 1603.03048
Mp_1603_03048  = 0.938272  ## GeV, proton mass
Mn_1603_03048  = 0.939565  ## GeV, neutron mass
MN_1603_03048  = (Mp_1603_03048 +Mn_1603_03048) /2 ## GeV, appx nucleon mass
me_1603_03048  = 0.000511  ## GeV, electron mass
mmu_1603_03048 = 0.1057    ## GeV, muon mass
#
cosThetaC_1603_03048 = 0.9743   ## Cos(Theta_C)
GFermi_1603_03048    = 1.166e-5 ## GeV^-2

## assumed from masses in 1603.03048
GEp0_2003_13640 = 1.
GEn0_2003_13640 = 0.
GMp0_2003_13640 = mu_p_2003_13640
GMn0_2003_13640 = mu_n_2003_13640 *(Mn_1603_03048 /Mp_1603_03048)

params_2003_13640 = {}
## 2003.13640 parameters
params_2003_13640['GEp(0)'] = GEp0_2003_13640
params_2003_13640['GMp(0)'] = GMp0_2003_13640
params_2003_13640['GEn(0)'] = GEn0_2003_13640
params_2003_13640['GMn(0)'] = GMn0_2003_13640
params_2003_13640['tc_V'] = tcV_2003_13640
params_2003_13640['tc_S'] = tcS_2003_13640
params_2003_13640['t0_V'] = t0V_2003_13640
params_2003_13640['t0_S'] = t0S_2003_13640
params_2003_13640['Mpi'] = Mpi_2003_13640
params_2003_13640['mu_p'] = mu_p_2003_13640
params_2003_13640['mu_n'] = mu_n_2003_13640
#
## 1603.03048 parameters
params_2003_13640['Mp'] = Mp_1603_03048
params_2003_13640['Mn'] = Mn_1603_03048
params_2003_13640['MN'] = MN_1603_03048
params_2003_13640['me'] = me_1603_03048
params_2003_13640['mmu'] = mmu_1603_03048
params_2003_13640['GF'] = GFermi_1603_03048
params_2003_13640['cosThetaC'] = cosThetaC_1603_03048

## get fit parameters and covariances
#

with open( 'data_2003_13640/GEn_coefs_kmax8.dat', 'r') as f: rd = f.read()
GEn_ak8_mean = np.array([ float( x) for x in rd.split() ])

with open( 'data_2003_13640/GEn_coefs_kmax9.dat', 'r') as f: rd = f.read()
GEn_ak9_mean = np.array([ float( x) for x in rd.split() ])

GEn_ak8_trunc = GEn_ak9_mean[:-1] -GEn_ak8_mean

with open( 'data_2003_13640/GEn_cov.dat', 'r') as f: rd = f.read()
GEn_ak8_cov_stat = np.array([[ float( y) for y in x.split() ] for x in rd.split('\n') ])
GEn_ak8_cov_syst = np.outer( GEn_ak8_trunc, GEn_ak8_trunc)

## GMn gets scaled by GMn(0)
with open( 'data_2003_13640/GMn_coefs_kmax8.dat', 'r') as f: rd = f.read()
GMn_ak8_mean = np.array([ float( x) /GMn0_2003_13640 for x in rd.split() ])

with open( 'data_2003_13640/GMn_coefs_kmax9.dat', 'r') as f: rd = f.read()
GMn_ak9_mean = np.array([ float( x) /GMn0_2003_13640 for x in rd.split() ])

GMn_ak8_trunc = GMn_ak9_mean[:-1] -GMn_ak8_mean

with open( 'data_2003_13640/GMn_cov.dat', 'r') as f: rd = f.read()
GMn2 = GMn0_2003_13640 *GMn0_2003_13640 
GMn_ak8_cov_stat = np.array([[ float( y) /GMn2 for y in x.split() ] for x in rd.split('\n') ])
GMn_ak8_cov_syst = np.outer( GMn_ak8_trunc, GMn_ak8_trunc)

## GEp and GMp are fit together
with open( 'data_2003_13640/proton_coefs_kmax8.dat', 'r') as f: rd = f.read()
GEp_ak8_mean, GMp_ak8_mean = tuple([
  np.array([ float( y) for y in x.split() ]) for x in rd.split('\n') ])
GMp_ak8_mean /= GMp0_2003_13640 ## scale to correct norm

## these are exactly the same as kmax8
with open( 'data_2003_13640/proton_coefs_kmax9.dat', 'r') as f: rd = f.read()
GEp_ak9_mean, GMp_ak9_mean = tuple([
  np.array([ float( y) for y in x.split() ]) for x in rd.split('\n') ])
GMp_ak9_mean /= GMp0_2003_13640 ## scale to correct norm

proton_ak8_mean = np.concatenate(( GEp_ak8_mean, GMp_ak8_mean))

## data files are not actually different, just set trunc to 0
#GEp_ak8_trunc = GEp_ak9_mean[:-1] -GEp_ak8_mean
#GMp_ak8_trunc = GMp_ak9_mean[:-1] -GMp_ak8_mean
GEp_ak8_trunc = 0 *GEp_ak8_mean
GMp_ak8_trunc = 0 *GMp_ak8_mean
proton_ak8_trunc = np.concatenate(( GEp_ak8_trunc, GMp_ak8_trunc))

## need to scale magnetic portion of covariance
proton_scale = np.diag( np.concatenate(( [1,1,1,1], np.array([1,1,1,1]) /GMp0_2003_13640 )))
with open( 'data_2003_13640/proton_cov.dat', 'r') as f: rd = f.read()
proton_ak8_cov_stat = np.array([[ float( y) for y in x.split() ] for x in rd.split('\n') ])
proton_ak8_cov_stat = proton_scale.dot( proton_ak8_cov_stat).dot( proton_scale)
proton_ak8_cov_syst = np.outer( proton_ak8_trunc, proton_ak8_trunc)

## all isospin-decomposed components are fit together
with open( 'data_2003_13640/iso1_coefs_kmax8.dat', 'r') as f: rd = f.read()
GES_ak8_mean, GMS_ak8_mean, GEV_ak8_mean, GMV_ak8_mean = tuple([
  np.array([ float( y) for y in x.split() ]) for x in rd.split('\n') ])
GMS_ak8_mean /= (GMp0_2003_13640 +GMn0_2003_13640)
GMV_ak8_mean /= (GMp0_2003_13640 -GMn0_2003_13640)

with open( 'data_2003_13640/iso1_coefs_kmax9.dat', 'r') as f: rd = f.read()
GES_ak9_mean, GMS_ak9_mean, GEV_ak9_mean, GMV_ak9_mean = tuple([
  np.array([ float( y) for y in x.split() ]) for x in rd.split('\n') ])
GMS_ak9_mean /= (GMp0_2003_13640 +GMn0_2003_13640)
GMV_ak9_mean /= (GMp0_2003_13640 -GMn0_2003_13640)

iso1_ak8_mean = np.concatenate(( GES_ak8_mean, GMS_ak8_mean, GEV_ak8_mean, GMV_ak8_mean))

GES_ak8_trunc = GES_ak9_mean[:-1] -GES_ak8_mean
GMS_ak8_trunc = GMS_ak9_mean[:-1] -GMS_ak8_mean
GEV_ak8_trunc = GEV_ak9_mean[:-1] -GEV_ak8_mean
GMV_ak8_trunc = GMV_ak9_mean[:-1] -GMV_ak8_mean
iso1_ak8_trunc = np.concatenate(( GES_ak8_trunc, GMS_ak8_trunc, GEV_ak8_trunc, GMV_ak8_trunc))

## need to scale magnetic portion of covariance
iso1_scale = np.diag( np.concatenate((
 [1,1,1,1],
 np.array([1,1,1,1]) /(GMp0_2003_13640 +GMn0_2003_13640), ## mag isoscalar
 [1,1,1,1],
 np.array([1,1,1,1]) /(GMp0_2003_13640 -GMn0_2003_13640)  ## mag isovector
 )))
with open( 'data_2003_13640/iso1_cov.dat', 'r') as f: rd = f.read()
iso1_ak8_cov_stat = np.array([[ float( y) for y in x.split() ] for x in rd.split('\n') ])
iso1_ak8_cov_stat = iso1_scale.dot( iso1_ak8_cov_stat).dot( iso1_scale)
iso1_ak8_cov_syst = np.outer( iso1_ak8_trunc, iso1_ak8_trunc)

GEn_ak8_stat = gv.gvar( GEn_ak8_mean, GEn_ak8_cov_stat)
GEn_ak8_syst = gv.gvar( GEn_ak8_mean, GEn_ak8_cov_syst)
GEn_ak8      = gv.gvar( GEn_ak8_mean, GEn_ak8_cov_stat +GEn_ak8_cov_syst)

GMn_ak8_stat = gv.gvar( GMn_ak8_mean, GMn_ak8_cov_stat)
GMn_ak8_syst = gv.gvar( GMn_ak8_mean, GMn_ak8_cov_syst)
GMn_ak8      = gv.gvar( GMn_ak8_mean, GMn_ak8_cov_stat +GMn_ak8_cov_syst)

proton_ak8_stat = gv.gvar( proton_ak8_mean, proton_ak8_cov_stat)
proton_ak8_syst = gv.gvar( proton_ak8_mean, proton_ak8_cov_syst)
proton_ak8      = gv.gvar( proton_ak8_mean, proton_ak8_cov_stat +proton_ak8_cov_syst)
GEp_ak8_stat = proton_ak8_stat[:4]
GEp_ak8_syst = proton_ak8_syst[:4]
GEp_ak8      = proton_ak8[:4]
GMp_ak8_stat = proton_ak8_stat[4:]
GMp_ak8_syst = proton_ak8_syst[4:]
GMp_ak8      = proton_ak8[4:]

iso1_ak8_stat = gv.gvar( iso1_ak8_mean, iso1_ak8_cov_stat)
iso1_ak8_syst = gv.gvar( iso1_ak8_mean, iso1_ak8_cov_syst)
iso1_ak8      = gv.gvar( iso1_ak8_mean, iso1_ak8_cov_stat +iso1_ak8_cov_syst)
GES_ak8_stat = iso1_ak8_stat[:4]
GES_ak8_syst = iso1_ak8_syst[:4]
GES_ak8      = iso1_ak8[:4]
GMS_ak8_stat = iso1_ak8_stat[4:8]
GMS_ak8_syst = iso1_ak8_syst[4:8]
GMS_ak8      = iso1_ak8[4:8]
GEV_ak8_stat = iso1_ak8_stat[8:12]
GEV_ak8_syst = iso1_ak8_syst[8:12]
GEV_ak8      = iso1_ak8[8:12]
GMV_ak8_stat = iso1_ak8_stat[12:]
GMV_ak8_syst = iso1_ak8_syst[12:]
GMV_ak8      = iso1_ak8[12:]

if 0:
  print( 'GEp', GEp_ak8_stat)
  print( 'GMp', GMp_ak8_stat)
  print( 'GEn', GEn_ak8_stat)
  print( 'GMn', GMn_ak8_stat)
  print( 'GEV', GEV_ak8_stat)
  print( 'GMV', GMV_ak8_stat)
  print( 'GES', GES_ak8_stat)
  print( 'GMS', GMS_ak8_stat)

