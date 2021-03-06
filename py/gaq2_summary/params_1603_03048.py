import numpy as np
import gvar as gv

## specific to 1603.03048
#sign_convention = -1       ## stupid choice
sign_convention =  1       ## less stupid choice
ga_1603_03048  = sign_convention *1.2723 ## axial charge
tc_1603_03048  = 9*.14*.14 ## GeV^2
t0_1603_03048  =-0.28      ## GeV^2
xi_1603_03048  = 3.7058    ## mu_p -mu_n -1, magnetic couplings
#
MN_1603_03048  = 0.9389    ## GeV, appx nucleon mass
Mpi_1603_03048 = 0.14      ## GeV, appx pion mass
#
Mp_1603_03048  = 0.938272  ## GeV, proton mass
Mn_1603_03048  = 0.939565  ## GeV, neutron mass
me_1603_03048  = 0.000511  ## GeV, electron mass
mmu_1603_03048 = 0.1057    ## GeV, muon mass
#
cosThetaC_1603_03048 = 0.9743   ## Cos(Theta_C)
GFermi_1603_03048    = 1.166e-5 ## GeV^-2
mu_p_1603_03048      = 2.7928
mu_n_1603_03048      =-1.9130

params_1603_03048 = {}
params_1603_03048['ga'] = ga_1603_03048
params_1603_03048['tc'] = tc_1603_03048
params_1603_03048['t0'] = t0_1603_03048
params_1603_03048['xi'] = xi_1603_03048
params_1603_03048['MN'] = MN_1603_03048
params_1603_03048['Mpi'] = Mpi_1603_03048
params_1603_03048['me'] = me_1603_03048
params_1603_03048['mmu'] = mmu_1603_03048
params_1603_03048['GF'] = GFermi_1603_03048
params_1603_03048['cosThetaC'] = cosThetaC_1603_03048
params_1603_03048['mu_p'] = mu_p_1603_03048
params_1603_03048['mu_n'] = mu_n_1603_03048

params_default = dict( params_1603_03048)

## full covariance matrix, including norms, acceptance, deuterium
cov_1603_03048 = np.array([[
   0.01809905011134995,    0.015579993139534929,   0.014408195751901184,
  -0.0007906214717252839, -0.12187056644288428,    0.24740904853231693,
  -0.12676327453720843,   -0.05708159524186798,   -0.027601801847324477,
  -0.0028967713002595553],
 [ 0.015579993139534929,   0.01792647773882416,    0.014067937111362037,
  -0.0007578229372500094, -0.11881570021535108,    0.2414754130108787,
  -0.12494372792361105,   -0.03803093977244627,   -0.04728529419765055,
  -0.0028318702768684095],
 [ 0.014408195751901184,   0.014067937111362037,   0.020019212714923948,
  -0.0014464357322799461, -0.11227610425272477,    0.23293859641569598,
  -0.1208827747186863,    -0.03433396515383613,   -0.024173403912361124,
  -0.018057328378971806],
 [-0.000790621471725283,  -0.0007578229372500085, -0.0014464357322799453,
   0.015458177972177668,   0.04518361307550629,   -0.21564117642240807,
   0.20647022296382583,   -0.015092037578060108,  -0.014091826407040136,
  -0.0022258267269955525],
 [-0.12187056644288428,   -0.1188157002153511,    -0.11227610425272477,
   0.04518361307550629,    1.0809068918229854,    -2.3870160925804846,
   1.0386012828863096,     0.252819542477964,      0.16904929429420593,
   0.015809477057394012],
 [ 0.24740904853231693,    0.24147541301087866,    0.23293859641569595,
  -0.21564117642240804,   -2.387016092580484,      6.535675656310291,
  -4.765774927800392,     -0.37983056599870024,   -0.22529011653629077,
  -0.013800886748013712],
 [-0.12676327453720843,   -0.12494372792361108,   -0.1208827747186863,
   0.20647022296382583,    1.0386012828863096,    -4.7657749278003925,
   7.3983193293132965,     0.09936774238866958,    0.040387597715876744,
  -0.00514411106438873],
 [-0.05708159524186798,   -0.03803093977244626,   -0.03433396515383613,
  -0.015092037578060108,   0.25281954247796395,   -0.3798305659987003,
   0.09936774238866955,    0.8606351104747707,     0.08447880761077406,
   0.009691205837289925],
 [-0.027601801847324477,  -0.04728529419765054,   -0.02417340391236112,
  -0.014091826407040136,   0.16904929429420593,   -0.22529011653629075,
   0.04038759771587677,    0.08447880761077406,    0.9055895517172868,
   0.007402848652492262],
 [-0.0028967713002595553, -0.0028318702768684104, -0.018057328378971806,
  -0.0022258267269955516,  0.015809477057394012,  -0.013800886748013712,
  -0.00514411106438873,    0.009691205837289897,   0.007402848652492262,
   0.9847142458891733]])[3:7,3:7]

## coefficients only
val_1603_03048 = -sign_convention *np.array([
   2.2962961514010414, -0.5702353490765429,
  -3.7873956246696396,  2.312770215323479 ])

ak_1603_03048 = gv.gvar( val_1603_03048, cov_1603_03048)

