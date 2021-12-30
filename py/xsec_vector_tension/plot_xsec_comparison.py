import numpy as np
import gvar as gv
import scipy.integrate as integrate
import time

from form_factors import *
from qe_xsec import Q2bd, define_A, define_B, define_C, define_dsigma_dQ2, get_prefactor
from params_generic import *
from params_1603_03048 import *
from params_2003_13640 import *

from plot_params import *
import matplotlib.pyplot  as plt
import matplotlib.lines   as mlines
import matplotlib.patches as mpatches

do_log = 1
do_pgf = 1

if do_pgf:
  mpl.use("pgf")
  mpl.rcParams.update({
    'pgf.texsystem': 'pdflatex',
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.preamble': '\n'.join([
         r'\usepackage[utf8]{inputenc}',
         r'\DeclareUnicodeCharacter{2212}{-}',
         r'\usepackage{amsmath,amssymb}',
         r'\usepackage{bm}',
         r'\usepackage{lmodern}',
         #r'\usepackage{helvet}',
         r'\usepackage{sansmath}',
         r'\sansmath',
       ])
  })

MN  = params_2003_13640[ 'MN']
mmu = params_1603_03048[ 'mmu']

## vector form factors from 2003.13640
## jacobian is not used
GEp0 = params_2003_13640[ 'GEp(0)']
GMp0 = params_2003_13640[ 'GMp(0)']
GEn0 = params_2003_13640[ 'GEn(0)']
GMn0 = params_2003_13640[ 'GMn(0)']
tc = params_2003_13640[ 'tc_V']
t0 = params_2003_13640[ 't0_V']
BHLT_GEV_Q2, _ = define_zexp_sum4( 4, tc=tc, t0=t0, g0=1, norm=GEp0-GEn0, **params_2003_13640)
BHLT_GMV_Q2, _ = define_zexp_sum4( 4, tc=tc, t0=t0, g0=1, norm=GMp0-GMn0, **params_2003_13640)
## need to build the coefficients into a lambda
## gvar version
BHLT_F1_Q2 = define_F1_Generic(
  lambda Q2: BHLT_GEV_Q2( Q2, GEV_ak8_stat),
  lambda Q2: BHLT_GMV_Q2( Q2, GMV_ak8_stat),
  **params_2003_13640)
BHLT_F2_Q2 = define_F2_Generic(
  lambda Q2: BHLT_GEV_Q2( Q2, GEV_ak8_stat),
  lambda Q2: BHLT_GMV_Q2( Q2, GMV_ak8_stat),
  **params_2003_13640)
## mean only
BHLT_F1_Q2_mean = lambda Q2: gv.mean( BHLT_F1_Q2( Q2))
BHLT_F2_Q2_mean = lambda Q2: gv.mean( BHLT_F2_Q2( Q2))

## BBBA05 vector form factors
BBBA_GEV_Q2 = define_GEV_BBA05( **params_1603_03048)
BBBA_GMV_Q2 = define_GMV_BBA05( **params_1603_03048)
BBBA_F1_Q2 = define_F1_Generic( BBBA_GEV_Q2, BBBA_GMV_Q2, **params_1603_03048)
BBBA_F2_Q2 = define_F2_Generic( BBBA_GEV_Q2, BBBA_GMV_Q2, **params_1603_03048)
BBBA_F1_Q2_mean = lambda Q2: gv.mean( BBBA_F1_Q2( Q2))
BBBA_F2_Q2_mean = lambda Q2: gv.mean( BBBA_F2_Q2( Q2))

## axial form factors from 1603.03048
ga = params_1603_03048[ 'ga']
FA_Q2, grad_FA_Q2 = define_zexp_sum4( 4, g0=ga, norm=1, **params_1603_03048)
FP_Q2, grad_FP_Q2 = define_FP_PCAC( FA_Q2, grad_FA_Q2, **params_1603_03048)
FA_Q2_mean = lambda Q2, ak: gv.mean( FA_Q2( Q2, ak))
FP_Q2_mean = lambda Q2, ak: gv.mean( FP_Q2( Q2, ak))

## axial form factors from LQCD
FA_Q2_LQCD, grad_FA_Q2_LQCD = define_zexp_sum4_fit_g0( 4, norm=1, **params_1603_03048)
FP_Q2_LQCD, grad_FP_Q2_LQCD = define_FP_PCAC( FA_Q2_LQCD, grad_FA_Q2_LQCD, **params_1603_03048)

### FA uncertainty only
#BBBA_A_Q2_FA, _ = define_A( BBBA_F1_Q2_mean, BBBA_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BBBA_B_Q2_FA, _ = define_B( BBBA_F1_Q2_mean, BBBA_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BBBA_C_Q2_FA, _ = define_C( BBBA_F1_Q2_mean, BBBA_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#
#BHLT_A_Q2_FA, _ = define_A( BHLT_F1_Q2_mean, BHLT_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BHLT_B_Q2_FA, _ = define_B( BHLT_F1_Q2_mean, BHLT_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BHLT_C_Q2_FA, _ = define_C( BHLT_F1_Q2_mean, BHLT_F2_Q2_mean, FA_Q2, FP_Q2,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#
### FV uncertainty only
#BBBA_A_Q2_FV, _ = define_A( BBBA_F1_Q2, BBBA_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BBBA_B_Q2_FV, _ = define_B( BBBA_F1_Q2, BBBA_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BBBA_C_Q2_FV, _ = define_C( BBBA_F1_Q2, BBBA_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#
#BHLT_A_Q2_FV, _ = define_A( BHLT_F1_Q2, BHLT_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BHLT_B_Q2_FV, _ = define_B( BHLT_F1_Q2, BHLT_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)
#BHLT_C_Q2_FV, _ = define_C( BHLT_F1_Q2, BHLT_F2_Q2, FA_Q2_mean, FP_Q2_mean,
#  grad_FA_Q2, grad_FP_Q2, **params_1603_03048)

## build lambda fns for mean, upper, lower
def fn_wrapper( fn):
  fn_mean = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak))
  fn_hi   = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak)) +gv.sdev( fn( Ev, Q2, ak))
  fn_lo   = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak)) -gv.sdev( fn( Ev, Q2, ak))
  return (fn_mean, fn_hi, fn_lo)

## the relevant dsigma_dQ2 functions
units='cm2'
do_antineutrino=0
BBBA_dsigma_dQ2_FV, _ = define_dsigma_dQ2( BBBA_F1_Q2, BBBA_F2_Q2, FA_Q2_mean, FP_Q2_mean,
  grad_FA_Q2, grad_FP_Q2, units='no_prefactor', do_antineutrino=do_antineutrino,
  **params_1603_03048)
BHLT_dsigma_dQ2_FV, _ = define_dsigma_dQ2( BHLT_F1_Q2, BHLT_F2_Q2, FA_Q2_mean, FP_Q2_mean,
  grad_FA_Q2, grad_FP_Q2, units='no_prefactor', do_antineutrino=do_antineutrino,
  **params_1603_03048)
BBBA_dsigma_dQ2_FA, _ = define_dsigma_dQ2( BBBA_F1_Q2_mean, BBBA_F2_Q2_mean, FA_Q2, FP_Q2,
  grad_FA_Q2, grad_FP_Q2, units='no_prefactor', do_antineutrino=do_antineutrino,
  **params_1603_03048)
BHLT_dsigma_dQ2_FA, _ = define_dsigma_dQ2( BHLT_F1_Q2_mean, BHLT_F2_Q2_mean, FA_Q2, FP_Q2,
  grad_FA_Q2, grad_FP_Q2, units='no_prefactor', do_antineutrino=do_antineutrino,
  **params_1603_03048)
BHLT_dsigma_dQ2_FA_LQCD, _ = define_dsigma_dQ2(
  BHLT_F1_Q2_mean, BHLT_F2_Q2_mean, FA_Q2_LQCD, FP_Q2_LQCD, grad_FA_Q2_LQCD, grad_FP_Q2_LQCD,
  units='no_prefactor', do_antineutrino=do_antineutrino, **params_1603_03048)

## build lambda fns for mean, upper, lower
def fn_wrapper( fn):
  fn_mean = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak))
  fn_lo   = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak)) -gv.sdev( fn( Ev, Q2, ak))
  fn_hi   = lambda Ev, Q2, ak: gv.mean( fn( Ev, Q2, ak)) +gv.sdev( fn( Ev, Q2, ak))
  return (fn_mean, fn_lo, fn_hi)

## slow but gets the job done
def xsec_integration_wrapper( dsigma_dQ2, **params):
  MN  = params[ 'MN' ]
  mmu = params[ 'mmu']
  prefactor = get_prefactor( **params)
  dsigma_dQ2_mean, dsigma_dQ2_hi, dsigma_dQ2_lo = fn_wrapper( dsigma_dQ2)
  def sigma_tot_mean( Enu, axial_param):
    Q2low, Q2high = Q2bd( Enu, MN, mmu)
    res = integrate.quad(
      lambda Q2: dsigma_dQ2_mean( Enu, Q2, axial_param),
      Q2low, Q2high, limit=100)[0]
    return prefactor *res
  def sigma_tot_lo( Enu, axial_param):
    Q2low, Q2high = Q2bd( Enu, MN, mmu)
    res = integrate.quad(
      lambda Q2: dsigma_dQ2_lo( Enu, Q2, axial_param),
      Q2low, Q2high, limit=100)[0]
    return prefactor *res
  def sigma_tot_hi( Enu, axial_param):
    Q2low, Q2high = Q2bd( Enu, MN, mmu)
    res = integrate.quad(
      lambda Q2: dsigma_dQ2_hi( Enu, Q2, axial_param),
      Q2low, Q2high, limit=100)[0]
    return prefactor *res
  return sigma_tot_mean, sigma_tot_lo, sigma_tot_hi

## mean, hi, lo functions
units = 'cm2'
BBBA_xsec_FV_mean, BBBA_xsec_FV_lo, BBBA_xsec_FV_hi = xsec_integration_wrapper(
  BBBA_dsigma_dQ2_FV, units=units, **params_1603_03048)
BHLT_xsec_FV_mean, BHLT_xsec_FV_lo, BHLT_xsec_FV_hi = xsec_integration_wrapper(
  BHLT_dsigma_dQ2_FV, units=units, **params_1603_03048)
BBBA_xsec_FA_mean, BBBA_xsec_FA_lo, BBBA_xsec_FA_hi = xsec_integration_wrapper(
  BBBA_dsigma_dQ2_FA, units=units, **params_1603_03048)
BHLT_xsec_FA_mean, BHLT_xsec_FA_lo, BHLT_xsec_FA_hi = xsec_integration_wrapper(
  BHLT_dsigma_dQ2_FA, units=units, **params_1603_03048)
BHLT_xsec_FA_LQCD_mean, BHLT_xsec_FA_LQCD_lo, BHLT_xsec_FA_LQCD_hi = xsec_integration_wrapper(
  BHLT_dsigma_dQ2_FA_LQCD, units=units, **params_1603_03048)

Q2_min, Q2_max = 0., 1.0
Q2_range = np.arange( Q2_min, Q2_max+1e-5, 1e-3)
#Ev_min, Ev_max = mmu+1e-3, 4.0
if do_log:
  Ev_min, Ev_max = 0.3, 10.
  Ev_range = np.exp( np.arange( np.log( Ev_min), np.log( Ev_max)+1e-5, 1e-1)) ## log scale
else:
  Ev_min, Ev_max = 0.3, 4.0
  Ev_range = np.arange( Ev_min, Ev_max+1e-5, 1e-1) ## linear scale

## compute form factor curves
print( 'computing xsecs')
ak = ak_1603_03048
startTime = taskTime = time.time()
print( ' -- BBBA FV     ', end='')
BBBA_FV_mean = np.array([ BBBA_xsec_FV_mean( Ev, ak) for Ev in Ev_range ])
BBBA_FV_lo   = np.array([ BBBA_xsec_FV_lo(   Ev, ak) for Ev in Ev_range ])
BBBA_FV_hi   = np.array([ BBBA_xsec_FV_hi(   Ev, ak) for Ev in Ev_range ])
print( ' done, time {:10.5f}s'.format( time.time() -taskTime))
taskTime = time.time()
print( ' -- BHLT FV     ', end='')
BHLT_mean    = np.array([ BHLT_xsec_FV_mean( Ev, ak) for Ev in Ev_range ])
BHLT_FV_lo   = np.array([ BHLT_xsec_FV_lo(   Ev, ak) for Ev in Ev_range ])
BHLT_FV_hi   = np.array([ BHLT_xsec_FV_hi(   Ev, ak) for Ev in Ev_range ])
print( ' done, time {:10.5f}s'.format( time.time() -taskTime))
## using 1603.03048 FA
taskTime = time.time()
print( ' -- BHLT FA D2  ', end='')
BHLT_D2_lo   = np.array([ BHLT_xsec_FA_lo(   Ev, ak) for Ev in Ev_range ])
BHLT_D2_hi   = np.array([ BHLT_xsec_FA_hi(   Ev, ak) for Ev in Ev_range ])
print( ' done, time {:10.5f}s'.format( time.time() -taskTime))

## using LQCD FA
ak_mean = np.array([ 0.914, -1.931, -0.63, 4.4, -2.2 ])
ak_cov = np.array([
  [ 1.07656068e-04, -5.67981167e-04, -3.20528306e-04,  1.48484537e-02, -1.32944995e-02],
  [-5.67981167e-04,  2.89580422e-03,  4.66404291e-04, -6.58295384e-02,  7.51052601e-02],
  [-3.20528306e-04,  4.66404291e-04,  8.84891256e-02, -3.57106039e-01, -9.12119801e-01],
  [ 1.48484537e-02, -6.58295384e-02, -3.57106039e-01,  2.75581331e+00,  1.80702909e+00],
  [-1.32944995e-02,  7.51052601e-02, -9.12119801e-01,  1.80702909e+00,  1.30170672e+01]])
ak = gv.gvar( ak_mean, ak_cov)
taskTime = time.time()
print( ' -- BHLT FA LQCD', end='')
BHLT_LQCD_mean = np.array([ BHLT_xsec_FA_LQCD_mean( Ev, ak) for Ev in Ev_range ])
BHLT_LQCD_lo   = np.array([ BHLT_xsec_FA_LQCD_lo(   Ev, ak) for Ev in Ev_range ])
BHLT_LQCD_hi   = np.array([ BHLT_xsec_FA_LQCD_hi(   Ev, ak) for Ev in Ev_range ])
print( ' done, time {:10.5f}s'.format( time.time() -taskTime))
#
print( 'total time {:10.5f}s'.format( time.time() -startTime))

plot_param_vals = {
  'left':   plot_axes_default[0],
  'bottom': plot_axes_default[1],
  'right':  plot_axes_default[2],
  'top':    plot_axes_default[3],
  'wspace': 0.300,
  'hspace': 0.290,
}

handles, labels = [], []
f0, ax = plt.subplots( 1)
f0.set_size_inches( *figure_size_default)
f0.subplots_adjust( **plot_param_vals)

## central curves
scale = 1e38
BBBA_FV_cv   = ax.plot( Ev_range, scale *BBBA_FV_mean, color='blue',  linestyle='-')
BHLT_FV_cv   = ax.plot( Ev_range, scale *BHLT_mean,    color='black', linestyle='-.')
BHLT_LQCD_cv = ax.plot( Ev_range, scale *BHLT_LQCD_mean, color='red', linestyle=':')

## uncertainty bands
color, alpha, linestyle = 'blue', 0.1, '-'
vals_lo = scale *BBBA_FV_lo; vals_hi = scale *BBBA_FV_hi
ax.fill_between( Ev_range,
  vals_lo, vals_hi, color=color, alpha=alpha, linestyle=linestyle)
BBBA_FV_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BBBA_FV_cv[0], BBBA_FV_pt))
labels.append( 'BBBA05')

color, alpha, linestyle = 'black', 0.2, '-'
vals_lo = scale *BHLT_FV_lo; vals_hi = scale *BHLT_FV_hi
ax.fill_between( Ev_range,
  vals_lo, vals_hi, color=color, alpha=alpha, linestyle=linestyle)
BHLT_FV_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BHLT_FV_cv[0], BHLT_FV_pt))
labels.append( r'$z$ exp, vector')

color, alpha, linestyle = 'green', 0.1, '-'
vals_lo = scale *BHLT_D2_lo; vals_hi = scale *BHLT_D2_hi
ax.fill_between( Ev_range,
  vals_lo, vals_hi, color=color, alpha=alpha, linestyle=linestyle)
BHLT_D2_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BHLT_FV_cv[0], BHLT_D2_pt))
labels.append( r'$z$ exp, $D_{2}$ axial')

color, alpha, linestyle = 'red', 0.1, '-'
vals_lo = scale *BHLT_LQCD_lo; vals_hi = scale *BHLT_LQCD_hi
ax.fill_between( Ev_range,
  vals_lo, vals_hi, color=color, alpha=alpha, linestyle=linestyle)
BHLT_LQCD_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BHLT_LQCD_cv[0], BHLT_LQCD_pt))
labels.append( r'$z$ exp, LQCD axial')

## axes
if do_log: plt.xscale( 'log')

plt.sca( ax)
plt.xlim([ Ev_min, Ev_max ])
plt.ylim([ 0.4, 1.5 ])
if do_log:
  xticks = [Ev_min, 0.4, 1.0, 4.0, 10.]
  plt.xticks( xticks, xticks, fontsize=14)
else:
  plt.xticks( np.concatenate(( [Ev_min], np.arange( 0.5, Ev_max+1e-3, 0.5))), fontsize=14)

plt.yticks( np.arange( 0.3, 1.55, 0.2), fontsize=14)
plt.xlabel( r'$E_{\nu}/$GeV', fontsize=14)
plt.ylabel( r'$\sigma(E_{\nu})/10^{-38}$cm$^{2}$', fontsize=14)
plt.xticks( fontsize=14)
ax.legend( handles, labels, loc='lower right', fontsize=12)

if do_pgf:
  fout = 'out_plot/xsec_comparison.pgf'
  print( 'output to file {}'.format( fout))
  plt.savefig( fout, transparent=True)
else:
  plt.show()

