import numpy as np
import gvar as gv

from form_factors import *
from params_generic import *
from params_1603_03048 import *
from params_2003_13640 import *

from plot_params import *
import matplotlib.pyplot as plt
import matplotlib.lines   as mlines
import matplotlib.patches as mpatches

## creates copycat of 2003.13640 Fig 4 upper right (proton magnetic)
#

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
#else:
#  mpl.rcParams.update({
#    'text.usetex': True,
#  })

MN  = params_2003_13640[ 'MN']
mmu = params_1603_03048[ 'mmu']

## vector form factors from 2003.13640
## jacobian is not used
GMp0 = params_2003_13640[ 'GMp(0)']
GMn0 = params_2003_13640[ 'GMn(0)']
tcv = params_2003_13640[ 'tc_V']
tcs = params_2003_13640[ 'tc_S']
t0v = params_2003_13640[ 't0_V']
t0s = params_2003_13640[ 't0_S']
BHLT_GMp_Q2, _ = define_zexp_sum4( 4, tc=tcv, t0=t0v, g0=1, norm=GMp0, **params_2003_13640)
BHLT_GMV_Q2, _ = define_zexp_sum4( 4, tc=tcv, t0=t0v, g0=1, norm=GMp0-GMn0, **params_2003_13640)
BHLT_GMS_Q2, _ = define_zexp_sum4( 4, tc=tcs, t0=t0s, g0=1, norm=GMp0+GMn0, **params_2003_13640)

## BBBA05 vector form factors
BBBA_GMp_Q2 = define_GMp_BBA05( **params_1603_03048)

## reference dipole
FD = define_dipole_fixed( 1, 0.84, **params_1603_03048)

Q2_min, Q2_max = 0., 1.0
Ev_min, Ev_max = mmu+1e-3, 4.0
Q2_range = np.arange( Q2_min, Q2_max+1e-5, 1e-3)
Ev_range = np.arange( Ev_min, Ev_max, 1e-4)

## compute form factor curves
BBBA_GMp_vals = np.array([ BBBA_GMp_Q2( Q2) for Q2 in Q2_range ])
BHLT_GMp_vals = np.array([ BHLT_GMp_Q2( Q2, GMp_ak8_stat) for Q2 in Q2_range ])
BHLT_GMV_vals = np.array([ BHLT_GMV_Q2( Q2, GMV_ak8_stat) for Q2 in Q2_range ])
BHLT_GMS_vals = np.array([ BHLT_GMS_Q2( Q2, GMS_ak8_stat) for Q2 in Q2_range ])
BHLT_GMi_vals = (BHLT_GMV_vals +BHLT_GMS_vals)/2.

mu_p, mu_n = mu_p_2003_13640, mu_n_2003_13640
dipl_vals = np.array([ FD( Q2) for Q2 in Q2_range ])
dipl_vals_p = dipl_vals *mu_p

handles, labels = [], []
f0, ax01 = plt.subplots()
plot_param_vals = {
  'left':   plot_axes_default[0],
  'bottom': plot_axes_default[1],
  'right':  plot_axes_default[2],
  'top':    plot_axes_default[3],
  'wspace': 0.300,
  'hspace': 0.290,
}
f0.set_size_inches( *figure_size_default)
f0.subplots_adjust( **plot_param_vals)

## central curves
BBBA_cv = ax01.plot( Q2_range, gv.mean( BBBA_GMp_vals) /dipl_vals_p, color='blue', linestyle='-')
BHLT_cv = ax01.plot( Q2_range, gv.mean( BHLT_GMp_vals) /dipl_vals_p, color='black', linestyle='-.')
#BHLT_cvi= ax01.plot( Q2_range, gv.mean( BHLT_GMi_vals) /dipl_vals_p, color='violet', linestyle=':')

## uncertainty bands
color, alpha, linestyle = 'blue', 0.1, '-'
vals = BBBA_GMp_vals
vals_lo = gv.mean( vals) -gv.sdev( vals); vals_hi = gv.mean( vals) +gv.sdev( vals)
ax01.fill_between( Q2_range,
  vals_lo /dipl_vals_p, vals_hi /dipl_vals_p, color=color, alpha=alpha, linestyle=linestyle)
BBBA_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BBBA_cv[0], BBBA_pt))
labels.append( r'BBBA05')

color, alpha, linestyle = 'black', 0.1, '-'
vals = BHLT_GMp_vals
vals_lo = gv.mean( vals) -gv.sdev( vals); vals_hi = gv.mean( vals) +gv.sdev( vals)
ax01.fill_between( Q2_range,
  vals_lo /dipl_vals_p, vals_hi /dipl_vals_p, color=color, alpha=alpha, linestyle=linestyle)
BHLT_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
handles.append(( BHLT_cv[0], BHLT_pt))
if do_pgf:
  labels.append( r"Borah $\textit{et al.}$")
else:
  labels.append( r"Borah $\it{et\;al.}$")

#color, alpha, linestyle = 'violet', 0.1, '-'
#vals = BHLT_GMi_vals
#vals_lo = gv.mean( vals) -gv.sdev( vals); vals_hi = gv.mean( vals) +gv.sdev( vals)
#ax01.fill_between( Q2_range,
#  vals_lo /dipl_vals_p, vals_hi /dipl_vals_p, color=color, alpha=alpha, linestyle=linestyle)
#BHLT_pt = mpatches.Patch( color=color, alpha=alpha, linestyle=linestyle)
#handles.append(( BHLT_cv[0], BHLT_pt))
#if do_pgf:
#  labels.append( r"Borah $\textit{et al.}$")
#else:
#  labels.append( r"Borah $\it{et\;al.}$ (iso fit)")

## axes
plt.sca( ax01)
plt.xlim([ Q2_min, Q2_max ])
plt.ylim([ 0.970, 1.105 ])
plt.xlabel( r'$Q^2/$GeV$^{2}$', fontsize=14)
plt.ylabel( r'$G_{M}^{p}/(\mu_{p}G_{D})$', fontsize=14)
plt.xticks( fontsize=14)
plt.yticks( np.arange( 0.98, 1.11, 0.02), fontsize=14)
ax01.legend( handles, labels, loc='upper left', fontsize=12)

if do_pgf:
  fout = 'out_plot/proton_magnetic.pgf'
  print( 'output to file {}'.format( fout))
  plt.savefig( fout, transparent=True)
else:
  plt.show()

