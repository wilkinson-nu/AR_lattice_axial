import numpy as np
import gvar as gv

from params_generic import *
from params_1603_03048 import *
from form_factors import *
from plot_params import *

import matplotlib.pyplot as plt

long_Q2 = 1
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
         r'\usepackage{sansmath}',
         r'\sansmath',
       ])
  })

def get_data( fn):
  data = []
  with open( fn, 'r') as f:
    rd = f.read()
  for line in rd.split('\n'):
    if line == '':
      continue
    if line[0] == '#':
      continue
    lsp = line.split()
    data.append([ float( lsp[0]), float( lsp[1]), float( lsp[2]) ])
  return np.array( data)

def get_rqcd( fn):
  data = []
  with open( fn, 'r') as f:
    rd = f.read()
  for line in rd.split('\n'):
    if line == '':
      continue
    if line[0] == '#':
      continue
    lsp = line.split()
    data.append([ float( lsp[0]), float( lsp[1]), float( lsp[3]) ])
  return np.array( data)

cls  = get_data( 'cls_gaq2.dat')
etm = get_data( 'etm_gaq2.dat')
mainz = get_data( 'mainz_gaq2.dat')
callat = get_data( 'callat_gaq2.dat')
pacs21 = get_data( 'pacs21_gaq2.dat')
pacs = get_data( 'pacs_data/pacs_gaq2.dat')
rqcd = get_rqcd( 'rqcd_gaq2_zexp4+3.dat')

nme = get_data( 'nme_gaq2_z2t00.5.dat')
## NME, optional error inflation using Q2 expansion
## suggested by Rajan, 02/06/2022 14:14 :
##   Use the P_2 Pade result in Eqn 55
##   Increase the error by a factor of 3 in the g_A and coeff of Q^2 term
##   ie, use 1.270(33) and 5.36(60)
#
if 1:
  Q2_range_nme = nme.T[0]
  gA = gv.gvar( 1.270, 0.033)
  b_list = gv.gvar([ 5.36, -0.22], [0.60, 0.81])
  MN = params_1603_03048[ "MN"]
  scaling_4MN = np.array([ np.power( 2. *MN, -2*n) for n in range( 1, 3) ])
  FA_nme_denom = define_BBA_generic( b_list *scaling_4MN, **params_1603_03048)
  FA_nme = (lambda Q2: gA *FA_nme_denom( Q2))
  FAQ2_nme_vals = np.array([ FA_nme( Q2) for Q2 in Q2_range_nme ])
  nme = np.array([ Q2_range_nme, gv.mean( FAQ2_nme_vals), gv.sdev( FAQ2_nme_vals) ]).T

FA_Q2_fn, _  = define_FA_zexp_sum4( 4, **params_1603_03048)

Q2_range = np.arange( 0.0, 2.02, 0.01)
FA_Q2 = np.array([ FA_Q2_fn( Q2, ak_1603_03048) for Q2 in Q2_range ])

ymean = gv.mean( FA_Q2)
ysdev = gv.sdev( FA_Q2)
yp = ymean+ysdev
ym = ymean-ysdev

default_size = plt.rcParams[ 'figure.figsize']
default_size[1] *= 1.005
plt.rcParams[ 'figure.figsize'] = default_size
plt.rcParams[ 'errorbar.capsize'] = 5

f0 = plt.figure()
f0.set_size_inches( *figure_size_default)
plt.subplots_adjust( *plot_axes_default)
plt.fill_between( Q2_range, ym, yp, color='red', alpha=0.3, zorder=-5)

lbc = 'CLS 17'
lbe = 'ETMC 20'
lbm = 'Mainz 21'
lbx = 'CalLat 21'
lbp = 'PACS 18 erratum'
lbp21= 'PACS 21'
lbn = 'NME 21'
lbr = 'RQCD 20'
## use zorder to change plot order for errorbars
phr = plt.fill_between( rqcd.T[0], rqcd.T[1] -rqcd.T[2], rqcd.T[1] +rqcd.T[2],
  color=green, alpha=0.3, label=lbr, zorder=-3)
php = plt.errorbar( pacs.T[0], pacs.T[1], yerr=pacs.T[2],
  color='#508EED',   marker='s', linestyle='', label=lbp, zorder=-2)
php21 = plt.errorbar( pacs21.T[0], pacs21.T[1], yerr=pacs21.T[2],
  color='#508EED',   marker='s', markerfacecolor='none', linestyle='', label=lbp21, zorder=-2)
phc = plt.errorbar( cls.T[0], cls.T[1], yerr=cls.T[2],
  color=sunkist,  marker='^', linestyle='', label=lbc, zorder=-1)
phe = plt.errorbar( etm.T[0], etm.T[1], yerr=etm.T[2],
  color='#3CB24C',   marker='o', linestyle='', label=lbe, zorder=0)
phm = plt.errorbar( mainz.T[0], mainz.T[1], yerr=mainz.T[2],
  color='#606060',     marker='v', linestyle='', label=lbm, zorder=1)
phx = plt.errorbar( callat.T[0], callat.T[1], yerr=callat.T[2],
  color='#FF0F0F',marker='*', linestyle='', label=lbx, zorder=2)
phn = plt.fill_between( nme.T[0],  nme.T[1]  -nme.T[2],  nme.T[1]  +nme.T[2],
  color=violet,  alpha=0.4, label=lbn, zorder=3)

if long_Q2:
  plt.xlim([-0.018,2.0])
  plt.ylim([ 0.0, 1.35])
  plt.xticks( np.arange( 0.0, 2.01, 0.5), fontsize=14)
  plt.text( x=0.33, y=0.42, s=r'$\nu$D $z$ exp', color='red', rotation=-32, fontsize=14)
  fout = 'gaq2-overlay-20.pgf'

else:
  plt.xlim([-0.009,1.0])
  plt.ylim([ 0.3, 1.35])
  plt.xticks( np.arange( 0.0, 1.01, 0.25), fontsize=14)
  plt.text( x=0.25, y=0.60, s=r'$\nu$D $z$ exp', color='red', rotation=-30, fontsize=14)
  fout = 'gaq2-overlay.pgf'

#plt.axis('off')
#plt.text(x=0.25, y=0.65, s=r'$\nu$D $z$ exp', color='red', rotation=-30, fontsize=20) ## old bds

plt.legend(
  [phn, phr, phm, phx, php21, php, phe, phc],
  [lbn, lbr, lbm, lbx, lbp21, lbp, lbe, lbc],
  fontsize=12, ncol=2)
plt.xticks( fontsize=14)
plt.yticks( fontsize=14)
plt.xlabel( r'$Q^2/$GeV$^2$', fontsize=14)
plt.ylabel( r'$F_A(Q^2)$', fontsize=14)

if do_pgf:
  plt.savefig( fout, transparent=True)
else:
  plt.show()

