import numpy as np
import gvar as gv

from params_generic import *
from params_1603_03048 import *
from form_factors import *
from plot_params import *

import matplotlib.pyplot as plt

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
nme = get_data( 'nme_gaq2_z2t00.5.dat')
pacs21 = get_data( 'pacs21_gaq2.dat')
pacs = get_data( 'pacs_data/pacs_gaq2.dat')
rqcd = get_rqcd( 'rqcd_gaq2_zexp4+3.dat')

FA_Q2_fn, _  = define_FA_zexp_sum4( 4, **params_1603_03048)

Q2_range = np.arange( 0.0, 1.02, 0.01)
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

plt.xlim([-0.009,1.0])
plt.ylim([ 0.3, 1.35])
#plt.axis('off')
#plt.text(x=0.25, y=0.65, s=r'$\nu$D $z$ exp', color='red', rotation=-30, fontsize=20) ## old bds
plt.text(x=0.25, y=0.60, s=r'$\nu$D $z$ exp', color='red', rotation=-30, fontsize=14)

plt.legend(
  [phn, phr, phm, phx, php21, php, phe, phc],
  [lbn, lbr, lbm, lbx, lbp21, lbp, lbe, lbc],
  fontsize=12)
plt.xticks( fontsize=14)
plt.yticks( fontsize=14)
plt.xlabel( r'$Q^2/$GeV$^2$', fontsize=14)
plt.ylabel( r'$g_A(Q^2)$', fontsize=14)

if do_pgf:
  plt.savefig('gaq2-overlay.pgf', transparent=True)
else:
  plt.show()

