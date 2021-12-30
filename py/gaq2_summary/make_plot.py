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
nme = get_data( 'nme_gaq2_z2t00.5.dat')
pacs21 = get_data( 'pacs21_gaq2.dat')
pacs = get_data( 'pacs_data/pacs_gaq2.dat')
rqcd = get_rqcd( 'rqcd_gaq2_zexp4+3.dat')

FA_Q2_fn, _  = define_FA_zexp_sum4( 4, **params_1603_03048)

Q2_range = np.arange( 0.0, 0.52, 0.01)
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
plt.subplots_adjust( bottom=0.15, top=0.95, right=0.95)
plt.fill_between( Q2_range, ym, yp, color='red', alpha=0.3)

lbc = 'CLS 17 (1 ens)'
lbe = 'ETMC 20 (1 ens)'
lbp = 'PACS 18 erratum (1 ens)'
lbp21= 'PACS 21 (1 ens)'
lbn = 'NME 21'
lbr = 'RQCD 20'
phc = plt.errorbar( cls.T[0], cls.T[1], yerr=cls.T[2],
  color=turquoise, marker='^', linestyle='', label=lbc)
phe = plt.errorbar( etm.T[0], etm.T[1], yerr=etm.T[2],
  color=orange, marker='o', linestyle='', label=lbe)
php = plt.errorbar( pacs.T[0], pacs.T[1], yerr=pacs.T[2],
  color=violet, marker='s', linestyle='', label=lbp)
php21 = plt.errorbar( pacs21.T[0], pacs21.T[1], yerr=pacs21.T[2],
  color=violet, marker='s', markerfacecolor='none', linestyle='', label=lbp21)
phn = plt.fill_between( nme.T[0],  nme.T[1]  -nme.T[2],  nme.T[1]  +nme.T[2],
  color='blue',  alpha=0.3, label=lbn)
phr = plt.fill_between( rqcd.T[0], rqcd.T[1] -rqcd.T[2], rqcd.T[1] +rqcd.T[2],
  color='green', alpha=0.3, label=lbr)

plt.xlim([-0.009,0.5])
plt.ylim([ 0.6, 1.35])
#plt.axis('off')
plt.text(x=0.25, y=0.65, s=r'$\nu$D $z$ exp', color='red', rotation=-30, fontsize=20)

plt.legend([phn, phr, phe, php, php21, phc], [lbn, lbr, lbe, lbp, lbp21, lbc], fontsize=12)
plt.xticks( fontsize=20)
plt.yticks( fontsize=20)
plt.xlabel( r'$Q^2/$GeV$^2$', fontsize=20)
plt.ylabel( r'$g_A(Q^2)$', fontsize=20)

if do_pgf:
  plt.savefig('gaq2-overlay.pgf', transparent=True)
else:
  plt.show()

