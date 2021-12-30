import numpy as np

from params_generic import *

## definition of conformal mapping for z
def zconf( Q2, tc, t0):
  return (np.sqrt(tc+Q2) - np.sqrt(tc-t0))/(np.sqrt(tc+Q2) + np.sqrt(tc-t0))

## set up sum rules for z-expansion coefficients, all 4 + norm
def zsum4_rules( n, ga, tc, t0):
  ## constants
  z0 = zconf(0.,tc,t0)
  np1 = n+1; np2 = n+2; np3 = n+3; np4 = n+4
  bd = 6 - np4*np3*np2*np.power(z0,np1) + 3*np4*np3*np1*np.power(z0,np2) \
   - 3*np4*np2*np1*np.power(z0,np3) + np3*np2*np1*np.power(z0,np4)
  zk = np.array([np.power(z0,k) for k in range(1,n+1)])
  zkn = np.array([np.power(z0,k) for k in range(n+1,n+5)])
  k1 = np.array([k for k in range(1,n+1)])
  k2 = np.array([k*(k-1) for k in range(1,n+1)])
  k3 = np.array([k*(k-1)*(k-2) for k in range(1,n+1)])

  ## sums of coefficients that are useful
  def asums(a):
    b0z = -ga + np.sum(a*zk)
    b0 = np.sum(a)
    b1 = np.sum(k1*a)
    b2 = np.sum(k2*a)
    b3 = np.sum(k3*a)
    return (b0z,b0,b1,b2,b3)

  ## sum rule coefficients
  def a0(asum):
    b0z,b0,b1,b2,b3 = asum
    return (1./bd)*( -6*b0z -b0*(bd-6) + np.sum( \
      b3*( np.array([-1, 3, -3, 1.])*zkn ) \
    + b2*( np.array([3*np2, -3*(3*n+5), 3*(3*n+4), -3*np1])*zkn ) \
    + b1*( np.array([-3*np3*np2, 3*np3*(3*n+4), -3*np1*(3*n+8), 3*np2*np1])*zkn ) ))

  def an1(asum):
    b0z,b0,b1,b2,b3 = asum
    zknx = np.array(zkn)
    zknx[0] = 1.
    return (1./bd)*( -(b0-b0z)*np4*np3*np2 + np.sum( \
      b3*( np.array([1, -.5*np4*np3, np4*np2, -.5*np3*np2])*zknx ) \
    + b2*( np.array([-3*np2, np4*np3*np2, -np4*np2*(2*n+3), np3*np2*np1])*zknx ) \
    + b1*( np.array([3*np3*np2, -.5*np4*np3*np3*np2, np4*np3*np2*np1, -.5*np3*np2*np2*np1])*zknx)))

  def an2(asum):
    b0z,b0,b1,b2,b3 = asum
    zknx = np.array(zkn)
    zknx[1] = 1.
    return (1./bd)*( 3*(b0-b0z)*np4*np3*np1 + np.sum( \
      b3*( np.array([.5*np4*np3, -3, -1.5*np4*np1, np3*np1])*zknx ) \
    + b2*( np.array([-np4*np3*np2, 3*(3*n+5), 3*np4*np1*np1, -np3*np1*(2*n+1)])*zknx ) \
    + b1*( np.array([.5*np4*np3*np3*np2, -3*np3*(3*n+4), -1.5*np4*np3*np1*n, np3*np2*np1*n])*zknx)))

  def an3(asum):
    b0z,b0,b1,b2,b3 = asum
    zknx = np.array(zkn)
    zknx[2] = 1.
    return (1./bd)*( -3*(b0-b0z)*np4*np2*np1 + np.sum( \
      b3*( np.array([-np4*np2, 1.5*np4*np1, 3, -.5*np2*np1])*zknx ) \
    + b2*( np.array([np4*np2*(2*n+3), -3*np4*np1*np1, -3*(3*n+4), np2*np1*n])*zknx ) \
    + b1*( np.array([-np4*np3*np2*np1, 1.5*np4*np3*np1*n, 3*np1*(3*n+8), -.5*np2*np1*np1*n])*zknx)))

  def an4(asum):
    b0z,b0,b1,b2,b3 = asum
    zknx = np.array(zkn)
    zknx[3] = 1.
    return (1./bd)*( (b0-b0z)*np3*np2*np1 + np.sum( \
      b3*( np.array([.5*np3*np2, -np3*np1, .5*np2*np1, -1])*zknx ) \
    + b2*( np.array([-np3*np2*np1, np3*np1*(2*n+1), -np2*np1*n, 3*np1])*zknx ) \
    + b1*( np.array([.5*np3*np2*np2*np1, -np3*np2*np1*n, .5*np2*np1*np1*n, -3*np2*np1])*zknx)))

  ## define analytic jacobians for better fitter behavior
  def asums_jac(a):
    a_jac = np.array([ 1 for _ in a ])
    b0z = zk *a_jac
    b0 = a_jac
    b1 = k1 *a_jac
    b2 = k2 *a_jac
    b3 = k3 *a_jac
    return (b0z,b0,b1,b2,b3)

  ## reuse the code from sum rules
  def a0_jac(asum_jac):
    return np.array([ a0( (b0z,b0,b1,b2,b3)) for b0z,b0,b1,b2,b3 in zip( *asum_jac) ])
  def an1_jac(asum_jac):
    return np.array([ an1( (b0z,b0,b1,b2,b3)) for b0z,b0,b1,b2,b3 in zip( *asum_jac) ])
  def an2_jac(asum_jac):
    return np.array([ an2( (b0z,b0,b1,b2,b3)) for b0z,b0,b1,b2,b3 in zip( *asum_jac) ])
  def an3_jac(asum_jac):
    return np.array([ an3( (b0z,b0,b1,b2,b3)) for b0z,b0,b1,b2,b3 in zip( *asum_jac) ])
  def an4_jac(asum_jac):
    return np.array([ an4( (b0z,b0,b1,b2,b3)) for b0z,b0,b1,b2,b3 in zip( *asum_jac) ])

  #def a0_jac(asum_jac):
  #  return np.array([
  #    (1./bd)*( -6*b0z -b0*(bd-6) + np.sum( \
  #    b3*( np.array([-1, 3, -3, 1.])*zkn ) \
  #  + b2*( np.array([3*np2, -3*(3*n+5), 3*(3*n+4), -3*np1])*zkn ) \
  #  + b1*( np.array([-3*np3*np2, 3*np3*(3*n+4), -3*np1*(3*n+8), 3*np2*np1])*zkn ) ))
  #    for b0z,b0,b1,b2,b3 in zip( asum_jac) ])

  #def an1_jac(asum_jac):
  #  zknx = np.array(zkn)
  #  zknx[0] = 1.
  #  return np.array([
  #    (1./bd)*( -(b0-b0z)*np4*np3*np2 + np.sum( \
  #    b3*( np.array([1, -.5*np4*np3, np4*np2, -.5*np3*np2])*zknx ) \
  #  + b2*( np.array([-3*np2, np4*np3*np2, -np4*np2*(2*n+3), np3*np2*np1])*zknx ) \
  #  + b1*( np.array([3*np3*np2, -.5*np4*np3*np3*np2, np4*np3*np2*np1, -.5*np3*np2*np2*np1])*zknx)))
  #    for b0z,b0,b1,b2,b3 in zip( asum_jac) ])

  #def an2_jac(asum_jac):
  #  zknx = np.array(zkn)
  #  zknx[1] = 1.
  #  return np.array([
  #    (1./bd)*( 3*(b0-b0z)*np4*np3*np1 + np.sum( \
  #    b3*( np.array([.5*np4*np3, -3, -1.5*np4*np1, np3*np1])*zknx ) \
  #  + b2*( np.array([-np4*np3*np2, 3*(3*n+5), 3*np4*np1*np1, -np3*np1*(2*n+1)])*zknx ) \
  #  + b1*( np.array([.5*np4*np3*np3*np2, -3*np3*(3*n+4), -1.5*np4*np3*np1*n, np3*np2*np1*n])*zknx)))
  #    for b0z,b0,b1,b2,b3 in zip( asum_jac) ])

  #def an3_jac(asum_jac):
  #  zknx = np.array(zkn)
  #  zknx[2] = 1.
  #  return np.array([
  #    (1./bd)*( -3*(b0-b0z)*np4*np2*np1 + np.sum( \
  #    b3*( np.array([-np4*np2, 1.5*np4*np1, 3, -.5*np2*np1])*zknx ) \
  #  + b2*( np.array([np4*np2*(2*n+3), -3*np4*np1*np1, -3*(3*n+4), np2*np1*n])*zknx ) \
  #  + b1*( np.array([-np4*np3*np2*np1, 1.5*np4*np3*np1*n, 3*np1*(3*n+8), -.5*np2*np1*np1*n])*zknx)))
  #    for b0z,b0,b1,b2,b3 in zip( asum_jac) ])

  #def an4_jac(asum_jac):
  #  zknx = np.array(zkn)
  #  zknx[3] = 1.
  #  return np.array([
  #    (1./bd)*( (b0-b0z)*np3*np2*np1 + np.sum( \
  #    b3*( np.array([.5*np3*np2, -np3*np1, .5*np2*np1, -1])*zknx ) \
  #  + b2*( np.array([-np3*np2*np1, np3*np1*(2*n+1), -np2*np1*n, 3*np1])*zknx ) \
  #  + b1*( np.array([.5*np3*np2*np2*np1, -np3*np2*np1*n, .5*np2*np1*np1*n, -3*np2*np1])*zknx)))
  #    for b0z,b0,b1,b2,b3 in zip( asum_jac) ])

  def apply_sumrules(a):
    asum = asums(a)
    ak = np.array( [a0(asum)] + list(a) + [an1(asum),an2(asum),an3(asum),an4(asum)] )
    return ak

  def apply_sumrules_jac(a):
    asum_jac = asums_jac(a)
    ak_jac = np.vstack((
      [a0_jac(asum_jac)],
      list( np.diag( np.ones(a.shape, dtype=a.dtype))),
      [an1_jac(asum_jac),an2_jac(asum_jac),an3_jac(asum_jac),an4_jac(asum_jac)] )).T
    return ak_jac

  ## make sure the sum rules are satisfied for some test values
  ak = apply_sumrules(np.array(range(1,n+1)))
  if (np.sum( [np.power(z0,k)*x for k,x in enumerate(ak)] ) - ga) > smallNum \
  or (np.sum(ak) ) > smallNum \
  or (np.sum( [k*x for k,x in enumerate(ak)] ) ) > smallNum \
  or (np.sum( [k*(k-1)*x for k,x in enumerate(ak)] ) ) > smallNum \
  or (np.sum( [k*(k-1)*(k-2)*x for k,x in enumerate(ak)] ) ) > smallNum:
    raise ValueError("sum rules not satisfied!")
  return apply_sumrules, apply_sumrules_jac

def define_dipole_fixed( g, M, **params):
  M2 = M*M
  def Dipole( Q2):
    return g/np.power(1+Q2/M2,2)
  return Dipole

## axial/pseudoscalar
def define_FA_dipole( **params):
  ga = params['ga']
  def FA_dipole( Q2, MA):
    MA2 = MA[0] *MA[0]
    return ga/np.power(1+Q2/MA2,2)
  def FA_dipole_jac( Q2, MA):
    MA2 = MA[0] *MA[0]
    ## match shape assumptions of zexp
    return np.array([ (4*Q2*ga)/np.power(MA[0]*(1+Q2/MA2),3) ])
  return FA_dipole,FA_dipole_jac

## create a z expansion form factor with the requested number of coefficients
## builds sum rules for form factor
def define_FA_zexp_sum4( n, **params):
  ga = params['ga']
  tc = params['tc']
  t0 = params['t0']
  apply_sumrules,apply_sumrules_jac = zsum4_rules( n, ga, tc, t0)
  def FA_zexp( Q2, a):
    ak = apply_sumrules( a)
    z = zconf( Q2, tc, t0)
    zk = np.array([ np.power( z, k) for k in range(n+5)])
    return np.dot( ak, zk)
  def FA_zexp_jac( Q2, a):
    ak_jac = apply_sumrules_jac( a)
    z = zconf( Q2, tc, t0)
    zk = np.array([ np.power( z, k) for k in range(n+5)])
    return np.dot( ak_jac, zk)
  return FA_zexp,FA_zexp_jac

def define_FP_PCAC( FA, FA_jac, **params):
  MN2  = params['MN'] *params['MN']
  Mpi2 = params['Mpi'] *params['Mpi']
  def FP( Q2, axial_param):
    return 2. *MN2 *FA( Q2, axial_param) /(Mpi2 +Q2)
  def FP_jac( Q2, axial_param):
    return 2. *MN2 *FA_jac( Q2, axial_param) /(Mpi2 +Q2)
  return FP,FP_jac

## BBA form factors
## defined as power series of Q2, not Q2/4.M_N^2
def define_BBA_generic( b_list, **params):
  b_list = np.array( b_list) ## denominator
  Nb = len( b_list)
  def BBA_generic( Q2):
    Q2_den = np.array([ np.power( Q2, n+1) for n in range( Nb) ])
    return 1. /(1. +np.dot( Q2_den, b_list))
  return BBA_generic

## BBA05 form factors
def define_BBA05_generic( a_list, b_list, **params):
  MN = params['MN']
  tau = 1./(4.*MN*MN)
  a_list = np.array( a_list) ## numerator
  b_list = np.array( b_list) ## denominator
  Na = len( a_list)
  Nb = len( b_list)
  def BBA05_generic( Q2):
    Q2_num = np.array([ np.power( Q2 *tau, n  ) for n in range( Na) ])
    Q2_den = np.array([ np.power( Q2 *tau, n+1) for n in range( Nb) ])
    return np.dot( Q2_num, a_list) /(1. +np.dot( Q2_den, b_list))
  return BBA05_generic

## numbers used in 1603.03048
BBA_b_list_Ep_1603_03048 = [3.253, 1.422,  0.08582, 0.3318,  -0.09371,    0.01076]
BBA_b_list_En_1603_03048 = [0.942, 4.61]
BBA_b_list_Mp_1603_03048 = [3.104, 1.428,  0.1112, -0.006981, 0.0003705, -0.7063e-5]
BBA_b_list_Mn_1603_03048 = [3.043, 0.8548, 0.6806, -0.1287,   0.008912]

## numbers used in 1603.03048; from Table 1 of 0602017[hep-ex]
BBA05_a_list_Ep_1603_03048 = [ 1.      ,  -0.0578]
BBA05_a_list_En_1603_03048 = [ 0.      ,   1.25      ,    1.3 ]
BBA05_a_list_Mp_1603_03048 = [ 1.      ,   0.150 ]
BBA05_a_list_Mn_1603_03048 = [ 1.      ,   1.81  ]
BBA05_b_list_Ep_1603_03048 = [11.1     ,  13.6       ,   33.  ]
BBA05_b_list_En_1603_03048 = [-9.86    , 305.        , -758.      , 802.]
BBA05_b_list_Mp_1603_03048 = [11.1     ,  19.6       ,    7.54]
BBA05_b_list_Mn_1603_03048 = [14.1     ,  20.7       ,   68.7 ]

## precise numbers from ??
BBA05_a_list_Ep_precise = [ 1.      ,  -0.05777087]
BBA05_a_list_En_precise = [ 0.      ,   1.249971  ,    1.297130]
BBA05_a_list_Mp_precise = [ 1.      ,   0.1502468 ]
BBA05_a_list_Mn_precise = [ 1.      ,   1.817919  ]
BBA05_b_list_Ep_precise = [11.17884 ,  13.64415   ,   33.0309  ]
BBA05_b_list_En_precise = [-9.861796, 305.5084    , -758.3796  , 801.7726]
BBA05_b_list_Mp_precise = [11.05393 ,  19.60770   ,    7.544214]
BBA05_b_list_Mn_precise = [14.09648 ,  20.70172   ,   68.66369 ]

BBA_b_list_Ep = BBA_b_list_Ep_1603_03048
BBA_b_list_En = BBA_b_list_En_1603_03048
BBA_b_list_Mp = BBA_b_list_Mp_1603_03048
BBA_b_list_Mn = BBA_b_list_Mn_1603_03048
if 1:
  BBA05_a_list_Ep = BBA05_a_list_Ep_1603_03048
  BBA05_a_list_En = BBA05_a_list_En_1603_03048
  BBA05_a_list_Mp = BBA05_a_list_Mp_1603_03048
  BBA05_a_list_Mn = BBA05_a_list_Mn_1603_03048
  BBA05_b_list_Ep = BBA05_b_list_Ep_1603_03048
  BBA05_b_list_En = BBA05_b_list_En_1603_03048
  BBA05_b_list_Mp = BBA05_b_list_Mp_1603_03048
  BBA05_b_list_Mn = BBA05_b_list_Mn_1603_03048
else:
  BBA05_a_list_Ep = BBA05_a_list_Ep_precise
  BBA05_a_list_En = BBA05_a_list_En_precise
  BBA05_a_list_Mp = BBA05_a_list_Mp_precise
  BBA05_a_list_Mn = BBA05_a_list_Mn_precise
  BBA05_b_list_Ep = BBA05_b_list_Ep_precise
  BBA05_b_list_En = BBA05_b_list_En_precise
  BBA05_b_list_Mp = BBA05_b_list_Mp_precise
  BBA05_b_list_Mn = BBA05_b_list_Mn_precise

## a bit different than the other BBA form factors
def define_GEn_BBA( **params):
  mu_n = params['mu_n']
  MN = params['MN']
  MV = params['MV']
  num_mu_n_tau = -BBA_b_list_En[0] *mu_n/(4.*MN*MN)
  den_tau = BBA_b_list_En[1] /(4.*MN*MN)
  GD = define_dipole_fixed( num_mu_n_tau, MV, **params)
  return (lambda Q2: Q2 *GD(Q2) /(1 +den_tau*Q2))

def define_GEp_BBA( **params):
  return define_BBA_generic( BBA_b_list_Ep, **params)

def define_GMp_BBA( **params):
  mu_p = params['mu_p']
  Mp = define_BBA_generic( BBA_b_list_Mp, **params)
  return (lambda Q2: mu_p *Mp( Q2))

def define_GMn_BBA( **params):
  mu_n = params['mu_n']
  Mn = define_BBA_generic( BBA_b_list_Mn, **params)
  return (lambda Q2: mu_n *Mn( Q2))

def define_GEp_BBA05( **params):
  return define_BBA05_generic( BBA05_a_list_Ep, BBA05_b_list_Ep, **params)

def define_GEn_BBA05( **params):
  return define_BBA05_generic( BBA05_a_list_En, BBA05_b_list_En, **params)

def define_GMp_BBA05( **params):
  mu_p = params['mu_p']
  Mp = define_BBA05_generic( BBA05_a_list_Mp, BBA05_b_list_Mp, **params)
  return (lambda Q2: mu_p *Mp( Q2))

def define_GMn_BBA05( **params):
  mu_n = params['mu_n']
  Mn = define_BBA05_generic( BBA05_a_list_Mn, BBA05_b_list_Mn, **params)
  return (lambda Q2: mu_n *Mn( Q2))

def define_GEV_Dipole_1603_03048_iter0( **params):
  MV = params['MV']
  dipole = define_dipole_fixed( 1., MV, **params)
  return dipole

def define_GMV_Dipole_1603_03048_iter0( **params):
  mu_p = params['mu_p']
  mu_n = params['mu_n']
  MV = params['MV']
  dipole = define_dipole_fixed( mu_p-mu_n, MV, **params)
  return dipole

def define_GEV_BBA( **params):
  GEp = define_GEp_BBA( **params)
  GEn = define_GEn_BBA( **params)
  return (lambda Q2: GEp( Q2) -GEn( Q2))

def define_GMV_BBA( **params):
  GMp = define_GMp_BBA( **params)
  GMn = define_GMn_BBA( **params)
  return (lambda Q2: GMp( Q2) -GMn( Q2))

def define_GEV_BBA05( **params):
  GEp = define_GEp_BBA05( **params)
  GEn = define_GEn_BBA05( **params)
  return (lambda Q2: GEp( Q2) -GEn( Q2))

def define_GMV_BBA05( **params):
  GMp = define_GMp_BBA05( **params)
  GMn = define_GMn_BBA05( **params)
  return (lambda Q2: GMp( Q2) -GMn( Q2))

def define_F1_Generic( GEV, GMV, **params):
  MN = params['MN']
  tau = 1. /(4.*MN*MN)
  def F1( Q2):
    return (GEV( Q2) +Q2*tau *GMV( Q2)) /(1. +Q2*tau)
  return F1

def define_F2_Generic( GEV, GMV, **params):
  MN = params['MN']
  tau = 1. /(4.*MN*MN)
  def F2( Q2):
    return (GMV( Q2) -GEV( Q2)) /(1. +Q2*tau)
  return F2

