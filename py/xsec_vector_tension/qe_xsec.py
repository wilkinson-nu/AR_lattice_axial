import numpy as np
import scipy.integrate as integrate

from params_generic import *

## kinematic functions
def Ecom( Enu, MN):
  return Enu *np.sqrt( MN /(MN +2.*Enu) )

def Elep( Enu, MN, mmu):
  Ecm = Ecom( Enu, MN)
  mmu2 = mmu *mmu
  MN2 = MN *MN
  return Ecm +mmu2/( 2.*Ecm + 2.*np.sqrt( Ecm*Ecm +MN2) )

def Q2bd( Enu, MN, mmu):
  mmu2 = mmu *mmu
  El = Elep( Enu, MN, mmu)
  if El*El < mmu2:
    return mmu2,mmu2
  Ecm = Ecom( Enu, MN)
  loBd = -mmu2 + 2.*Ecm*( El - np.sqrt( El*El - mmu2 ) )
  upBd = -mmu2 + 2.*Ecm*( El + np.sqrt( El*El - mmu2 ) )
  return loBd,upBd

def sminusu( Enu, Q2, MN, mmu):
  mmu2 = mmu *mmu
  return 4.*MN*Enu -Q2 -mmu2

## differential cross sections
def define_A( F1, F2, FA, FP, FA_jac, FP_jac, **params):
  mmu2 = params['mmu'] *params['mmu']
  MN2 = params['MN'] *params['MN']
  def Afn( Q2, axial_param):
    F1Q2 = F1(Q2)
    F2Q2 = F2(Q2)
    FAQ2 = FA(Q2, axial_param)
    FPQ2 = FP(Q2, axial_param)
    return ((mmu2 +Q2)/(4.*MN2)) *(
      (4.+Q2/MN2)*FAQ2*FAQ2
    - (4.-Q2/MN2)*(F1Q2*F1Q2 - (Q2/(4.*MN2))*F2Q2*F2Q2)
    + (4.*Q2/MN2)*F1Q2*F2Q2
    - (mmu2/MN2)*(
        np.power(FAQ2 + 2.*FPQ2,2.)
      + np.power(F1Q2 + F2Q2,2.)
      - (4.+Q2/MN2)*FPQ2*FPQ2 ) )
  def Afn_jac( Q2, axial_param):
    F1Q2 = F1(Q2)
    F2Q2 = F2(Q2)
    FAQ2 = FA(Q2, axial_param)
    FPQ2 = FP(Q2, axial_param)
    FAQ2_jac = FA_jac(Q2, axial_param)
    FPQ2_jac = FP_jac(Q2, axial_param)
    return ((mmu2 +Q2)/(4.*MN2)) *(
      (4.+Q2/MN2)*2*FAQ2*FAQ2_jac
    - (mmu2/MN2)*(
        (FAQ2 +2.*FPQ2)*2*(FAQ2_jac +2.*FPQ2_jac)
      - (4.+Q2/MN2)*2*FPQ2*FPQ2_jac ) )
  return Afn,Afn_jac

def define_B( F1, F2, FA, FP, FA_jac, FP_jac, **params):
  sign_convention = np.sign( params['ga'])
  mmu2 = params['mmu'] *params['mmu']
  MN2 = params['MN'] *params['MN']
  def Bfn( Q2, axial_param):
    return sign_convention *(Q2/MN2) *FA(Q2, axial_param)*(F1(Q2) + F2(Q2))
  def Bfn_jac( Q2, axial_param):
    return sign_convention *(Q2/MN2) *FA_jac(Q2, axial_param)*(F1(Q2) + F2(Q2))
  return Bfn,Bfn_jac

def define_C( F1, F2, FA, FP, FA_jac, FP_jac, **params):
  mmu2 = params['mmu'] *params['mmu']
  MN2 = params['MN'] *params['MN']
  def Cfn( Q2, axial_param):
    F1Q2 = F1(Q2)
    F2Q2 = F2(Q2)
    FAQ2 = FA(Q2, axial_param)
    return (F1Q2*F1Q2 + FAQ2*FAQ2 + (Q2/(4.*MN2))*F2Q2*F2Q2)/4.
  def Cfn_jac( Q2, axial_param):
    return (FA(Q2, axial_param) *FA_jac(Q2, axial_param)) /2.
  return Cfn,Cfn_jac

def get_prefactor( units, **params):
  if units == 'no_prefactor':
    ## very small prefactor can cause problems for integration
    ## return a trivial prefactor
    return 1.
  MN = params['MN']
  mmu = params['mmu']
  GF = params['GF']
  cosThetaC = params['cosThetaC']
  prefactor = np.power( MN *GF *cosThetaC, 2) /(8. *np.pi)
  if   units == 'cm2':
    prefactor *= GeVFm*GeVFm *1e-26
  elif units == 'GeV-2':
    pass ## already in correct units
  else:
    raise ValueError( "invalid units")
  return prefactor

def define_dsigma_dQ2( F1, F2, FA, FP, FA_jac, FP_jac,
  units='cm2', do_antineutrino=False, **params):
  ## generic differential cross section in Q2
  MN = params['MN']
  mmu = params['mmu']
  neutrino_sign = (-1 if do_antineutrino else 1)
  prefactor = get_prefactor( units, **params)
  Afn,Afn_jac = define_A( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  Bfn,Bfn_jac = define_B( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  Cfn,Cfn_jac = define_C( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  def smu( Enu, Q2):
    return sminusu( Enu, Q2, MN, mmu)
  def dsigma_dQ2( Enu, Q2, axial_param):
    su_M2 = smu( Enu, Q2) /(MN*MN)
    AQ2 = Afn( Q2, axial_param)
    BQ2 = Bfn( Q2, axial_param)
    CQ2 = Cfn( Q2, axial_param)
    return (prefactor /(Enu*Enu)) *( AQ2 +neutrino_sign *BQ2 *su_M2 +CQ2 *su_M2*su_M2)
  def dsigma_dQ2_jac( Enu, Q2, axial_param):
    su_M2 = smu( Enu, Q2) /(MN*MN)
    AQ2_jac = Afn_jac( Q2, axial_param)
    BQ2_jac = Bfn_jac( Q2, axial_param)
    CQ2_jac = Cfn_jac( Q2, axial_param)
    return (prefactor /(Enu*Enu)) *(
      AQ2_jac +neutrino_sign *BQ2_jac *su_M2 +CQ2_jac *su_M2*su_M2)
  return dsigma_dQ2,dsigma_dQ2_jac

def define_vectorized_dsigma_dQ2( F1, F2, FA, FP, FA_jac, FP_jac,
  units='cm2', do_antineutrino=False, **params):
  ## differential cross section vectorized to speed up computation of several Q2, Ev
  MN = params['MN']
  mmu = params['mmu']
  neutrino_sign = (-1 if do_antineutrino else 1)
  prefactor = get_prefactor( units, **params)
  Afn,Afn_jac = define_A( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  Bfn,Bfn_jac = define_B( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  Cfn,Cfn_jac = define_C( F1, F2, FA, FP, FA_jac, FP_jac, **params)
  def smu( Enu, Q2):
    return sminusu( Enu, Q2, MN, mmu)
  ## returns values of dsigma_dQ2(Ev_i, Q2_j) in a table X_{ij}
  ## mask is assumed to be same shape as X
  def vectorized_dsigma_dQ2( Enu_list, Q2_list, axial_param, mask=False):
    su_M2 = np.array([[ smu( Enu, Q2) /(MN*MN)
      for Q2 in Q2_list] for Enu in Enu_list ])
    pf  = np.array([ (prefactor /(Enu*Enu)) for Enu in Enu_list ])
    AQ2 = np.array([ Afn( Q2, axial_param) for Q2 in Q2_list ])
    BQ2 = np.array([ Bfn( Q2, axial_param) for Q2 in Q2_list ])
    CQ2 = np.array([ Cfn( Q2, axial_param) for Q2 in Q2_list ])
    Aterm = np.outer( pf, AQ2)
    Bterm = neutrino_sign *np.outer( pf, BQ2) *su_M2
    Cterm = np.outer( pf, CQ2) *su_M2 *su_M2
    if np.any( mask):
      return (Aterm +Bterm +Cterm)*mask
    return Aterm +Bterm +Cterm
  def vectorized_dsigma_dQ2_jac( Enu_list, Q2_list, axial_param, mask=False):
    su_M2 = np.array([[ smu( Enu, Q2) /(MN*MN)
      for Q2 in Q2_list] for Enu in Enu_list ])
    pf  = np.array([ (prefactor /(Enu*Enu)) for Enu in Enu_list ])
    AQ2_jac = np.array([ Afn_jac( Q2, axial_param) for Q2 in Q2_list ])
    BQ2_jac = np.array([ Bfn_jac( Q2, axial_param) for Q2 in Q2_list ])
    CQ2_jac = np.array([ Cfn_jac( Q2, axial_param) for Q2 in Q2_list ])
    Aterm = np.einsum( 'e,qj->eqj', pf, AQ2_jac)
    Bterm = neutrino_sign *np.einsum( 'e,qj,eq->eqj', pf, BQ2_jac, su_M2)
    Cterm = np.einsum( 'e,qj,eq->eqj', pf, CQ2_jac, su_M2 *su_M2)
    if np.any( mask):
      return np.einsum( 'eqj,eq->eqj', Aterm +Bterm +Cterm, mask)
    return Aterm +Bterm +Cterm
  return vectorized_dsigma_dQ2,vectorized_dsigma_dQ2_jac

def define_sigma_tot( F1, F2, FA, FP, FA_jac, FP_jac,
  units='cm2', do_antineutrino=False,
  include_tolerance=False, **params):
  ## generic total cross section
  #
  MN = params['MN']
  mmu = params['mmu']
  prefactor = get_prefactor( units, **params)
  dsigma_dQ2, dsigma_dQ2_jac = define_dsigma_dQ2( F1, F2, FA, FP, FA_jac, FP_jac,
    units='no_prefactor', do_antineutrino=do_antineutrino, **params)
  if include_tolerance:
    def sigma_tot( Enu, axial_param):
      Q2low, Q2high = Q2bd( Enu, MN, mmu)
      res = integrate.quad(
        lambda Q2: dsigma_dQ2( Enu, Q2, axial_param),
        Q2low, Q2high, limit=100)
      return prefactor *np.array( res)
  else:
    def sigma_tot( Enu, axial_param):
      Q2low, Q2high = Q2bd( Enu, MN, mmu)
      res = integrate.quad(
        lambda Q2: dsigma_dQ2( Enu, Q2, axial_param),
        Q2low, Q2high, limit=100)[0]
      return prefactor *res
  return sigma_tot

