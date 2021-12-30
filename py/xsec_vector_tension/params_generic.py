import numpy as np
import gvar as gv

smallNum = 1e-8

## BBA05 parameters
mu_p_BBA05 = 2.793
mu_n_BBA05 = -1.913
MA_world     = 1.014       ## GeV
err_MA_world = 0.014       ## GeV

MA_dune = MA_world *gv.gvar( 1., 0.15)

## other parameters
GeVFm        = 0.197326    ## GeV.fm, with more precision

