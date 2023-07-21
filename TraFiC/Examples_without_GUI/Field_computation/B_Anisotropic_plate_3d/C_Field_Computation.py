# Version 1.0 - 2023, July, 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Imported modules
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import sys, os
#++++ TraFiC location ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Relative path to TraFiC code:
rel_TraFiC_path = "../../.." 
root = os.path.abspath(rel_TraFiC_path)
sys.path.append(root)
import TraFiC_init
#++++ TraFiC classes and utilities +++++++++++++++++++++++++++++++++++++++
from FieldComputationInProgress import FieldComputationInProgress
from TraFiC_utilities import now
#=========================================================================
# I.1) Loading the FieldComputationInProgress instance
comp_path = "./Results/Computation_in_progress.txt"
my_computation = FieldComputationInProgress(comp_path, verbose=True)
print(f"Excitations:\n{my_computation.list_of_excitation_labels}")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I.2) Loading the FieldComputation instance for a given excitation
for simu_name in ("Normal", "Tangential") :
    simu = my_computation(simu_name)
#=========================================================================
# II - Field computation
    simu.compute( ["Ux", "Uy", "Uz"], [ 0.0, 1.8 ] )
    print( os.listdir(simu.field_path) )
#=========================================================================
