# Version 1.0 - 2023, July, 17
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
rel_TraFiC_path = "../.." 
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
simu = my_computation("Demo_Fx")
simu.plot_excitations(show_now=False)
#=========================================================================
# II - Field computation
simu.compute( ["Ux", "Uy", "Uz"], \
              [ 0.0, 0.6, 1.35, 2.5, 3.49, 3.51, 5.0] )
print( os.listdir(simu.field_path) )
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
#=========================================================================
