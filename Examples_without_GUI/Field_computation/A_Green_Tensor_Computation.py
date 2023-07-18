# Version 1.0 - 2023, July, 11
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Imported modules
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
import sys, os, pickle
#++++ TraFiC location ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Relative path to TraFiC code:
rel_TraFiC_path = "../.." 
root = os.path.abspath(rel_TraFiC_path)
sys.path.append(root)
import TraFiC_init
#++++ TraFiC classes and utilities +++++++++++++++++++++++++++++++++++++++
from MaterialClasses import *
from USMultilayerPlate import USMultilayerPlate as USMP
from ComputationParameters import ComputationParameters
from FieldComputationInProgress import FieldComputationInProgress
from TraFiC_utilities import now
#=========================================================================
# I. Simulation parameters in time and space
#     Simplifying assumptions :
#          > 2D simulation : fields with respect to position x along the
#            plate, position z normal to the plate and time t
#=========================================================================
# I.1) Input
# Total duration of the simulation, in microseconds:
duration_µs = 170.0
# Delay such that the excitation signal can be centered at t = 0
delay_µs = 30.0
print(f"Time Interval [{-delay_µs:.2f}, " + \
      f"{duration_µs-delay_µs:.2f}] in µs")
# Maximum frequency in MegaHertz:
max_frequency_MHz = 1.25
# Maximum distance between source(s) and observation point(s), in meters
max_length_m = 1.0
# Minimum wavelength in millimeters (x-direction)
min_wavelength_mm = 1.0
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I.2) Call of the ComputationParameters constructor
CP = ComputationParameters("Demo", duration_µs, delay_µs, \
                           max_frequency_MHz, max_length_m, \
                           min_wavelength_mm, "Results")
print(CP)
print(f"Maximum wave velocity: {CP.maximum_wave_velocity} mm/µs")
#=========================================================================
# II. Definition of the multilayer plate by reading file
#     Simplifying assumptions :
#          > Upper half-space is vacuum
#          > The first layer is solid
#=========================================================================
# II.1) Import of a USMultilayerPlate (USMP) from a text file
plate_path = os.path.join( CP.abs_plate_path, "Demo_july23.txt" )
my_plate = USMP.import_from_file(plate_path)
#print(my_plate)
if my_plate.topUSmat is not None :
    print(f"Warning: the top half-space is not vacuum!")
if isinstance( my_plate.layers[0].usm.mat, Fluid) :
    print(f"Warning: the first layer is not solid!")
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.2) Initialization of the computation   
my_computation = FieldComputationInProgress(CP, my_plate, verbose=True)
#=========================================================================
# III. Green tensor computation in the (s, k_x)-domain
#     Simplifying assumptions :
#          > Surface force exerted on the upper interface with vacuum,
#            with 3 possible directions "Fx", "Fy" and "Fz"
my_computation.compute_Green_tensors( ["Fx", "Fy", "Fz"] )
print(my_computation.log)

##########################################################################
###   Field computations are done from the saved Green tensors and after
###   defining sources. See "B_Excitation definition.py" and
###                         "C_Field_Computation.py"
##########################################################################
