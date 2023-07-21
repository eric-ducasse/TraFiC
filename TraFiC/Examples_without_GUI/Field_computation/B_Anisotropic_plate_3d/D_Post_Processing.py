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
from SpaceGridClasses import Space2DGrid
from TimeGridClass import TimeGrid
#=========================================================================
# I.1) Loading the FieldComputationInProgress instance
comp_path = "./Results/Computation_in_progress.txt"
my_computation = FieldComputationInProgress(comp_path)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I.2) Loading the FieldComputation instance for a given excitation
simu = my_computation("Tangential")
print(f"Computed fields for '{simu.name}':\n\t" + \
       "\n\t".join( [ f"{i:2d} > {nm}" for i,nm in \
                      enumerate(simu.computed_fields) ] ) )
#=========================================================================
# II - Visualization
# Field
file_idx = 0
file_pth = os.path.join( simu.field_path, simu.computed_fields[file_idx] )
fld_name, pos_mm = simu.computed_cases[file_idx]
F = np.load(file_pth, mmap_mode="r")
# Time and space grids
tm_gd = my_computation.time_grid
µs_gd = TimeGrid( 1e6*tm_gd.duration, 1e6*tm_gd.Ts, 1e6*tm_gd.t0 )
print(f"{20*'*'} Time grid in microseconds {20*'*'}\n{µs_gd}")
sp_gd = my_computation.space_grid
mm_gd = Space2DGrid( sp_gd.nx, sp_gd.ny, 1e3*sp_gd.dx, 1e3*sp_gd.dy )
print(f"{20*'*'} Space grid in millimeters {20*'*'}\n{mm_gd}")
# Zero-padding
no_t = 168
t_µs = µs_gd.T[no_t]
title = f"Snapshot of '{fld_name}' at t = {t_µs:.2f} µs " + \
        f"and z = {pos_mm} mm"
print(title)
F = 1e12*F[no_t,...]
mm_zp, F2 = mm_gd.zero_padding(F, 2 , 2, centered=True)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure in the space-time domain
fig = plt.figure(title, figsize=(12.0,7.2))
fig.subplots_adjust(0.08,0.08,0.92,0.92)
opt = simu.PLT_OPT
ax = fig.subplots(1,1)
fig.suptitle(title, **opt)
mm_zp.plot( F2, centered=True,  draw_axis=ax)
ax.set_xlabel("Position $x$ [mm]", **opt)
ax.set_ylabel("Position $y$ [mm]", **opt)
ax.grid()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
