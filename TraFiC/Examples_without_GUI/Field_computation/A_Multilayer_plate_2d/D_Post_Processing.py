# Version 1.01 - 2023, July, 20
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
my_computation = FieldComputationInProgress(comp_path)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# I.2) Loading the FieldComputation instance for a given excitation
simu = my_computation("Demo_Fx")
print(f"Computed fields for '{simu.name}':\n\t" + \
       "\n\t".join( [ f"{i:2d} > {nm}" for i,nm in \
                      enumerate(simu.computed_fields) ] ) )
#=========================================================================
# II - Visualization
# Field
file_idx = 15
file_pth = os.path.join( simu.field_path, simu.computed_fields[file_idx] )
fld_name, pos_mm = simu.computed_cases[file_idx]
F0 = 1e12 * np.load(file_pth) # Unit force excitation => correction
# Time and space grids
tm_gd = my_computation.time_grid
sp_gd = my_computation.space_grid
# Zero-padding
tm_zp, F = tm_gd.zero_padding( F0, 5, Laplace=True)
sp_zp, F = sp_gd.zero_padding( F, 3, axis=1, centered=True )
# Ranges
xrange = 1e3*sp_zp.g_range                             # [mm]
trange = 1e6*tm_zp.t_range                             # [µs]
frange = 1e-6*tm_gd.f_range                            # [MHz]
krange = 1e-3*np.array( [sp_gd.k_min - 0.5*sp_gd.dk, \
                         sp_gd.k_max + 0.5*sp_gd.dk] ) # [/mm]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure in the space-time domain
title = f"Visualization of '{fld_name}' at z = {pos_mm:.2f} mm"
fig = plt.figure(title, figsize=(10,7.2))
fig.subplots_adjust(0.08,0.08,0.92,0.92)
opt = simu.PLT_OPT
ax = fig.subplots(1,1)
fig.suptitle(title, **opt)
ax.set_title("(colors in logarithmic scale)")
absF = np.abs(F)
absF /= absF.max()
db_max = 3.1
thres = 10**-db_max
absF = (absF>thres)*absF + (absF<=thres)*thres
log10F = (db_max + np.log10(absF)) * ( (F>0)*1 + (F<0)*(-1) )
im = ax.imshow( log10F, cmap="seismic", vmin=-db_max, vmax=db_max, \
                interpolation="none", aspect="auto", \
                extent=np.append(xrange,trange[::-1]) )
ax.set_xlabel("Position $x$ [mm]", **opt)
ax.set_xlim(-350, 350)
ax.set_ylabel("Time $t$ [µs]", **opt)
ax.set_ylim(52, -12)
ax.grid()
dvd = make_axes_locatable(ax)
cax = dvd.append_axes('right', size='2%', pad=0.06)
plt.colorbar(im, cax=cax)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Figure in the wavenumber-frequency domain
SK = np.abs( sp_gd.fft( tm_gd.rfft( F0 ), axis=1, centered=True ) ).T
SK /= SK.max()
db_max_fk = 2.6
thres_fk = 10**-db_max_fk
SK = (SK>thres_fk)*SK + (SK<=thres_fk)*thres
log10SK = db_max_fk + np.log10(SK)
titfk = f"'{fld_name}' at z = {pos_mm:.2f} mm in the f-k domain"
figfk = plt.figure(titfk, figsize=(9,7))
figfk.subplots_adjust(0.08,0.08,0.92,0.92)
figfk.suptitle(titfk, **opt)
fk = figfk.subplots(1,1)
fk.set_title("(colors in logarithmic scale)")
imfk = fk.imshow( log10SK , cmap="seismic", vmin=0, vmax=db_max_fk, \
                  interpolation="none", aspect="auto", origin="lower",\
                  extent=np.append(frange,krange) )
dvdfk = make_axes_locatable(fk)
caxfk = dvdfk.append_axes('right', size='2%', pad=0.06)
plt.colorbar(imfk, cax=caxfk)
fk.set_xlabel("Frequency $f$ [MHz]", **opt)
fk.set_ylabel(r"Wavenumber $k$ [$\mathrm{mm}^{-1}$]", **opt)
fk.grid( color="#AAFFAA" )
fk.set_xlim(-3e-3, 0.82)
fk.set_ylim(-3.5, 3.5)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
