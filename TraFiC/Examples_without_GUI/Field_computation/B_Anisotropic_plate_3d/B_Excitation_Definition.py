# Version 1.01 - 2024, January, 29
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
import sys, os, pickle
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
#=========================================================================
# I. Loading the FieldComputationInProgress instance
comp_path = "./Results/Computation_in_progress.txt"
my_computation = FieldComputationInProgress(comp_path, verbose=True)
print(my_computation.stage)
print(my_computation.possible_excitations)
#=========================================================================
# II. Excitation definition
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.1) Signal : n-cycle pulse, with checking in the time grid
number_of_cycles = 1.5
central_frequency = 0.2e6 # 200 kHz
def Ncycle_pulse(t, fc=central_frequency, nbc=number_of_cycles) :
    wt = 2*np.pi*fc*t
    at = np.sqrt(20)*fc*t/nbc
    window = ((at>-5)&(at<5))*np.exp(-np.clip(at,-5,5)**2)
    return np.sin(wt)*window
# Time grid
tm_gd = my_computation.time_grid
# Sampled signal
val_sig = Ncycle_pulse( tm_gd.time_values )
# Zero-padding
zp_gd, val_zp = tm_gd.zero_padding(val_sig, 9, Laplace=True)
# Amplitude Spectral Density (ASD)
val_asd = 2*np.abs(tm_gd.rfft(val_sig))
# Time and frequency plots
fig_sig = plt.figure("Signals and ASD", figsize=(14,7))
opt = {"size":14, "family":"Arial", "weight":"bold"}
ax_dict = fig_sig.subplot_mosaic( \
    """AA
       BC""")
axT, axZ, axF = [ ax_dict[c] for c in "ABC" ]
fig_sig.subplots_adjust(0.06,0.08,0.99,0.94,0.2,0.3)
fig_sig.suptitle("Excitation signal and its amplitude " + \
                 "spectral density (ASD)", **opt)
for ax in axT, axZ :
    ax.set_xlabel("Time $t$ [µs]", **opt)
    ax.set_ylabel("Signal values [a.u.]", **opt)
    ax.plot( 1e6*zp_gd.T, val_zp, "-b")
    ax.plot( 1e6*tm_gd.T, val_sig, ".r")
    ax.grid()
axT.set_xlim(1e6*tm_gd.t0,1e6*tm_gd.T[-1])
axZ.set_xlim( -5.3, 8.3 )
axF.set_xlabel("Frequency $f$ [MHz]", **opt)
axF.set_ylabel("ASD values [a.u./MHz]", **opt)
axF.plot( 1e-6*tm_gd.F, 1e6*val_asd, ".r")
axF.grid()
axF.set_xlim(-1e-8*tm_gd.f_max,1.02e-6*tm_gd.f_max)
plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.2) Shape : circular piezo-electric transducer
# II.2.a/ Normal excitation
transducer_width = 10.0e-3 # 10 mm
smooth_width = 5.0e-3      #  5 mm
def smooth_door(x, w=transducer_width, dw=smooth_width) :
    ax_L = np.sqrt(20)*(x+0.5*w)/dw
    ax_R = np.sqrt(20)*(x-0.5*w)/dw
    val_L = 0.5*(1+erf(np.clip(ax_L,-5,5)))
    val_R = 0.5*(1+erf(np.clip(ax_R,-5,5)))
    val_L = (ax_L>-5)*( (ax_L<5)*val_L + (ax_L>=5)*1 )
    val_R = (ax_R>-5)*( (ax_R<5)*val_R + (ax_R>=5)*1 )
    return  (val_L - val_R)
# Space grid (2d)
sp_gd = my_computation.space_grid
# X grid
x_gd = sp_gd.x_grid
# Values
sd_val = smooth_door( x_gd.Xc )
# Zero-padding
szpg, sd_zp = x_gd.zero_padding(sd_val, 9, centered=True)
# 1D Fourier Transform
nk = x_gd.n_max +1
Kx = x_gd.K[:nk]
sd_asd = 2*np.abs(x_gd.fft(x_gd.cent2sort(sd_val))[:nk])
# 2D plot
fig_shp = plt.figure("Normal excitation", figsize=(13.8,7.3))
(axX, axK), (axXY, Zoom) = fig_shp.subplots(2,2)
fig_shp.subplots_adjust(0.06,0.08,0.94,0.94,0.25,0.4)
axX.set_title("Shape function", **opt)
axX.set_xlabel("Space $x$ [mm]", **opt)
axX.set_ylabel("Shape values [a.u.]", **opt)
axX.plot( 1e3*szpg.Xc, sd_zp, "-g")
axX.plot( 1e3*x_gd.Xc, sd_val, ".m")
xmx_mm = 17.0
axX.set_xlim( -xmx_mm, xmx_mm )
axX.grid()
axK.set_title("Spatial amplitude spectrum", **opt)
axK.set_xlabel("Wavenumber $k_x$ [$/$mm]", **opt)
axK.set_ylabel(r"ASD values [a.u.$\times$mm]", **opt)
axK.plot( 1e-3*Kx, 1e3*sd_asd, "-m")
axK.set_xlim( -1e-5*x_gd.k_max, 1.02e-3*x_gd.k_max )
axK.grid()
# plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# XY grid and zero-padding
MX,MY = sp_gd.MX_MY
Fz = smooth_door( np.sqrt( MX**2 + MY**2 ) )
axXY.set_title("Normal excitation (full range)", **opt)
axXY.set_xlabel("Position $x$ [mm]", **opt)
axXY.set_ylabel("Position $y$ [mm]", **opt)
axXY.grid()
sp_gd_mm = Space2DGrid(sp_gd.nx, sp_gd.ny, 1e3*sp_gd.dx, 1e3*sp_gd.dy)
sp_gd_mm.plot(Fz, draw_axis=axXY)
# Zoom with zero-padding
zp_gd, ZFz = sp_gd.zero_padding( Fz, cx=5, cy=5 )
Zoom.set_title("Normal excitation (zoom with zero-padding)", **opt)
Zoom.set_xlabel("Position $x$ [mm]", **opt)
Zoom.set_ylabel("Position $y$ [mm]", **opt)
Zoom.grid()
zp_gd_mm = Space2DGrid(zp_gd.nx, zp_gd.ny, 1e3*zp_gd.dx, 1e3*zp_gd.dy)
zp_gd_mm.plot(ZFz, draw_axis=Zoom)
Zoom.set_xlim(-12.4,12.4)
Zoom.set_ylim(-7.4,7.4)
#plt.show()
# II.2) Shape : circular piezo-electric transducer
# II.2.b/ tangential excitation (breathing)
# Values
T_val = smooth_door( x_gd.Xc ) * x_gd.Xc * 2 / transducer_width
# Zero-padding
_, T_zp = x_gd.zero_padding(T_val, 9, centered=True)
# 1D Fourier Transform
T_asd = 2*np.abs(x_gd.fft(x_gd.cent2sort(T_val))[:nk])
# 2D plot
fig_shp2 = plt.figure("Tangential excitation", figsize=(13.6,7.2))
(axX2, axK2), (ZoomFx, ZoomFy) = fig_shp2.subplots(2,2)
fig_shp2.subplots_adjust(0.06,0.08,0.94,0.94,0.25,0.4)
axX2.set_title("Shape function", **opt)
axX2.set_xlabel("Space $x$ [mm]", **opt)
axX2.set_ylabel("Shape values [a.u.]", **opt)
axX2.plot( 1e3*szpg.Xc, T_zp, "-g")
axX2.plot( 1e3*x_gd.Xc, T_val, ".m")
axX2.set_xlim( -xmx_mm, xmx_mm )
axX2.grid()
axK2.set_title("Spatial amplitude spectrum", **opt)
axK2.set_xlabel("Wavenumber $k_x$ [$/$mm]", **opt)
axK2.set_ylabel(r"ASD values [a.u.$\times$mm]", **opt)
axK2.plot( 1e-3*Kx, 1e3*T_asd, "-m")
axK2.set_xlim( -1e-5*x_gd.k_max, 1.02e-3*x_gd.k_max )
axK2.grid()
#plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# XY grid and zero-padding
Fx = Fz * MX * 2 / transducer_width
Fy = Fz * MY * 2 / transducer_width
_, ZFx = sp_gd.zero_padding( Fx, cx=5, cy=5 )
ZoomFx.set_title("Tangential excitation in the $x$-direction", **opt)
ZoomFx.set_xlabel("Position $x$ [mm]", **opt)
ZoomFx.set_ylabel("Position $y$ [mm]", **opt)
zp_gd_mm.plot(ZFx, draw_axis=ZoomFx)
ZoomFx.grid()
ZoomFx.set_xlim(-12.4,12.4)
ZoomFx.set_ylim(-7.4,7.4)
# Zoom with zero-padding
_, ZFy = sp_gd.zero_padding( Fy, cx=5, cy=5 )
ZoomFy.set_title("Tangential excitation in the $y$-direction", **opt)
ZoomFy.set_xlabel("Position $x$ [mm]", **opt)
ZoomFy.set_ylabel("Position $y$ [mm]", **opt)
ZoomFy.grid()
zp_gd_mm.plot(ZFy, draw_axis=ZoomFy)
ZoomFy.set_xlim(-12.4,12.4)
ZoomFy.set_ylim(-7.4,7.4)
#plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.3) Storage of the excitations:
#       tab_exc (numpy.ndarray) with it_min, it_max,
#                                  [ iy_min, iy_max, ]
#                                    ix_min, ix_max
epsilon = 1e-6
tbools = np.abs( val_sig )
tbools = (tbools > epsilon*tbools.max())
it_min, it_max = tbools.argmax(), tm_gd.nt - tbools[::-1].argmax()
Vsig = val_sig[it_min:it_max]
#plt.plot(1e6*tm_gd.T[it_min:it_max], Vsig, "dr")
#plt.grid() ; plt.show()

Lexc = []
nx, ny = sp_gd.nx, sp_gd.ny
for F in Fx, Fy, Fz :
    F = sp_gd.sort2cent(F)
    yxbools = np.abs( F )
    yxbools = (yxbools > epsilon*yxbools.max())
    xbools = yxbools.any(axis=0)
    ix_min, ix_max = xbools.argmax(), nx - xbools[::-1].argmax()
    ybools = yxbools.any(axis=1)
    iy_min, iy_max = ybools.argmax(), ny - ybools[::-1].argmax()
    Lexc.append( ( np.einsum("i,jk->ijk" , Vsig,
                             F[iy_min:iy_max, ix_min:ix_max]), \
                   (it_min, it_max, iy_min, iy_max, ix_min, ix_max) ) )
    #plt.imshow(F[iy_min:iy_max, ix_min:ix_max], cmap="seismic", \
    #           vmin=-1, vmax=1)
    #plt.show()
for (T,ranges),S in zip(Lexc,["Fx", "Fy", "Fz"]) :
    print(f"'{S}': T.nbytes: {T.nbytes/2**10:.1f} kB, {ranges}")
#=========================================================================
# III. Init field computation
tang = ( (0, Lexc[0][1], Lexc[0][0]), (1, Lexc[1][1], Lexc[1][0]))
my_computation.setFieldComputation("Tangential", tang)
norm = ( (2, Lexc[2][1], Lexc[2][0]), )
my_computation.setFieldComputation("Normal", norm)
plt.show()
#=========================================================================
