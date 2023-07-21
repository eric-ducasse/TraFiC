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
number_of_cycles = 3
central_frequency = 0.4e6 # 400 kHz
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
#plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.2) Shape : ``smooth door function''
transducer_width = 20.0e-3 # 20 mm
smooth_width = 4.0e-3      #  4 mm
def smooth_door(x, w=transducer_width, dw=smooth_width) :
    ax_L = np.sqrt(20)*(x+0.5*w)/dw
    ax_R = np.sqrt(20)*(x-0.5*w)/dw
    val_L = 0.5*(1+erf(np.clip(ax_L,-5,5)))
    val_R = 0.5*(1+erf(np.clip(ax_R,-5,5)))
    val_L = (ax_L>-5)*( (ax_L<5)*val_L + (ax_L>=5)*1 )
    val_R = (ax_R>-5)*( (ax_R<5)*val_R + (ax_R>=5)*1 )
    return  (val_L - val_R)
# Space grid (1d)
sp_gd = my_computation.space_grid
# Values
sd_val = smooth_door( sp_gd.Xc )
# Zero-padding
szpg, sd_zp = sp_gd.zero_padding(sd_val, 9, centered=True)
# Fourier Transform
nk = sp_gd.n_max +1
Kx = sp_gd.K[:nk]
sd_asd = 2*np.abs(sp_gd.fft(sp_gd.sort2cent(sd_val))[:nk])
# Space, wavenumber and excitation plots
fig_shp = plt.figure("Space, wavenumber and excitation plots", \
                     figsize=(13.8,7.3))
ax_dict2 = fig_shp.subplot_mosaic( \
    """AC
       BC""")
axX, axK, axXT = [ ax_dict2[c] for c in "ABC" ]
fig_shp.subplots_adjust(0.06,0.08,0.94,0.94,0.25,0.4)
axX.set_title("Shape function", **opt)
axX.set_xlabel("Space $x$ [mm]", **opt)
axX.set_ylabel("Shape values [a.u.]", **opt)
axX.plot( 1e3*szpg.Xc, sd_zp, "-g")
axX.plot( 1e3*sp_gd.Xc, sd_val, ".m")
xmx_mm = 17.0
axX.set_xlim( -xmx_mm, xmx_mm )
axX.grid()
axK.set_title("Spatial amplitude spectrum", **opt)
axK.set_xlabel("Wavenumber $k_x$ [$/$mm]", **opt)
axK.set_ylabel(r"ASD values [a.u.$\times$mm]", **opt)
axK.plot( 1e-3*Kx, 1e3*sd_asd, ".m")
axK.set_xlim( -1e-5*sp_gd.k_max, 1.02e-3*sp_gd.k_max )
axK.grid()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.3) Excitation:
##          XY_func(x,t) = smooth_door(x) Ncycle_pulse(t - d_by_x*x)
delay_by_x = 6.7e-4 # 0.67 µs/mm
def XY_func(x, t, d_by_x=delay_by_x) :
    return smooth_door(x)*Ncycle_pulse(t - d_by_x*x)
x_range_mm = np.linspace(-xmx_mm, xmx_mm, 201 )
dxs2 = 0.5 * (x_range_mm[1] - x_range_mm[0] )
xmn_mm, xmx_mm = x_range_mm[0]-dxs2,x_range_mm[-1]+dxs2
t_range_mus = np.linspace(-13.0, 17.0, 501 )
dts2 = 0.5 * (t_range_mus[1] - t_range_mus[0] )
tmn_mus, tmx_mus = t_range_mus[0]-dts2,t_range_mus[-1]+dts2
Xm,Ts = np.meshgrid(1e-3*x_range_mm, 1e-6*t_range_mus)
tab_exc = XY_func(Xm,Ts)
# Plot
axXT.set_xlabel("Space $x$ [mm]", **opt)
axXT.set_ylabel("Time $t$ [µs]", **opt)
axXT.set_title("Excitation function", **opt)
im = axXT.imshow(tab_exc, cmap="seismic", vmin=-1.0, vmax=1.0, \
                 aspect="auto", extent=(xmn_mm, xmx_mm,tmx_mus, tmn_mus) )
dvd = make_axes_locatable(axXT)
cax = dvd.append_axes('right', size='2%', pad=0.06)
plt.colorbar(im, cax=cax, extend="both")
axXT.grid()
#plt.show()
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# II.4) Storage of the excitation:
#       tab_exc (numpy.ndarray) with it_min, it_max,
#                                    ix_min, ix_max [,
#                                    iy_min, iy_max ]
epsilon = 1e-6
nx = sd_val.shape[0]
xbools = np.abs( sd_val )
xbools = (xbools > epsilon*xbools.max())
ix_min, ix_max = xbools.argmax(), nx - xbools[::-1].argmax()
tbools = np.array( [ np.abs(Ncycle_pulse( \
                             tm_gd.T - delay_by_x*sp_gd.Xc[ix_min] ) ),
                     np.abs(Ncycle_pulse( \
                             tm_gd.T - delay_by_x*sp_gd.Xc[ix_max-1] ) )
                    ] ).max(axis=0)
tbools = (tbools > epsilon*tbools.max())
nt = tm_gd.nt
it_min, it_max = tbools.argmax(), nt - tbools[::-1].argmax()
print(f"t range [{it_min}:{it_max}], x range [{ix_min}:{ix_max}]")
MX,MT = np.meshgrid(sp_gd.Xc[ix_min:ix_max], tm_gd.T[it_min:it_max])
tab_exc = XY_func(MX, MT).astype(np.float32)
print("tab_exc.nbytes:", tab_exc.nbytes)
#=========================================================================
# III. Init field computation
excit = ( (0,(it_min,it_max,ix_min,ix_max), tab_exc), )
my_computation.setFieldComputation("Demo_Fx", excit)
my_computation("Demo_Fx").plot_excitations(1.0, False)
plt.show()
#=========================================================================
