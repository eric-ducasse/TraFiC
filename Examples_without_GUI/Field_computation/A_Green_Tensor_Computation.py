# Version 1.0 - 2023, July, 7
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Imported modules
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
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
from SpaceGridClasses import Space1DGrid
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
duration_µs = 150.0
# Delay such that the excitation signal can be centered at t = 0
delay_µs = 30.0
print(f"Time Interval [{-delay_µs:.2f}, " + \
      f"{duration_µs-delay_µs:.2f}] in µs")
# Maximum frequency in MegaHertz:
max_frequency_MHz = 2.0
# Maximum distance between source(s) and observation point(s), in meters
max_length_m = 0.9
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
print(my_plate)
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
Vs = time_gd.S # Valeurs de la variable de Laplace s
# Time checking
if new_cal :
    np.save(os.path.join(execpath,"S.npy"),Vs)
else :
    rS = np.load(os.path.join(execpath,"S.npy"))
    if not np.allclose(rS,Vs) : print("Problem on S values")
# Space checking
Vk = space_gd.K # Valeurs de k_x
if new_cal :
    np.save(os.path.join(execpath,"K.npy"),Vk)
else :
    rK = np.load(os.path.join(execpath,"K.npy"))
    if not np.allclose(rK,Vk) : print("Problem on K values")
#++++++++++++++++++++++++++++
# Estimation de la taille mémoire pour chaque valeur de s
# Matrice M + Coefficients + Polarisations + Nombres d'onde
nbpw = my_plate.nbPartialWaves
nbc = my_plate.dim
SizeByS = 2**-30 * nx*(nbc*(nbc+1)+nbpw*7)*16
nbSmax = MAXMEM//SizeByS
if nbSmax == 0 :
    print(f"Problem : {SizeByS:.3e} GB for each s value")
    nbSmax = 1
if nbSmax < ns :
    nbpart = int(np.ceil(ns/nbSmax))
    dns = int(np.ceil(ns/nbpart))
    print(nbpart,"successive computations by sequences of",dns,\
          "values\n are necessary.")
    PS = [ [i-dns,i] for i in range(dns,ns,dns) ]
    PS.append([PS[-1][-1],ns])
else :
    nbpart = 1
    PS = [ [0,ns] ]
#++++++++++++++++++++++++++++
nsm1 = ns-1 # Indice de la fréquence de Nyquist (fs/2)
SRC_K = np.ones_like(Vk)
SRC_K[space_gd.n_max] = 0.0 # Coefficient nul pour le nombre
                                # d'onde de Nyquist
#++++ Lancement du calcul... ++++++++++++++++++++++++
for (g,d) in PS :
    print(f"{now()}")
    xt = f"{g+1:03d}-{d:03d}"
    plate_file = "Plate"+xt+".pckl"
    coef_file_Fx = "Coef"+xt+"_Fx.npy"
    coef_file_Fy = "Coef"+xt+"_Fy.npy"
    coef_file_Fz = "Coef"+xt+"_Fz.npy"
    if plate_file in previous_files and coef_file_Fx in previous_files \
      and coef_file_Fy in previous_files and \
      coef_file_Fz in previous_files :
        print("Computation already done.")
        continue
    # Linear system to be solved
    print(f"Computation of huge arrays [{g+1}:{d}]/{ns}...")
    beg = time.time()
    my_plate.update(Vs[g:d], Vk) # On charge les valeurs pour le calcul
    dur = time.time()-beg
    print(f"{now()} - ...done in {dur:.2e} s")
    print("Solving the linear systems...")
    beg = time.time()
    # Source
    SRC_S = (np.arange(g,d)!=nsm1)*1.0
    SRC = np.einsum("i,j->ij",SRC_S,SRC_K)
    B = np.zeros( (my_plate.M.shape)[:-1] , dtype = complex)
#                       Uz     Sxz     Syz     Szz
# Interface du haut   : 0       -       -       1
# Interface du milieu : 2       3       4       5
# Interface du bas    : 6       7       8       9
# On excite avec un saut de déplacement vertical
    B[...,0] = SRC 
    # Coefficients of the partial waves
    C = solve(my_plate.M,B)
    coef_file = "Coef"+xt+f"_Uz.npy"
    np.save(os.path.join(execpath,coef_file),C)
    dur = time.time()-beg
    print(f"{now()} - '{coef_file}' ...done in {dur:.2e} s")
    print(f"Saving the big file '{plate_file}'...")
    beg = time.time()
    my_plate.clearM() # Memory release
    with open(os.path.join(execpath,plate_file),"bw") as f :
        pickle.dump(my_plate,f)
    dur = time.time()-beg
    print(f"{now()} - ...done in {dur:.2e} s")
##########################################################################
###   LA SUITE EST TRAITÉE DANS LE FICHIER "B_Calcul _de_champ.py"
##########################################################################
