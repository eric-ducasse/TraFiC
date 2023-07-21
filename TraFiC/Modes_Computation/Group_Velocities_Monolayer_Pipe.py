# Version 1.0 / 2019, January, 10
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution : Arts & Metiers ParisTech / I2M
# ====== Initialization ==================================================
import numpy as np
import matplotlib.pyplot as plt
import sys
TraFiCpath = "H:\Recherche\TraFiC"
sys.path.append(TraFiCpath)
if __name__ == "__main__" : import TraFiC_init
# ====== Material Classes ================================================
from MaterialClasses import IsotropicElasticSolid
from Modes_Monolayer_Pipe import ModesMonolayerPipe
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
FMIN,FMAX = 75.0e3,115.0e3
KMIN,KMAX = 40,240
NBFD = 30
MODES = ( (0,2), (1,2), (4,1) ) # (n,rk)
SUBDIV = 10 # subdivision for each iteration
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def K_F_mode(n,rk,cL,cT,rho=2703.0,k_min=KMIN,k_max=KMAX,nbk=38,\
             r_min=28.0e-3,r_max=30.0e-3) :
    prm = {"cL" : cL,"cT" : cT,"rho" : rho }
    material = IsotropicElasticSolid(prm,"Aluminum")
    K = np.linspace(k_min,k_max,nbk+1)
    modes = ModesMonolayerPipe(material,r_min,r_max, K, NBFD, n)
    return K,modes.F[:,rk]
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def dispersion_point(frequency,nb_iter,n,rk,cL,cT,rho=2703.0, \
                     k_min=KMIN,k_max=KMAX,nbk=40,\
                     r_min=28.0e-3,r_max=30.0e-3) :
    K,F = K_F_mode(n,rk,cL,cT,rho,k_min,k_max,nbk,r_min,r_max)
    for i in range(nb_iter) :
        idx = np.argmin( abs(F-frequency) )
        K,F = K_F_mode(n,rk,cL,cT,rho,K[idx-1],K[idx+1],SUBDIV,r_min,r_max)
    idx = np.argmin( abs(F-frequency) )
    k0,dk = K[idx],K[idx]-K[idx-1]
    Fm1,F0,Fp1 = F[idx-1:idx+2]
    b,c,p = frequency/F0-1,(Fm1-2*F0+Fp1)/F0,0.5*(Fp1-Fm1)/F0
    if abs(c) > 1e-6 :
        if p > 0 : a = (p-np.sqrt(p**2+2*b*c))/c
        else : a = (p+np.sqrt(p**2+2*b*c))/c
    else : a = -b/p * ( 1 - b*c/(2*p**2) )
    k_c_app = k0 - a*dk
    prm = {"cL" : cL,"cT" : cT,"rho" : rho }
    material = IsotropicElasticSolid(prm,"Aluminum")
    modes = ModesMonolayerPipe(material,r_min,r_max,\
                               np.array([k_c_app]), NBFD, n)
    F_c_app = modes.F[0,rk]
    Vg = 2*np.pi*F0/dk * (p-a*c)
    return F_c_app,k_c_app,Vg,F_c_app-frequency,dk
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    import time
    cL,cT = 6320.0,3130.0
    K,F0 = K_F_mode(0,2,cL,cT)
    _,F1 = K_F_mode(1,2,cL,cT)
    _,F4 = K_F_mode(4,1,cL,cT)
    n0 = np.where( (F0>=FMIN)&(F0<=FMAX) )[0]
    n1 = np.where( (F1>=FMIN)&(F1<=FMAX) )[0]
    n4 = np.where( (F4>=FMIN)&(F4<=FMAX) )[0]
    k_min = K[min(n0.min(),n1.min(),n4.min())]
    k_max = K[max(n0.max(),n1.max(),n4.max())]
    LP0,LP1,LP4 = [],[],[]
    central_frequencies = [80e3,90e3,100e3,110e3]
    beg = time.time()
    print("Computation in progress. Wait please...")
    nb_iter = 2
    for fc in central_frequencies :
        LP0.append(dispersion_point(fc,nb_iter,0,2,cL,cT))
        LP1.append(dispersion_point(fc,nb_iter,1,2,cL,cT))
        LP4.append(dispersion_point(fc,nb_iter,4,1,cL,cT))
    dur = time.time()-beg
    print("... done in ~ {:.1f} seconds.".format(dur))
    msg = 51*"*"+"\n"
    msg += ("* Vg for cL = {:.2f} m/ms and cT = {:.2f} m/ms  *\n"\
            ).format(cL,cT)
    msg += 51*"*"+"\n"
    msg += "* fc [kHz] * mode (0,2) * mode (1,2) * mode (4,1) *\n"
    msg += 51*"*"+"\n"
    for fc,p0,p1,p4 in zip(central_frequencies,LP0,LP1,LP4) :
        msg += ("* {:8.4f} * {:9.2f}  * {:9.2f}  * {:9.2f}  *\n"\
                ).format(fc*1e-3,p0[2],p1[2],p4[2])
        msg += 51*"*"+"\n"
    msg += ("Maximal error on central frequencies ~ {:.2e} Hz\n"\
            ).format(max( \
            [abs(p[3]) for LP in [LP0,LP1,LP4] for p in LP ]))
    msg += ("Wavenumber step ~ {:.2e} mm^-1\n"\
            ).format(1e-3*max( \
            [p[4] for LP in [LP0,LP1,LP4] for p in LP ]))
    msg += ("Number of subintervals with respect to radius: {}"\
            ).format(NBFD)
    print(msg)
    plt.figure("Dispersion curves at f = {:.2} kHz".format(fc*1e-3),
                figsize=(8,6))
    for fc in central_frequencies :
        plt.plot([fc*1e-3,fc*1e-3],[KMIN*1e-3,KMAX*1e-3],"-b")
    plt.plot(F0*1e-3,K*1e-3,".r",label="(0,2)")
    plt.plot([p[0]*1e-3 for p in LP0],[p[1]*1e-3 for p in LP0],"Dr")
    deltaf = 2e3
    for p in LP0 :
        fc,kc,vg = p[:3]
        deltak = 2*np.pi*deltaf / vg
        plt.plot([(fc-deltaf)*1e-3,(fc+deltaf)*1e-3],\
                 [(kc-deltak)*1e-3,(kc+deltak)*1e-3],"-r")
    plt.plot(F1*1e-3,K*1e-3,".m",label="(1,2)")
    plt.plot([p[0]*1e-3 for p in LP1],[p[1]*1e-3 for p in LP1],"Dm")
    for p in LP1 :
        fc,kc,vg = p[:3]
        deltak = 2*np.pi*deltaf / vg
        plt.plot([(fc-deltaf)*1e-3,(fc+deltaf)*1e-3],\
                 [(kc-deltak)*1e-3,(kc+deltak)*1e-3],"-m")
    plt.plot(F4*1e-3,K*1e-3,".g",label="(4,1)")
    plt.plot([p[0]*1e-3 for p in LP4],[p[1]*1e-3 for p in LP4],"Dg")
    for p in LP4 :
        fc,kc,vg = p[:3]
        deltak = 2*np.pi*deltaf / vg
        plt.plot([(fc-deltaf)*1e-3,(fc+deltaf)*1e-3],\
                 [(kc-deltak)*1e-3,(kc+deltak)*1e-3],"-g")
    plt.legend(loc="best")
    plt.xlim(FMIN*1e-3,FMAX*1e-3)
    plt.ylim(k_min*1e-3,k_max*1e-3)
    plt.xlabel("Frequency [$kHz$]",size=14,family="Times New Roman")
    plt.ylabel("Wavenumber [$mm^{-1}$]",size=14,family="Times New Roman")
    plt.grid()
    plt.show()
