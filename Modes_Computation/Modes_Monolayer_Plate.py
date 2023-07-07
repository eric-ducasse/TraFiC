# Version 1.0 - 2017, July, 27
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
from numpy import multiply
outer = multiply.outer
import matplotlib.pyplot as plt
from numpy.linalg import solve
import scipy.sparse as sprs
import sys, os
np.set_printoptions(precision=3,suppress=True) # for printing of ndarrays
sys.path.append(os.path.dirname(os.getcwd()))
from MaterialClasses import *
#=========================================================================
#==   FDScheme                                                          ==
#=========================================================================
class FDScheme:
    """ Finite difference scheme of even order n """
    def __init__(self,order) :
        if order%2 == 1 :
            order += 1
            print("[FDScheme::FDScheme] Warning: odd order -> order",order)
        self.__order = order
        # First derivative
        n = self.order + 1
        M = np.zeros( (n,n) )
        puis = np.arange(n)
        B = np.zeros( (n,) )
        B[1] = 1.0
        D1 = []
        col0 = np.zeros( (n,) )
        col0[0] = 1.0
        for rd in range(n) :
            rangs = np.array(range(-rd,n-rd))
            for i,r in enumerate(rangs) :
                if r==0 : M[:,i] = col0
                else : M[:,i]= float(r)**puis
            D1.append(solve(M,B).round(14))
        self.__D1 = np.array(D1)
        # Second derivative
        n = self.order + 2
        M = np.zeros( (n,n) )
        puis = np.arange(n)
        B = np.zeros( (n,) )
        B[2] = 2.0
        D2 = []
        col0 = np.zeros( (n,) )
        col0[0] = 1.0
        for rd in range(n) :
            rangs = np.array(range(-rd,n-rd))
            for i,r in enumerate(rangs) :
                if r==0 : M[:,i] = col0
                else : M[:,i]= float(r)**puis
            
            D2.append(solve(M,B).round(14))
        self.__D2 = np.array(D2)
        # Derivatives of order greater than 2 are not taken into account
        # here
    @property
    def order(self) : return self.__order
    @property
    def D1(self) : return self.__D1
    @property
    def D2(self) : return self.__D2
#==========================================================================
#==   DiscretizedLayer                                        ==
#==========================================================================
class DiscretizedLayer:
    """ DiscretizedLayer of vertical position between Zmin and Zmax,
        discretized into n sub-intervals of thickness (Zmax-Zmin)/nb.
        'fds' is the Finite Difference Scheme """
    def __init__(self,material,thickness,nb,fds) :
        if isinstance(material,IsotropicElasticSolid) :
            self.__mat = material.export().export()
        elif isinstance(material,TransverselyIsotropicElasticSolid) :
            self.__mat = material.export()
        elif isinstance(material,AnisotropicElasticSolid) :
            self.__mat = material
        else :
            print("[DiscretizedLayer::DiscretizedLayer"+\
                  "]\nMaterial of type "+\
                  "'{}' cannot be taken into account".format(
                                             type(material).__name__))
        if nb < fds.order + 2 : nb = fds.order + 2
        # radial discretization
        self.__Z = np.linspace(0,thickness,nb+1)
        self.__dz = thickness/nb
        unsdz = 1.0/self.__dz
        T = np.array([[0,-1,0],[1,0,0],[0,0,0]])
        lol = self.material.lol/self.material.rho
        lom = self.material.lom/self.material.rho
        lon = self.material.lon/self.material.rho
        mol = self.material.mol/self.material.rho
        mom = self.material.mom/self.material.rho
        mon = self.material.mon/self.material.rho
        nol = self.material.nol/self.material.rho
        nom = self.material.nom/self.material.rho
        non = self.material.non/self.material.rho
        size = 3*nb-3
        os2 = fds.order//2
        self.__FDM0 = sprs.lil_matrix( (size,size+6) ) # constant
        # Second derivative
        A2 = -non*unsdz*unsdz
        # First derivative
        self.__FDMX = sprs.lil_matrix( (size,size+6) ) # * i*kx
        self.__FDMY = sprs.lil_matrix( (size,size+6) ) # * i*kx
        A1X = (lon+nol)*unsdz
        A1Y = (mon+nom)*unsdz
        # off-centered finite differences
        for i in range(os2-1) :
            timi = 3*i ; tima = timi + 3
            tipa = size - timi ; tipi = tipa - 3
            for j in range(fds.order+1) :
                tjmi = 3*j ; tjma = tjmi + 3
                tjpa = size + 6 - tjmi ; tjpi = tjpa - 3
                self.__FDM0[timi:tima,tjmi:tjma] = fds.D2[i+1,j]*A2
                self.__FDM0[tipi:tipa,tjpi:tjpa] = fds.D2[-i-2,-j-1]*A2
                self.__FDMX[timi:tima,tjmi:tjma] = fds.D1[i+1,j]*A1X
                self.__FDMX[tipi:tipa,tjpi:tjpa] = fds.D1[-i-2,-j-1]*A1X
                self.__FDMY[timi:tima,tjmi:tjma] = fds.D1[i+1,j]*A1Y
                self.__FDMY[tipi:tipa,tjpi:tjpa] = fds.D1[-i-2,-j-1]*A1Y
            j = fds.order+1
            tjmi = 3*j ; tjma = tjmi + 3
            tjpa = size + 6 - tjmi ; tjpi = tjpa - 3
            self.__FDM0[timi:tima,tjmi:tjma] = fds.D2[i+1,j]*A2
            self.__FDM0[tipi:tipa,tjpi:tjpa] = fds.D2[-i-2,-j-1]*A2
        # centered finite differences
        i0 = os2-1
        ti0 = 3*i0
        nw = 3*fds.order+3
        R2 = np.ndarray( (3,nw) )
        R1X = np.ndarray( (3,nw) )
        R1Y = np.ndarray( (3,nw) )
        for j in range(fds.order+1) :
            tjmi = 3*j ; tjma = tjmi + 3
            R2[:,tjmi:tjma] = fds.D2[os2,j]*A2
            R1X[:,tjmi:tjma] = fds.D1[os2,j]*A1X
            R1Y[:,tjmi:tjma] = fds.D1[os2,j]*A1Y
        for i in range(i0,nb-1-i0) :
            ti = 3*i
            tj = ti - ti0
            self.__FDM0[ti:ti+3,tj:tj+nw] = R2
            self.__FDMX[ti:ti+3,tj:tj+nw] = R1X.copy()
            self.__FDMY[ti:ti+3,tj:tj+nw] = R1Y.copy()
        # Without derivative
        self.__FDMX2 = sprs.lil_matrix( (size,size+6) ) # * kx**2
        self.__FDMXY = sprs.lil_matrix( (size,size+6) ) # * kx*ky
        self.__FDMY2 = sprs.lil_matrix( (size,size+6) ) # * ky**2
        A0X2 = lol
        A0XY = mol + lom
        A0Y2 = mom
        for j in range(1,nb) :
            tj = 3*j
            self.__FDMX2[tj-3:tj,tj:tj+3] = A0X2.copy()
            self.__FDMXY[tj-3:tj,tj:tj+3] = A0XY.copy()
            self.__FDMY2[tj-3:tj,tj:tj+3] = A0Y2.copy()
        # Radial stresses on the boudaries
        NoL = self.material.nol
        NoM = self.material.nom
        NoNsdz = self.material.non*unsdz
        self.__SzInf0 = np.ndarray( (3,nw) ) # constant
        self.__SzSup0 = np.ndarray( (3,nw) ) # constant
        self.__SzX = -NoL                    # * i*n
        self.__SzY = -NoM                    # * i*k
        for j in range(fds.order+1) :
            tjmi = 3*j ; tjma = tjmi + 3
            self.__SzInf0[:,tjmi:tjma] = fds.D1[0,j]*NoNsdz
            self.__SzSup0[:,tjmi:tjma] = fds.D1[-1,j]*NoNsdz
        
    @property
    def material(self) : return self.__mat
    @property
    def Z(self) : return self.__Z
    @property
    def FDM0(self) : return self.__FDM0
    @property
    def FDMX(self) : return self.__FDMX
    @property
    def FDMY(self) : return self.__FDMY
    @property
    def FDMX2(self) : return self.__FDMX2
    @property
    def FDMXY(self) : return self.__FDMXY
    @property
    def FDMY2(self) : return self.__FDMY2
    @property
    def SzInf0(self) : return self.__SzInf0
    @property
    def SzSup0(self) : return self.__SzSup0
    @property
    def SzX(self) : return self.__SzX
    @property
    def SzY(self) : return self.__SzY
#==========================================================================
#==   ModesMonolayerPlate                                                ==
#==========================================================================
class ModesMonolayerPlate :
    def __init__(self,material,thickness,nb,Kx,Ky=None,theta=0,order=8) :
        self.__fds = FDScheme(order)
        self.__mat = material
        self.__h = thickness            # Thickness
        self.__nb = nb                  # Number of subintervals
        # wavenumbers in the xy-plane
        self.__Kx = np.array(Kx).copy() 
        if Ky is None : 
            self.__Ky = np.sin(theta)*self.__Kx
            self.__Kx *= np.cos(theta)
        else : self.__Ky = np.array(Ky).copy() 
        layer1 = DiscretizedLayer(material,thickness,nb,self.fds)
        KX,KY = self.__Kx,self.__Ky
        U = np.ones_like(self.__Kx)
        FDMA = outer(U,layer1.FDM0.toarray()) + 1j * ( \
               outer(KX,layer1.FDMX.toarray()) + \
               outer(KY,layer1.FDMY.toarray()) ) + \
               outer(KX*KX,layer1.FDMX2.toarray()) + \
               outer(KX*KY,layer1.FDMXY.toarray()) + \
               outer(KY*KY,layer1.FDMY2.toarray())
        # Zero radial stress
        # Boundary at z=0
        M = outer(U,layer1.SzInf0[:,:3]) + 1j * ( \
            outer(KX,layer1.SzX) + outer(KY,layer1.SzY) )
        R = solve(M,-outer(U,layer1.SzInf0[:,3:]))
        nL = 3*self.fds.order//2
        nC = R.shape[2]
        SInf = np.einsum("ijk,ikl->ijl",FDMA[:,:nL,:3],R)
        # Boundary at z=thickness
        M = outer(U,layer1.SzSup0[:,-3:]) + 1j * ( \
            outer(KX,layer1.SzX) + outer(KY,layer1.SzY) )
        R = solve(M,-outer(U,layer1.SzSup0[:,:-3]))
        SSup = np.einsum("ijk,ikl->ijl",FDMA[:,-nL:,-3:],R)
        FDMA = FDMA[:,:,3:-3]
        FDMA[:,:nL,:nC] += SInf
        FDMA[:,-nL:,-nC:] += SSup
        # Eigen-frequencies 
        VPA=np.linalg.eigvals(FDMA)
        if not (0.0 < VPA.real).all() :
            print("ModesMonolayerPlate::Warning: "+
                  "eigenvalues with negative real part")
        if not (abs(VPA.imag) < VPA.real*1e-10).all() :
            print("ModesMonolayerPlate::Warning: eigenvalues with "+
                  "non-neglictible imaginary part")
        self.__f = np.sqrt(VPA.real)*0.5/np.pi
        self.__f.sort(1)
    @property
    def fds(self) : return self.__fds
    @property
    def Material(self) : return self.__mat
    @property
    def h(self) : return self.__h
    @property
    def nb(self) : return self.__nb
    @property
    def F(self) : return self.__f
    @property
    def K(self) : return np.sqrt(self.__Kx**2+self.__Ky**2)
    
#==========================================================================
if __name__ == "__main__" :
    simu_name = "Aluminum_plate_{:.1f}mm_dispersion_curves"
    alu = IsotropicElasticSolid({"rho":2700.0,"Young modulus":6.9e10,\
                                 "Poisson ratio":0.3},"Aluminum")
    vecK = np.arange(0.0,8000.0,20.0)
    epaisseur = 0.001
    simu_name = simu_name.format(1e3*epaisseur).replace(".","v")
    modes = ModesMonolayerPlate(alu,epaisseur,40,vecK)
    plt.figure("Dispersion curves",figsize=(12,8))
    nb_modes_to_save = 15
    for no in range(nb_modes_to_save) :
        plt.plot(1e-6*modes.F[:,no],1e-3*modes.K,".",markersize=3)
    plt.xlabel(r"Frequency $f$ [$MHz$]",fontsize=16)
    plt.ylabel(r"Horizontal wavenumber $k$ [$mm^{-1}$]",fontsize=16)
    plt.suptitle("Aluminum plate : $c_L="+"{:.3f}".format(1e-3*alu.cL)+\
              "\,mm\cdot\mu s^{-1}$, $c_T="+"{:.3f}".format(1e-3*alu.cT)+\
              r"\,mm\cdot\mu s^{-1}$, $\rho="+"{:.3f}".format(1e-3*alu.rho)+\
              "\,mg\cdot mm^{-3}$, $h="+"{:.1f}\,mm$".format(1e3*epaisseur),
                 fontsize=16)
    plt.xlim(0,6.8)
    plt.grid()
    plt.subplots_adjust(left=0.08,right=0.99,bottom=0.08,top=0.92)
    plt.savefig(simu_name+".png")
    tableau = np.empty( modes.K.shape+(1+nb_modes_to_save,) )
    tableau[:,0] = 1e-3*modes.K
    tableau[:,1:] = 1e-6*modes.F[:,:nb_modes_to_save]
    texte = "Wavenumber [mm^-1] / Frequencies [MHz]"
    for ligne in tableau :
        texte += "\n"
        for val in ligne :
            texte += "{:.5f}  ".format(val)
    with open(simu_name+".txt","w") as strm :
        strm.write(texte)
    plt.show()
