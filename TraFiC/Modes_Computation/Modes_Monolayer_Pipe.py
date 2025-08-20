# Version 1.02 / 2025, August, 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
# ====== Initialization ==================================================
import numpy as np
if np.__version__[0] == "1":
    solve = np.linalg.solve
elif np.__version__[0] == "2": # solve must be redefined with numpy 2.x
    def solve(A, b):
        shp_M = A.shape
        shp_V = b.shape
        matrices_vectors =  len(shp_M) == len(shp_V)+1
        if matrices_vectors: b.shape = shp_V+(1,)
        X = np.linalg.solve(A, b)
        if matrices_vectors:
            b.shape = shp_V
            X.shape = X.shape[:-1]
        return X
else:
    raise SystemError(f"Unknown '{np.__version__}' of numpy.")
import matplotlib.pyplot as plt
import sys, os
if __name__ == "__main__" :
    TraFiCpath = ".."
    sys.path.append( os.path.abspath(TraFiCpath) )
    import TraFiC_init
# ====== Material Classes ================================================
from MaterialClasses import *
import scipy.sparse as sprs
import os
np.set_printoptions(precision=3,suppress=True) # for printing of ndarrays
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
#==   DiscretizedCylindricalLayer                                        ==
#==========================================================================
class DiscretizedCylindricalLayer:
    """ DiscretizedCylindricalLayer of radii between Rmin and Rmax,
        discretized into n sub-intervals of thickness (Rmax-Rmin)/nb.
        'fds' is the Finite Difference Scheme """
    def __init__(self,material,Rmin,Rmax,nb,fds) :
        if isinstance(material,IsotropicElasticSolid) :
            self.__mat = material.export().export()
        elif isinstance(material,TransverselyIsotropicElasticSolid) :
            self.__mat = material.export()
        elif isinstance(material,AnisotropicElasticSolid) :
            self.__mat = material
        else :
            print("[DiscretizedCylindricalLayer::DiscretizedCylindrical"+\
                  "Layer]\nMaterial of type "+\
                  "'{}' cannot be taken into account".format(
                                             type(material).__name__))
        if nb < fds.order + 2 : nb = fds.order + 2
        # radial discretization
        self.__R = np.linspace(Rmin,Rmax,nb+1)
        self.__dr = (Rmax-Rmin)/nb
        unsdr = 1.0/self.__dr
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
        A2 = -lol*unsdr*unsdr
        # First derivative
        self.__FDMK = sprs.lil_matrix( (size,size+6) ) # * i*k
        self.__FDMN = sprs.lil_matrix( (size,size+6) ) # * i*n
        A10 = -(lol+lom.dot(T)+T.dot(mol))*unsdr
        A1K = (lon+nol)*unsdr
        A1N = (lom+mol)*unsdr
        # off-centered finite differences
        for i in range(os2-1) :
            timi = 3*i ; tima = timi + 3
            tipa = size - timi ; tipi = tipa - 3
            unsrint = 1.0/self.__R[i+1]
            A10sri = unsrint*A10
            A1Nsri = unsrint*A1N
            unsrext = 1.0/self.__R[-i-2]
            A10sre = unsrext*A10
            A1Nsre = unsrext*A1N
            for j in range(fds.order+1) :
                tjmi = 3*j ; tjma = tjmi + 3
                tjpa = size + 6 - tjmi ; tjpi = tjpa - 3
                self.__FDM0[timi:tima,tjmi:tjma] = \
                            fds.D2[i+1,j]*A2 + fds.D1[i+1,j]*A10sri
                self.__FDM0[tipi:tipa,tjpi:tjpa] = \
                            fds.D2[-i-2,-j-1]*A2 + fds.D1[-i-2,-j-1]*A10sre
                self.__FDMK[timi:tima,tjmi:tjma] = fds.D1[i+1,j]*A1K
                self.__FDMK[tipi:tipa,tjpi:tjpa] = fds.D1[-i-2,-j-1]*A1K
                self.__FDMN[timi:tima,tjmi:tjma] = fds.D1[i+1,j]*A1Nsri
                self.__FDMN[tipi:tipa,tjpi:tjpa] = fds.D1[-i-2,-j-1]*A1Nsre
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
        R10 = np.ndarray( (3,nw) )
        R1K = np.ndarray( (3,nw) )
        R1N = np.ndarray( (3,nw) )
        for j in range(fds.order+1) :
            tjmi = 3*j ; tjma = tjmi + 3
            R2[:,tjmi:tjma] = fds.D2[os2,j]*A2
            R10[:,tjmi:tjma] = fds.D1[os2,j]*A10
            R1K[:,tjmi:tjma] = fds.D1[os2,j]*A1K
            R1N[:,tjmi:tjma] = fds.D1[os2,j]*A1N
        for i in range(i0,nb-1-i0) :
            ti = 3*i
            tj = ti - ti0
            unsr = 1.0/self.__R[i+1]
            self.__FDM0[ti:ti+3,tj:tj+nw] = R2 + R10*unsr
            self.__FDMK[ti:ti+3,tj:tj+nw] = R1K.copy()
            self.__FDMN[ti:ti+3,tj:tj+nw] = R1N*unsr
        # Without derivative
        self.__FDMKN = sprs.lil_matrix( (size,size+6) ) # * k*n
        self.__FDMK2 = sprs.lil_matrix( (size,size+6) ) # * k*k
        self.__FDMN2 = sprs.lil_matrix( (size,size+6) ) # * n*n
        A0K = lon + T.dot(mon) + nom.dot(T)
        A0KN = mon + nom
        A00 = -T.dot(mom).dot(T)
        A0N = T.dot(mom) + mom.dot(T)
        for j in range(1,nb) :
            tj = 3*j
            unsr = 1.0/self.__R[j]
            unsr2 = unsr*unsr
            self.__FDMK2[tj-3:tj,tj:tj+3] = non
            self.__FDMK[tj-3:tj,tj:tj+3] += A0K*unsr
            self.__FDMKN[tj-3:tj,tj:tj+3] = A0KN*unsr
            self.__FDMN2[tj-3:tj,tj:tj+3] = mom*unsr2
            self.__FDMN[tj-3:tj,tj:tj+3] += A0N*unsr2
            self.__FDM0[tj-3:tj,tj:tj+3] += A00*unsr2
        # Radial stresses on the boudaries
        LoL = self.material.lol
        LoLsdr = LoL*unsdr
        LoM = self.material.lom
        LoN = self.material.lon
        self.__SrInt0 = np.ndarray( (3,nw) ) # constant
        self.__SrIntN = -LoM / Rmin          # * i*n
        self.__SrIntK = -LoN                 # * i*k
        self.__SrExt0 = np.ndarray( (3,nw) ) # constant
        self.__SrExtN = -LoM / Rmax          # * i*n
        self.__SrExtK = -LoN                 # * i*k
        for j in range(fds.order+1) :
            tjmi = 3*j ; tjma = tjmi + 3
            self.__SrInt0[:,tjmi:tjma] = fds.D1[0,j]*LoLsdr
            self.__SrExt0[:,tjmi:tjma] = fds.D1[-1,j]*LoLsdr
        self.__SrInt0[:,:3] += LoM.dot(T) / Rmin
        self.__SrExt0[:,-3:] += LoM.dot(T) / Rmax
        
    @property
    def material(self) : return self.__mat
    @property
    def radii(self) : return self.__R
    @property
    def FDM0(self) : return self.__FDM0
    @property
    def FDMK(self) : return self.__FDMK
    @property
    def FDMN(self) : return self.__FDMN
    @property
    def FDMK2(self) : return self.__FDMK2
    @property
    def FDMN2(self) : return self.__FDMN2
    @property
    def FDMKN(self) : return self.__FDMKN
    @property
    def SrInt0(self) : return self.__SrInt0
    @property
    def SrExt0(self) : return self.__SrExt0
    @property
    def SrIntN(self) : return self.__SrIntN
    @property
    def SrExtN(self) : return self.__SrExtN
    @property
    def SrIntK(self) : return self.__SrIntK
    @property
    def SrExtK(self) : return self.__SrExtK
#==========================================================================
#==   ModesMonolayerPipe                                                 ==
#==========================================================================
class ModesMonolayerPipe :
    def __init__(self,material,Rmin,Rmax,K,nb,n=0,order=8,verbose=False) :
        self.__fds = FDScheme(order)
        self.__mat = material
        self.__rmin = Rmin            # Inner Radius
        self.__rmax = Rmax            # Outer Radius
        self.__nb = nb                # Number of subintervals
        self.__K = np.array(K).copy() # Axial wavenumbers 
        self.__n = n                  # Azimuthal wavenumber 
        layer1 = DiscretizedCylindricalLayer(\
                  material,Rmin,Rmax,nb,self.fds)
        U = np.ones_like(K)
        N = n * U
        FDMA = np.multiply.outer(U,layer1.FDM0.toarray()) + 1j * ( \
               np.multiply.outer(K,layer1.FDMK.toarray()) + \
               np.multiply.outer(N,layer1.FDMN.toarray()) ) + \
               np.multiply.outer(K*K,layer1.FDMK2.toarray()) + \
               np.multiply.outer(K*N,layer1.FDMKN.toarray()) + \
               np.multiply.outer(N*N,layer1.FDMN2.toarray())
        # Zero radial stress
        # Inner boundary
        M = np.multiply.outer(U,layer1.SrInt0[:,:3]) + 1j * ( \
            np.multiply.outer(N,layer1.SrIntN) + \
            np.multiply.outer(K,layer1.SrIntK) )
        R = np.linalg.solve(M,-np.multiply.outer(U,layer1.SrInt0[:,3:]))
        nL = 3*self.fds.order//2
        nC = R.shape[2]
        SInt = np.einsum("ijk,ikl->ijl",FDMA[:,:nL,:3],R)
        # Outer boundary
        M = np.multiply.outer(U,layer1.SrExt0[:,-3:]) + 1j * ( \
            np.multiply.outer(N,layer1.SrExtN) + \
            np.multiply.outer(K,layer1.SrExtK) )
        R = np.linalg.solve(M,-np.multiply.outer(U,layer1.SrExt0[:,:-3]))
        SExt = np.einsum("ijk,ikl->ijl",FDMA[:,-nL:,-3:],R)
        FDMA = FDMA[:,:,3:-3]
        FDMA[:,:nL,:nC] += SInt
        FDMA[:,-nL:,-nC:] += SExt
        # eigenfrequencies
        VPA=np.linalg.eigvals(FDMA)
        if verbose :
            if not (0.0 < VPA.real).all() :
                print("ModesMonolayerPipe::Warning: "+
                      "eigenvalues with negative real part")
            if not (abs(VPA.imag) < VPA.real*1e-10).all() :
                print("ModesMonolayerPipe::Warning: eigenvalues with "+
                      "non-neglictible imaginary part")
        self.__f = np.sqrt(VPA.real)*0.5/np.pi
        self.__f.sort(1)
    @property
    def fds(self) : return self.__fds
    @property
    def Material(self) : return self.__mat
    @property
    def Rmin(self) : return self.__rmin
    @property
    def Rmax(self) : return self.__rmax
    @property
    def nb(self) : return self.__nb
    @property
    def F(self) : return self.__f
    @property
    def K(self) : return self.__K
    
#==========================================================================
if __name__ == "__main__" :
    prm = {"cL" : 6320.0,"cT" : 3130.0,"rho" : 2700.0}
    alu = IsotropicElasticSolid(prm,"Aluminum")
    vecKz = np.arange(0.0,4000.0,20.0)
    modes = ModesMonolayerPipe(alu,0.09,0.092,vecKz,40)
    plt.figure("Dispersion curves",figsize=(12,12))
    for no in range(10) :
        plt.plot(1e-6*modes.F[:,no],1e-3*modes.K,".")
        plt.xlabel(r"Frequency $f$ [$MHz$]",fontsize=14)
        plt.ylabel(r"Axial wavenumber $k_z$ [$mm^{-1}$]",fontsize=14)
        plt.xlim(0,2.1)
        plt.grid()    
    plt.show()
