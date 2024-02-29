# Version 1.61 - 2023, July, 3rd
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
# ====== Initialization ====================================================
if __name__ == "__main__" :
    import os, sys
    sys.path.append( os.path.abspath("..") )
    import TraFiC_init 
# ====== Material Classes ==================================================
from MaterialClasses import *
# ====== Numerical tools ===================================================
import numpy as np
from numpy import sqrt, exp, pi, sin, cos, log, array, meshgrid, multiply
outer =  multiply.outer # (i1,...,in),(j1,...,jk) -> (i1,...,i2,j1,...jk)
from numpy.linalg import eig
#===========================================================================
class staticproperty(property):
    """ Creating the  '@staticproperty' decorator"""
    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()
#===========================================================================
class USMat :
    """Virtual mother class of all classes"""
    __ListOfOrMat = dict()
    __2powm20 = 2**-20
    def __init__(self, material, angles, USset=None) :
        self.__initial_mat = material
        self.__mat = material # by default, for fluid and vertical T.I.E.
        self.__IES = None # For "Isotropic Elastic Solid" case (saving
                          #   in file)
        self.__angles = angles
        self.__nbpw = None # Number of Partial Waves in each direction
        self.__Kz = None # Wave numbers of Partial Waves
        self.__P = None # Polarizations of Partial Waves
        self.__MMprod = None # Matrix-Matrix product for np.einsum
        self.__MVprod = None # Matrix-Vector product for np.einsum
        idset = id(USset)
        if idset in USMat.__ListOfOrMat.keys() : # existing USset
            USMat.__ListOfOrMat[idset].append(self)
        else : # new USset
            USMat.__ListOfOrMat[idset] = [self]
    @property
    def initial_material(self) : return self.__initial_mat
    @property
    def mat(self) : return self.__mat
    @property
    def mat_diff_IES(self) :
        if self.__IES is None : return self.__mat
        else : return self.__IES
    def _set_mat(self, material) : self.__mat = material
    def _set_IES_mat(self, material) :
        self.__IES = material
    @property
    def angles(self) : return self.__angles
    @property
    def nb_pw(self) :
        """ Number of partial waves in each direction."""
        return self.__nbpw
    def _set_nb_pw(self, nb) : self.__nbpw = nb
    @property
    def Kz(self) : return self.__Kz
    def _set_Kz(self, Kz) : self.__Kz = Kz
    @property
    def P(self) : return self.__P
    def _set_P(self, P) : self.__P = P   
    @property
    def MMprod(self) : return self.__MMprod
    def _set_MMprod(self, MMprod) : self.__MMprod = MMprod   
    @property
    def MVprod(self) : return self.__MVprod
    def _set_MVprod(self, MVprod) : self.__MVprod = MVprod  
    @property
    def nMBytes(self) :
        if self.Kz is None : # and self.P is None too
            return 0.0
        else :
            return USMat.__2powm20*(self.Kz.nbytes + self.P.nbytes) 
    @property
    def nBytes_max_per_value(self) :
        nb  = self.nb_pw*224   # Kz vector containing np.complex128
             # 224 = 2*16*(1+6)  and polarization matrix
        return nb
    def update(self, S, Kx, Ky=None) :
        """ Should not be called """
        print("Error : USMat::update called by a "+\
              "'{}' instance".format(type(self).__name__))
        
    def fieldComponents(self,dzup,dzdown) :
        """ returns PW ( shape=(ns,nx,[ny,],6,2*self.nb_pw) ) 
            such that U = np.einsum(self.MVprod,PW,a), where a 
            denotes the vector of the coefficients of the partial
            waves ( shape=(ns,nx,[ny,],2*self.nb_pw) )."""
        C = -1j*array(self.nb_pw*[dzup]+self.nb_pw*[dzdown])
        shp = self.Kz.shape[:-1] # (ns,nx[,ny])
        C = outer( np.ones( shp, dtype=np.complex128),C )
        E = outer(exp(self.Kz*C),np.ones(6))
        idx = list(range(E.ndim))
        idx[-1],idx[-2] = idx[-2],idx[-1]
        E = E.transpose(idx)
        return self.P*E

    def clearSK(self) :
        """Clears Kz and P; Useful to release memory."""
        self.__Kz = None
        self.__P = None

    @staticproperty
    def ListOfOrientedMaterials() : return USMat.__ListOfOrMat
    @staticmethod
    def sort_eig(M) :
        K,P = eig(M)
        shp = K.shape
        dm = shp[-1]
        ones = np.ones(dm,dtype=int)
        sort_keys = K.imag
        # renumbering of the Partial Waves (nos)
        Vmax = outer(abs(sort_keys).max(axis=-1),ones)
        sort_keys = (sort_keys>0)*sort_keys +\
                    (sort_keys<0)*(Vmax-sort_keys)
        nos = np.argsort(sort_keys)
        # renumbering tools
        stridx,Lidx = "",[]
        for i,n in enumerate(shp[:-1]) :
            stridx += "idx{},".format(i)
            Lidx.append(range(n))
        Lidx.append(range(dm))
        idxJ = stridx[:-1]
        idxK = idxJ + ",idx{}".format(i+1)
        idxP = idxK + ",idx{}".format(i+2)
        # Wave numbers
        exec("{} = meshgrid( *Lidx , indexing='ij')".format(idxK))
        K = eval("K[{},nos]".format(idxJ))
        # Polarizations        
        Lidx.append(range(dm))
        exec("{} = meshgrid( *Lidx , indexing='ij')".format(idxP))
        idx = list(range(len(shp)+1))
        idx[-2],idx[-1] = idx[-1],idx[-2]
        nos = outer(nos,ones).transpose(idx)
        P = eval("P[{},nos]".format(idxK))
        return K,P
    
    @staticmethod
    def _sigmaHor(C0,Ckx,Cky,U,Vkx,Vky=None) : # internal method
        """Returns the stress Sxx, Syy or Sxy, from the 6-dimensional
           field U. Needs the horizontal wavenumbers Vkx [, Vky] and the
           6-dimensional vectors C0,Ckx,Cky."""
        if Vky is None : # 2D
            onesX = np.ones_like(Vkx)
            CSxx  = np.einsum("i,j->ij",onesX,C0)
            CSxx += np.einsum("i,j->ij",Vkx,Ckx)
            return  np.einsum("jk,ijk->ij",CSxx,U)
        else : #3D
            onesX,onesY = np.ones_like(Vkx),np.ones_like(Vky)
            CSxx =  np.einsum("ij,k->ijk", \
                              np.einsum("i,j->ij",onesX,onesY),C0)
            CSxx += np.einsum("ij,k->ijk", \
                              np.einsum("i,j->ij",Vkx,onesY),Ckx)
            CSxx += np.einsum("ij,k->ijk", \
                              np.einsum("i,j->ij",onesX,Vky),Cky)
            return np.einsum("jkl,ijkl->ijk",CSxx,U)
            
#===========================================================================
def USMaterial(material, angles=None, USset=None) :
    """ Angles : (phi=0.0, theta=0.0, alpha=None) for anisotropy;
                 (theta=0.0, phi=0.0) for transversely isotropy
                        (if theta = 90°, the value of phi does not matter).
    """
    idset = id(USset)
    dict_LEOM = USMat.ListOfOrientedMaterials
    if idset in dict_LEOM.keys() :
        LEOM = dict_LEOM[idset]
    else :
        LEOM = []
    materials = [e.initial_material for e in LEOM]
    # isotropic cases
    if isinstance(material, Fluid) :
        if material in materials : # existing USFluid instance
             return LEOM[ materials.index(material)] 
        return USFluid(material, USset) # creating a new USFluid instance
    if isinstance(material, IsotropicElasticSolid) :
        if material in materials : # existing USTIE instance
             return LEOM[materials.index(material)]
        return USTIE(material, USset) # creating a new USTIE instance
    # other cases neading angles
    Langles = [e.angles for e in LEOM]
    if angles == None : angles = [0.0,0.0]
    else :
        da = 0.1
        angles = [ round(a/da)*da for a in angles ]
    if isinstance(material, TransverselyIsotropicElasticSolid) :
        # two angles only : theta and alpha (if theta != 0)
        if angles[0] == 0.0 : # Vertical Symmetry Axis
            for no,(m,a) in enumerate(zip(materials,Langles)) :
                if m == material : # same material
                    if a == None : 
                        return LEOM[no]
            # no similar material found; creating a new USMaterial instance
            return USTIE(material, USset)
        else : # Anisotropic case
            for no,(m,a) in enumerate( zip(materials, Langles) ) :
                if m == material : # same material
                    if a != None :
                        if a[0] == angles[0] and a[1] == angles[1] : 
                            return LEOM[no]
        return USAniE(material,angles,USset)
    if isinstance(material, AnisotropicElasticSolid) :
        # three angles : phi, theta and alpha (if theta != 0)
        phi,theta = angles[:2]
        for no,(m,a) in enumerate(zip(materials,Langles)) :
            if m == material : # same material
                if a[0] == phi and a[1] == theta : # same first two angles
                    if a[1] == 0.0 : # no alpha
                        return LEOM[no]
                    elif a[2] == angles[2] : # same alpha angle
                        return LEOM[no]
        # no similar material found; creating a new USMaterial instance
        return USAniE(material,angles,USset)

#===========================================================================
class USFluid(USMat) :
    """ US Fluid """
    def __init__(self, material, USset=None) :
        USMat.__init__(self, material, None, USset)
        self._set_nb_pw(1) # number of partial waves in each direction
    def update(self, S2, K, K2, Q=None) :
        """ S2, Kx, Ky and other arrays have same dimensions """
        # Initialisations
        m = self.mat
        shp = list(S2.shape)
        self._set_Kz(np.zeros( shp+[2], dtype = np.complex128 ))
        self._set_P(np.zeros( shp+[6,2], dtype = np.complex128 ))
        Kz = 1j*sqrt( K2 + S2/m.c**2 ) # Im(Kz)>0
        isurhoS2 = 1j*S2**(-1)/m.rho # Corrected in V1.6
        if Q is None : # 2D
            self._set_MMprod("ijkl,ijlm->ijkm")
            self._set_MVprod("ijkl,ijl->ijk")
            self.Kz[:,:,0] = Kz
            self.Kz[:,:,1] = -Kz
            self.P[:,:,0,0] = isurhoS2*K # Ux = 1j*kx/(rho*s**2)*p
            self.P[:,:,0,1] = self.P[:,:,0,0]
            self.P[:,:,1,:].fill(0.0) # Uy = 0.0
            self.P[:,:,2,0] = isurhoS2*Kz # Uz = 1j*kz/(rho*s**2)*p
            self.P[:,:,2,1] = -self.P[:,:,2,0]
            self.P[:,:,3:5,:].fill(0.0) # Sxz = Syz = 0
            self.P[:,:,5,:].fill(-1.0) # Szz = -1.0*p
        else : # 3D
            self._set_MMprod("ijklm,ijkmn->ijkln")
            self._set_MVprod("ijklm,ijkm->ijkl")
            Kx,Ky = K*cos(Q),K*sin(Q)
            self.Kz[:,:,:,0] = Kz
            self.Kz[:,:,:,1] = -Kz
            self.P[:,:,:,0,0] = isurhoS2*Kx # Ux = 1j*kx/(rho*s**2)*p
            self.P[:,:,:,0,1] = self.P[:,:,:,0,0]
            self.P[:,:,:,1,0] = isurhoS2*Ky # Uy = 1j*ky/(rho*s**2)*p
            self.P[:,:,:,1,1] = self.P[:,:,:,1,0]
            self.P[:,:,:,2,0] = isurhoS2*Kz # Uz = 1j*kz/(rho*s**2)*p
            self.P[:,:,:,2,1] = -self.P[:,:,:,2,0]
            self.P[:,:,:,3:5,:].fill(0.0) # Sxz = Syz = 0
            self.P[:,:,:,5,:].fill(-1.0) # Szz = -1.0*p
    def sigmaXX(self,U,Vkx=None,Vky=None) : # Vkx, Vky not used in a fluid
        """Returns the stress Sxx = Szz = -pressure,
           from the 6-dimensional field U."""
        return U[...,5]
    def sigmaYY(self,U,Vkx=None,Vky=None) : # Vkx, Vky not used in a fluid
        """Returns the stress Syy = Szz = -pressure,
           from the 6-dimensional field U."""
        return U[...,5]
    def sigmaXY(self,U,Vkx=None,Vky=None) : # Vkx, Vky not used in a fluid
        """Returns the stress Sxy = 0, from the 6-dimensional field U."""
        return np.zeros_like(U[...,5])
#===========================================================================
class USTIE(USMat) :
    """ US Vertical Transversely Isotropic Elastic Solid (with the symmetry
        axis #1) in the z-direction """
    def __init__(self, material, USset=None) :
        USMat.__init__(self, material, None, USset)
        self._set_nb_pw(3) # number of partial waves in each direction
        if isinstance(material, IsotropicElasticSolid) :
            self._set_IES_mat( material )
            self._set_mat( material.export() )
        m = self.mat
        cs = m.c13/m.c33
        cii = 1j*(cs*m.c13 - m.c11)
        cij = cii + 2j*m.c66
        # Coefficients for Sxx
        VZ = np.zeros( (6,), dtype=np.complex128)
        self.__CSxx_0 = VZ.copy()
        self.__CSxx_0[-1] = cs
        self.__CSxx_Kx = VZ.copy()
        self.__CSxx_Kx[0] = cii
        self.__CSxx_Ky = VZ.copy()
        self.__CSxx_Ky[1] = cij
        # Coefficients for Syy
        self.__CSyy_0 = self.__CSxx_0
        self.__CSyy_Kx = VZ.copy()
        self.__CSyy_Kx[0] = cij
        self.__CSyy_Ky = VZ.copy()
        self.__CSyy_Ky[1] = cii
        # Coefficients for Sxy
        self.__CSxy_0 = VZ.copy()
        self.__CSxy_Kx = VZ.copy()
        self.__CSxy_Kx[1] = -1j*m.c66
        self.__CSxy_Ky = VZ.copy()
        self.__CSxy_Ky[0] = -1j*m.c66
        
    def update(self, S2, K, K2, Q=None) :
        """ S2, K, K2 and other arrays have same dimensions """
        if Q is None : # 2D
            self.__Kx = K[0,:].copy()
            self.__Ky = None
            self._set_MMprod("ijkl,ijlm->ijkm")
            self._set_MVprod("ijkl,ijl->ijk")
        else : # 3D
            self._set_MMprod("ijklm,ijkmn->ijkln")
            self._set_MVprod("ijklm,ijkm->ijkl")
        # Initialisations
        m = self.mat
        shp = list(S2.shape)
        self._set_Kz(np.zeros( shp+[6], dtype = np.complex128 ))
        self._set_P(np.zeros( shp+[6,6], dtype = np.complex128 ))
        # SH Waves (out-of-plane)
        KSH = 1j*sqrt( (m.rho*S2+m.c66*K2)/m.c44 )
        # In-plane waves
        N4 = np.zeros( shp+[4,4], dtype = np.complex128 )
        if Q is None : # 2D
            self.Kz[:,:,2] = KSH
            self.P[:,:,1,2] = 1j/(m.c44*KSH)
            self.P[:,:,4,2].fill(1.0)
            self.Kz[:,:,5] = -KSH
            self.P[:,:,1,5] = -self.P[:,:,1,2]
            self.P[:,:,4,5].fill(1.0)
            N4[:,:,0,2].fill(1j/m.c44)
            N4[:,:,1,3].fill(1j/m.c33)
            N4[:,:,2,0] = 1j*m.rho*S2
            N4[:,:,3,1] = N4[:,:,2,0]
            N4[:,:,2,0] = 1j*m.rho*S2
            N4[:,:,0,1] = -K
            N4[:,:,1,0] = N4[:,:,0,1]* (m.c13/m.c33)
            N4[:,:,2,3] = N4[:,:,1,0]
            N4[:,:,3,2] = N4[:,:,0,1]
            N4[:,:,2,0] += 1j*(m.c11-m.c13*m.c13/m.c33)*K2
            K4,P4 = USMat.sort_eig(N4)
            self.Kz[:,:,:2] = K4[:,:,:2]
            self.Kz[:,:,3:5] = K4[:,:,2:]
            self.P[:,:,:3:2,:2] = P4[:,:,:2,:2]
            self.P[:,:,3:6:2,:2] = P4[:,:,2:,:2]
            self.P[:,:,:3:2,3:5] = P4[:,:,:2,2:]
            self.P[:,:,3:6:2,3:5] = P4[:,:,2:,2:]

        else : # 3D
            u0,u1,T = -sin(Q),cos(Q),1j/(m.c44*KSH)
            self.Kz[:,:,:,2] = KSH
            self.Kz[:,:,:,5] = -KSH
            self.P[:,:,:,0,2] = u0*T
            self.P[:,:,:,1,2] = u1*T
            self.P[:,:,:,3,2] = u0
            self.P[:,:,:,4,2] = u1
            self.P[:,:,:,0,5] = -self.P[:,:,:,0,2]
            self.P[:,:,:,1,5] = -self.P[:,:,:,1,2]
            self.P[:,:,:,3,5] = u0
            self.P[:,:,:,4,5] = u1
            N4[:,:,:,0,2].fill(1j/m.c44)
            N4[:,:,:,1,3].fill(1j/m.c33)
            N4[:,:,:,2,0] = 1j*m.rho*S2
            N4[:,:,:,3,1] = N4[:,:,:,2,0]
            N4[:,:,:,2,0] = 1j*m.rho*S2
            N4[:,:,:,0,1] = -K
            N4[:,:,:,1,0] = N4[:,:,:,0,1]* (m.c13/m.c33)
            N4[:,:,:,2,3] = N4[:,:,:,1,0]
            N4[:,:,:,3,2] = N4[:,:,:,0,1]
            N4[:,:,:,2,0] += 1j*(m.c11-m.c13*m.c13/m.c33)*K2
            K4,P4 = USMat.sort_eig(N4)
            co = [[1,1],[1,1]]
            U1,U0 = outer(u1,co),outer(-u0,co)
            self.Kz[:,:,:,:2] = K4[:,:,:,:2]
            self.Kz[:,:,:,3:5] = K4[:,:,:,2:]
            self.P[:,:,:,0::3,:2] += P4[:,:,:,0::2,:2]*U1
            self.P[:,:,:,1::3,:2] += P4[:,:,:,0::2,:2]*U0
            self.P[:,:,:,2::3,:2] = P4[:,:,:,1::2,:2]
            self.P[:,:,:,0::3,3:5] += P4[:,:,:,0::2,2:]*U1
            self.P[:,:,:,1::3,3:5] += P4[:,:,:,0::2,2:]*U0
            self.P[:,:,:,2::3,3:5] = P4[:,:,:,1::2,2:]
        return
        
    def sigmaXX(self, U, Vkx, Vky=None ) :
        """Returns the stress Sxx, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSxx_0, self.__CSxx_Kx, \
                               self.__CSxx_Ky, U, Vkx, Vky)
    
    def sigmaYY(self, U, Vkx, Vky=None ) :
        """Returns the stress Syy, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSyy_0, self.__CSyy_Kx, \
                               self.__CSyy_Ky, U, Vkx, Vky)
        
    def sigmaXY(self, U, Vkx, Vky=None ) :
        """Returns the stress Sxy, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSxy_0, self.__CSxy_Kx, \
                               self.__CSxy_Ky, U, Vkx, Vky)
    
#===========================================================================
class USAniE(USMat) :
    """ US Anisotropic Elastic Solid """
    def __init__(self, material, angles, USset=None ) :
        """ angles : (phi, beta, alpha) in degrees """
        USMat.__init__(self, material, angles, USset )
        self._set_nb_pw(3) # number of partial waves in each direction
        if isinstance(material, AnisotropicElasticSolid) :
            self._set_mat(material.rotate(*angles))
        else : # Transversely Isotropic Elastic Solid
            theta, alpha = angles
            self._set_mat(material.export().rotate(0,theta,alpha))
        LoL,LoM,LoN = self.mat.lol,self.mat.lom,self.mat.lon
        MoL,MoM,MoN = self.mat.mol,self.mat.mom,self.mat.mon
        NoL,NoM,NoN = self.mat.nol,self.mat.nom,self.mat.non
        NoN_inv = np.linalg.inv(NoN)
        Z = np.zeros( (6,6), dtype=np.complex128)
        self.__N0 = Z.copy()
        self.__N0[:3,3:] = 1j*NoN_inv
        self.__NS2 = Z.copy()
        self.__NS2[3:,:3] = 1j*self.mat.rho*np.eye(3)
        ML = -NoN_inv.dot(NoL)              #  -non^-1.nol
        self.__NKx = Z.copy()
        self.__NKx[:3,:3] = ML              #  -non^-1.nol
        self.__NKx[3:,3:] = ML.transpose()  #  -lon.non^-1
        MN = -NoN_inv.dot(NoM)              #  -non^-1.nom
        self.__NKy = Z.copy()
        self.__NKy[:3,:3] = MN              #  -non^-1.nom
        self.__NKy[3:,3:] = MN.transpose()  #  -mon.non^-1
        self.__NKx2 = Z.copy()
        self.__NKx2[3:,:3] = 1j*(LoL+LoN.dot(ML))
        self.__NKy2 = Z.copy()
        self.__NKy2[3:,:3] = 1j*(MoM+MoN.dot(MN))
        self.__NKxy = Z
        self.__NKxy[3:,:3] = 1j*(MoL+LoM+MoN.dot(ML)+LoN.dot(MN))
        # Coefficients for Sxx and Sxy (and Sxz too)
        CSxSz = LoN.dot(NoN_inv)
        CSxU_Kx = 1j*(CSxSz.dot(NoL)-LoL)
        CSxU_Ky = 1j*(CSxSz.dot(NoM)-LoM)
        VZ = np.zeros( (6,), dtype=np.complex128)
        self.__CSxx_0 = VZ.copy()
        self.__CSxx_0[-3:] = CSxSz[0,:]
        self.__CSxx_Kx = VZ.copy()
        self.__CSxx_Kx[:3] = CSxU_Kx[0,:]
        self.__CSxx_Ky = VZ.copy()
        self.__CSxx_Ky[:3] = CSxU_Ky[0,:]
        self.__CSxy_0 = VZ.copy()
        self.__CSxy_0[-3:] = CSxSz[1,:]
        self.__CSxy_Kx = VZ.copy()
        self.__CSxy_Kx[:3] = CSxU_Kx[1,:]
        self.__CSxy_Ky = VZ.copy()
        self.__CSxy_Ky[:3] = CSxU_Ky[1,:]
        # Coefficients for Sxy and Syy (and Syz too)
        CSySz = MoN.dot(NoN_inv)
        CSyU_Kx = 1j*(CSySz.dot(NoL)-MoL)
        CSyU_Ky = 1j*(CSySz.dot(NoM)-MoM)
        #print("Sxy_0:",CSxSz[1,:]==CSySz[0,:])
        #print("Sxy_Kx:",CSxU_Kx[1,:]==CSyU_Kx[0,:])
        #print("Sxy_Ky:",CSxU_Ky[1,:]==CSyU_Ky[0,:])
        self.__CSyy_0 = VZ.copy()
        self.__CSyy_0[-3:] = CSySz[1,:]
        self.__CSyy_Kx = VZ.copy()
        self.__CSyy_Kx[:3] = CSyU_Kx[1,:]
        self.__CSyy_Ky = VZ.copy()
        self.__CSyy_Ky[:3] = CSyU_Ky[1,:]

    def update(self, S2, Kx, Kx2, Ky=None, Ky2=None, Kxy=None) :
        """ S2, Kx, Ky and other arrays have same dimensions """
        if Ky is None : # 2D
            self._set_MMprod("ijkl,ijlm->ijkm")
            self._set_MVprod("ijkl,ijl->ijk")
        else : # 3D
            self._set_MMprod("ijklm,ijkmn->ijkln")
            self._set_MVprod("ijklm,ijkm->ijkl")

        N = outer(np.ones_like(S2),self.__N0) +  outer(S2,self.__NS2) + \
            outer(Kx,self.__NKx)  + outer(Kx2,self.__NKx2)
        if Ky is not None :
            N += outer(Ky,self.__NKy)  + outer(Ky2,self.__NKy2) + \
                 outer(Kxy,self.__NKxy)
        K,P = USMat.sort_eig(N)
        self._set_Kz(K)
        self._set_P(P)
        
    def sigmaXX(self, U, Vkx, Vky=None ) :
        """Returns the stress Sxx, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSxx_0, self.__CSxx_Kx, \
                               self.__CSxx_Ky, U, Vkx, Vky)
    
    def sigmaYY(self, U, Vkx, Vky=None ) :
        """Returns the stress Syy, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSyy_0, self.__CSyy_Kx, \
                               self.__CSyy_Ky, U, Vkx, Vky)
        
    def sigmaXY(self, U, Vkx, Vky=None ) :
        """Returns the stress Sxy, from the 6-dimensional field U.
           Needs the horizontal wavenumbers Vkx [, Vky]."""
        return USMat._sigmaHor(self.__CSxy_0, self.__CSxy_Kx, \
                               self.__CSxy_Ky, U, Vkx, Vky)
        
#=========== Tests =========================================================
if __name__ == "__main__" :
    # np.set_printoptions(precision=2)
    flu = Fluid({"rho":1000.0,"c":1500.0},"Water")
    USflu1 = USMaterial(flu)
    # print(USMat.ListOfOrientedMaterials)
    USflu2 = USMaterial(flu)
    test = "USflu1 is USflu2"
    print(test,"->",eval(test))
    alu = IsotropicElasticSolid({"rho":2700.0,"Young modulus":6.9e10,\
                                 "Poisson ratio":0.3},"Aluminum")
    USti1 = USMaterial(alu)
    #print(USMat.ListOfOrientedMaterials)
    USti1bis = USMaterial(alu)
    test = "USti1 is USti1bis"
    print(test,"->",eval(test))
    crbepx = TransverselyIsotropicElasticSolid({"rho":1560.0,\
                   "c11":1.4e10, "c12": 8.0e9, "c33": 8.7e10,\
                   "c13": 9.0e9, "c44": 4.7e9}, "Carbon-Epoxy")
    USti2 = USMaterial(crbepx)
    USti2bis = USMaterial(crbepx)
    test = "USti2 is USti2bis"
    print(test,"->",eval(test))
    USani1 = USMaterial(crbepx,(90,45))
    USani2 = USMaterial(crbepx,(90,45))
    test = "USani1 is USani2"
    print(test,"->",eval(test))
    # Anisotropic case for comparison of the two classes
    USani0 = USMaterial(crbepx.export(),(0,0))
    for ns, k in enumerate(USMat.ListOfOrientedMaterials.keys()) :
        for no, usm in enumerate(USMat.ListOfOrientedMaterials[k]) :
            print((ns,no),"->",(usm.mat.type,usm.mat.name,usm.angles))
    # US Calculation
    tmax = 100e-6 # Duration of the signal(s)
    gamma = -log(1e-5)/tmax
    print("Duration ~{:.2f} mus, gamma ~{:.2f} kHz".format(\
        1e6*tmax,1e-3*gamma))
    dw = 2*pi/tmax
    Nts2 = 2
    Nt = 2*Nts2
    Vs = array( [gamma+1j*dw*n for n in range(Nt)] )
    xmax = 0.01 # x in [-xmax+dx,xmax]
    dkx = pi/xmax
    Nxs2 = 3
    Vkx = array(\
         [n*dkx for n in list(range(Nxs2+1))+list(range(-Nxs2+1,0))] )
    # == 2D Case ==
    S2, Kx = meshgrid(Vs**2,Vkx,indexing='ij')
    Kx2 = Kx**2
    USti2.update(S2,Kx,Kx2)
    print("USti2.nMBytes : ~ {:.3f}".format(USti2.nMBytes))
    Kz2D = USti2.Kz.copy()
    P2D = USti2.P.copy()
    USani0.update(S2,Kx,Kx2)
    Kz2Dani = USani0.Kz.copy()
    P2Dani = USani0.P.copy()
    # == 3D Case == 
    ymax = 0.008 # x in [-xmax+dx,xmax]
    dky = pi/ymax
    Nys2 = 4
    Vky = array(\
           [n*dky for n in list(range(Nys2+1))+list(range(-Nys2+1,0))] )
    S2, Kx, Ky = meshgrid(Vs**2,Vkx,Vky,indexing='ij')
    Kx2, Ky2, Kxy = Kx**2,Ky**2,Kx*Ky
    # for fluids and T.I. media
    K2 = Kx2+Ky2
    K = sqrt(K2)
    Q = np.arctan2(Ky, Kx) # theta angles
    USti2.update(S2,K,K2,Q)
    print("USti2.nMBytes : ~ {:.3f}".format(USti2.nMBytes))
    Kz3D = USti2.Kz.copy()
    P3D = USti2.P.copy()
    # Checking for Ky = 0.0
    Kzmax = abs(Kz2D).max()
    print("Comparing 2D & 3D for Kz: error <= {:.2e}%".format(\
        100*abs(Kz2D-Kz3D[:,:,0]).max()/Kzmax))
    Pmax = abs(P2D).max()
    cp = abs(P3D[:,:,0]+P2D)
    cm = abs(P3D[:,:,0]-P2D)
    dst = (cp<cm)*cp+(cm<=cp)*cm
    print("Comparing 2D & 3D for P: error <= {:.2e}%".format(\
        100*abs(dst).max()/Pmax))
    USani0.update(S2,Kx,Kx2,Ky,Ky2,Kxy)
    Kz3Dani = USani0.Kz.copy()
    P3Dani = USani0.P.copy()
    # comparisons in Python shell...

