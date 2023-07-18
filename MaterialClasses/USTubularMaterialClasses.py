# Version 1.21 - 2023, July, 5
# Author : Eric Ducasse (coll. Aditya Krishna)
# License : CC-BY-NC
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
# ====== Initialization ====================================================
if __name__ == "__main__" : import TraFiC_init
# ====== Material Classes ==================================================
from MaterialClasses import  *
# ====== Numerical tools ===================================================
import numpy as np
from numpy import multiply
outer =  multiply.outer # (i1,...,in),(j1,...,jk) -> (i1,...,i2,j1,...jk)
from scipy.special import ive,kve
#===========================================================================
class staticproperty(property):
    """ Creating the  '@staticproperty' decorator"""
    def __get__(self, cls, owner):
        return staticmethod(self.fget).__get__(None, owner)()
#===========================================================================
class USTubularMat :
    """Virtual mother class of all classes of tubular US Materials."""
    __ListOfOrMat = dict()
    def __init__(self,material,angles,USset=None) :
        self.__initial_mat = material
        self.__mat = material # by default, for fluid and radial T.I.E.
        self.__IES = None # For "Isotropic Elastic Solid" case (saving
                          #   in file)
        self.__angles = angles
        self.__nbpw = None # Number of Partial Waves in each direction
        self.__Eta = None # i * Radial wave numbers of Partial Waves
        self.__params = dict() # dictionary of parameters for field comput.
        self.__MMprod = None # Matrix-Matrix product for np.einsum
        self.__MVprod = None # Matrix-Vector product for np.einsum
        self.__SMprod = None # (Scalar independent from n)-Matrix product
        self.__SVprod = None # (Scalar independent from n)-Vector product
        self.__SSprod = None # (Scalar independent from n)-Scalar product
        idset = id(USset)
        if idset in USTubularMat.__ListOfOrMat.keys() : # existing USset
            USTubularMat.__ListOfOrMat[idset].append(self)
        else : # new USset
            USTubularMat.__ListOfOrMat[idset] = [self]
    @property
    def initial_material(self) : return self.__initial_mat
    @property
    def mat(self) : return self.__mat
    @property
    def mat_diff_IES(self) :
        if self.__IES is None : return self.__mat
        else : return self.__IES
    def _set_mat(self,material) : self.__mat = material
    def _set_IES_mat(self, material) :
        self.__IES = material
    @property
    def angles(self) : return self.__angles
    @property
    def nb_pw(self) :
        """ Number of partial waves in each direction."""
        return self.__nbpw
    def _set_nb_pw(self,nb) : self.__nbpw = nb
    @property
    def Eta(self) : return self.__Eta
    def _set_Eta(self,Eta) : self.__Eta = Eta
    @property
    def parameters(self) : return self.__params # dictionary
    @property
    def MMprod(self) : return self.__MMprod
    def _set_MMprod(self,MMprod) : self.__MMprod = MMprod   
    @property
    def MVprod(self) : return self.__MVprod
    def _set_MVprod(self,MVprod) : self.__MVprod = MVprod 
    @property
    def SMprod(self) : return self.__SMprod 
    @property
    def SVprod(self) : return self.__SVprod 
    @property
    def SSprod(self) : return self.__SSprod 
    @property
    def nMBytes(self) :
        if self.Kz is None : # and self.P is None too
            return 0.0
        else :
            return 2**-20*(self.Kr.nbytes + sum([v.nbytes\
                                       for k,v in self.__params.items() ]))
    def update(self,S,K,N=None) :
        """ Should not be called """
        print("Error : USTubularMat::update called by a "+\
              "'{}' instance".format(type(self).__name__))
    def _update(self,axisym=False) :
        if axisym :
            self.__MMprod = "ijkl,ijlm->ijkm"
            self.__MVprod = "ijkl,ijl->ijk"
            self.__SMprod = "ij,ijkl->ijkl"
            self.__SVprod = "ij,ijk->ijk"
            self.__SSprod = "ij,ij->ij"
        else :
            self.__MMprod = "ijklm,ijkmn->ijkln"
            self.__MVprod = "ijklm,ijkm->ijkl"
            self.__SMprod = "ij,ijklm->ijklm"
            self.__SVprod = "ij,ijkl->ijkl"
            self.__SSprod = "ij,ijk->ijk"
    def field_components(self,r,r0,S2,K,N=None) :
        """ Should not be called """
        print("Error : USTubularMat::field_components called by a "+\
              "'{}' instance".format(type(self).__name__))

    def field(self,C,r,r0,S2,K,N=None) :
        """Field at the radial position r, with a normalization with
           respect to the radial position r0. C is the array of
           coefficients."""
        fc = self.field_components(r,r0,S2,K,N)
        return np.einsum(self.MMprod,fc,C)
    
    def clearEta(self) :
        """Clears Eta and other parameters; useful to release memory."""
        self.__Eta = None
        self.__params = dict() # Empty dictionary

    @staticproperty
    def ListOfOrientedMaterials() : return USTubularMat.__ListOfOrMat            
#===========================================================================
def USTubularMaterial(material,angles=None,USset=None) :
    """ Angles : (phi=0.0, theta=0.0, alpha=None) for anisotropy;
                 (theta=0.0, phi=0.0) for transversely isotropy
                        (if theta = 90°, the value of phi does not matter).
    """
    idset = id(USset)
    dict_LEOM = USTubularMat.ListOfOrientedMaterials
    if idset in dict_LEOM.keys() :
        LEOM = dict_LEOM[idset]
    else :
        LEOM = []
    materials = [e.initial_material for e in LEOM]
    # isotropic cases
    if isinstance(material,Fluid) :
        if material in materials : # existing USFluid instance
             return LEOM[materials.index(material)]
        return USTubularFluid(material,USset) # creating a new
                                              # USTubularFluid instance
    if isinstance(material,IsotropicElasticSolid) :
        if material in materials : # existing USTIE instance
             return LEOM[materials.index(material)]
        return USTubTIE(material,USset) # creating a new USTTubIE instance
    # other cases neading angles
    Langles = [e.angles for e in LEOM]
    if angles == None : angles = [0.0,0.0]
    else :
        da = 0.1
        angles = [ round(a/da)*da for a in angles ]
    if isinstance(material,TransverselyIsotropicElasticSolid) :
        # two angles only : theta and alpha (if theta != 0)
        if angles[0] == 0.0 : # Axial Symmetry Axis
            for no,(m,a) in enumerate(zip(materials,Langles)) :
                if m == material : # same material
                    if a == None : 
                        return LEOM[no]
            # no similar material found; creating a new USTTubIE instance
            return USTubTIE(material,USset)
        else : # Anisotropic case
            print("Not yet available because solution has not been found "+\
                  "analytically.")
            return None
    if isinstance(material,AnisotropicElasticSolid) :
        print("Not yet available because solution has not been found "+\
                  "analytically.")
        return None
#===========================================================================
#   Ingoing waves: ive(v, z)
# Exponentially scaled modified Bessel function of the first kind
# Defined as:
#    ive(v, z) = iv(v, z) * exp(-abs(z.real))
#----------------------------------------------------------------
#   Outgoing waves: kve(v, z)
# Exponentially scaled modified Bessel function of the second kind.
# Defined as, for real order `v` at complex `z`, as:
#    kve(v, z) = kv(v, z) * exp(z)
#===========================================================================
class USTubularFluid(USTubularMat) :
    """ US Tubular Fluid """
    def __init__(self,material,USset=None) :
        USTubularMat.__init__(self,material,None,USset)
        self._set_nb_pw(1) # number of partial waves in each direction
    def update(self,S2,K,K2,N=None) :
        """ S2, K, K2, N and other arrays have same dimensions.
            S2 is the array of the square of the Laplace variable;
            K contains the axial wavenumbers, K2 their squares,
            N contains the azimuthal wavenumbers (integers)."""
        # Initializations
        self._update(axisym=(N is None) ) 
        if N is not None : # Eta and coefficients independent from n
            S2,K,K2 = S2[:,:,0],K[:,:,0],K2[:,:,0] 
        m = self.mat
        self._set_Eta(np.sqrt( K2 + S2/m.c**2 ))
    def field_components(self,r,r0,S2,K,N=None) :
        shp = S2.shape
        FC = np.zeros( shp+(4,2), dtype = np.complex128)
        # (Ur,Ua,Uz,Srr=-P),(Outgoing,Ingoing)
        Eta_r,dr,iK = self.Eta*r, r-r0, 1.0j*K
        # Outgoing waves:
        #    K(n,eta*r)*exp(eta*r0) = kve(n,eta*r)*exp(-eta*dr)
        # Ingoing waves:
        #    I(n,eta*r)*exp(-eta.real*r0) = ive(n,eta*r)*exp(eta.real*dr)
        Eplus,Emoins = np.exp(-self.Eta*dr),np.exp(self.Eta.real*dr)
        cSigma = -self.mat.rho*S2
        if N is None : # Axisymmetrical            
            K0,K1 = Eplus*kve(0,Eta_r),Eplus*kve(1,Eta_r)
            I0,I1 = Emoins*ive(0,Eta_r),Emoins*ive(1,Eta_r)
            FC[:,:,0,0],FC[:,:,0,1] = self.Eta*K1,-self.Eta*I1
            #FC[:,:,1,0],FC[:,:,1,1] = 0,0
            FC[:,:,2,0],FC[:,:,2,1] = iK*K0,iK*I0
            FC[:,:,3,0],FC[:,:,3,1] = cSigma*K0,cSigma*I0
        else : # 3D
            ones_N = np.ones( (shp[-1],), dtype = np.complex128)
            Eta_r = np.einsum("ij,k->ijk",Eta_r,ones_N)
            Km1,K0,Kp1 = np.einsum(self.SSprod,Eplus,kve(N-1,Eta_r)),\
                         np.einsum(self.SSprod,Eplus,kve(N,Eta_r)),\
                         np.einsum(self.SSprod,Eplus,kve(N+1,Eta_r))
            Im1,I0,Ip1 = np.einsum(self.SSprod,Emoins,ive(N-1,Eta_r)),\
                         np.einsum(self.SSprod,Emoins,ive(N,Eta_r)),\
                         np.einsum(self.SSprod,Emoins,ive(N+1,Eta_r))
            FC[:,:,:,0,0] = np.einsum(self.SSprod,0.5*self.Eta,Km1+Kp1)
            FC[:,:,:,0,1] = np.einsum(self.SSprod,-0.5*self.Eta,Im1+Ip1)
            FC[:,:,:,1,0] = np.einsum(self.SSprod,0.5j*self.Eta,Kp1-Km1)
            FC[:,:,:,1,1] = np.einsum(self.SSprod,-0.5j*self.Eta,Ip1-Im1)
            FC[:,:,:,2,0],FC[:,:,:,2,1] = iK*K0,iK*I0
            FC[:,:,:,3,0],FC[:,:,:,3,1] = cSigma*K0,cSigma*I0
        return FC
        
    def sigmaThetaTheta(self,U,r=None,K=None,N=None) : # r,K,N not used in
                                                       # a fluid
        """Returns the stress Saa = Srr = Szz = -pressure,
           from the 4-dimensional field U."""
        return U[...,3]
    
    def sigmaZZ(self,U,r=None,K=None,N=None) : # r,K,N not used in
                                               # a fluid
        """Returns the stress Szz = Srr = Saa = -pressure,
           from the 4-dimensional field U."""
        return U[...,3]
        
    def sigmaThetaZ(self,U,r=None,K=None,N=None) :
        """Returns the stress Saz = 0,
           from the 4-dimensional field U."""
        return np.zeros_like(U[...,3])
    
#===========================================================================
class USTubularTransverselyIsotropicElastic( USTubularMat ) :
    """ US Vertical Transversely Isotropic Elastic Solid (with the symmetry
        axis #1) in the radial direction """
    def __init__(self, material, USset=None ) :
        USTubularMat.__init__(self, material, None, USset )
        self._set_nb_pw(3) # number of partial waves in each direction
        if isinstance(material, IsotropicElasticSolid ) :
            self._set_IES_mat( material )
            self._set_mat( material.export() )
        self.__idx_k_eq_0 = None # indexes for which k == 0
        self.__idx_k_ne_0 = None # indexes for which k != 0
        # Coefficients for Saa,Szz,Saz
        ani = self.mat.export()
        T = np.array( [ [0,-1,0], [1,0,0], [0,0,0] ], dtype=np.float64)
        invL = np.linalg.inv(ani.lol)
        ML,NL = ani.mol@invL,ani.nol@invL
        MLM,MLN,NLM,NLN = ML@ani.lom,ML@ani.lon,NL@ani.lom,NL@ani.lon
        MQ_nsr = 1j*(MLM-ani.mom)
        MQ_k = 1j*(MLN-ani.mon)
        MQ_1sr = (ani.mom-MLM)@T
        MQ_S = ML
        MZ_nsr = 1j*(NLM-ani.nom)
        MZ_k = 1j*(NLN-ani.non)
        MZ_1sr = (ani.nom-NLM)@T
        MZ_S = NL
        self.__CSaa_nsr = MQ_nsr[1,:]
        self.__CSaz_nsr = MQ_nsr[2,:]
        self.__CSzz_nsr = MZ_nsr[2,:]
        self.__CSaa_k = MQ_k[1,:]
        self.__CSaz_k = MQ_k[2,:]
        self.__CSzz_k = MZ_k[2,:]
        self.__CSaa_1sr = MQ_1sr[1,:]
        self.__CSaz_1sr = MQ_1sr[2,:]
        self.__CSzz_1sr = MZ_1sr[2,:]
        self.__CSaa_Sr = MQ_S[1,:]
        self.__CSaz_Sr = MQ_S[2,:]
        self.__CSzz_Sr = MZ_S[2,:]
        if False :
            print("Verifications :")
            print("\tMQ_nsr[0,:]:",MQ_nsr[0,:])
            print("\tMZ_nsr[0,:]:",MZ_nsr[0,:])
            print("\tMQ_nsr[2,:]:",MQ_nsr[2,:])
            print("\tMZ_nsr[1,:]:",MZ_nsr[1,:])
            print("\tMQ_k[0,:]:",MQ_k[0,:])
            print("\tMZ_k[0,:]:",MZ_k[0,:])
            print("\tMQ_k[2,:]:",MQ_k[2,:])
            print("\tMZ_k[1,:]:",MZ_k[1,:])
            print("\tMQ_1sr[0,:]:",MQ_1sr[0,:])
            print("\tMZ_1sr[0,:]:",MZ_1sr[0,:])
            print("\tMQ_1sr[2,:]:",MQ_1sr[2,:])
            print("\tMZ_1sr[1,:]:",MZ_1sr[1,:])
            print("\tMQ_S[0,:]:",MQ_S[0,:])
            print("\tMZ_S[0,:]:",MZ_S[0,:])
            print("\tMQ_S[2,:]:",MQ_S[2,:])
            print("\tMZ_S[1,:]:",MZ_S[1,:])
    #-------------------------------------
    def update(self,S2,K,K2,N=None) :
        """ S2, K, K2 and other arrays have same dimensions """
        # Initializations
        self._update(axisym=(N is None) )
        if N is not None : # Eta and coefficients independent from n
            S2,K,K2 = S2[:,:,0],K[:,:,0],K2[:,:,0]
        Vk = K[0,:]
        K_is_zero = (Vk == 0)
        self.__idx_k_eq_0 = np.where(K_is_zero)[0]
        self.__idx_k_ne_0 = np.where(~K_is_zero)[0]
        m = self.mat
        rhoS2 = m.rho*S2
        # Wave number (eta) calculations
        a44 = rhoS2 + m.c44*K2
        a33 = rhoS2 + m.c33*K2
        a13 = (m.c13+m.c44)*K
        A = m.c11*m.c44
        mBs2 = 0.5*(m.c11*a33 + m.c44*a44 - a13**2)
        C = a33*a44
        D = np.sqrt(mBs2**2-A*C)
        Eta1_square = (mBs2 - D)/A
        Eta1_square[:,self.__idx_k_eq_0] = rhoS2[:,self.__idx_k_eq_0]/m.c11
        Eta1 = np.sqrt(Eta1_square)
        Eta2_square = (mBs2 + D)/A
        Eta2_square[:,self.__idx_k_eq_0] = rhoS2[:,self.__idx_k_eq_0]/m.c44
        Eta2 = np.sqrt(Eta2_square)
        Eta3 = np.sqrt((rhoS2 + m.c44*K2)/m.c66)
        shp = rhoS2.shape
        M_Eta = np.empty( shp +(3,), dtype = np.complex128)
        M_Eta[:,:,0] = Eta1
        M_Eta[:,:,1] = Eta2
        M_Eta[:,:,2] = Eta3
        self._set_Eta(M_Eta)
        # b1 coefficient :
        denom1,denom2 =  -a13*Eta1,m.c44*Eta1_square - a33
        test = abs(denom1) > abs(denom2)
        b1 = np.empty_like(S2)
        idx = np.where(test)
        b1[idx] = (m.c11*Eta1_square[idx]-a44[idx])/denom1[idx]
        test0 = (~test)&(denom2==0)
        idx = np.where( (~test)&(~test0) )
        b1[idx] = a13[idx]*Eta1[idx]/denom2[idx]
        idx = np.where(test0) # Should be empty
        nb0 = len(idx[0])
        if nb0 > 0 :
            print("Case 0 for eta1:",nb0,"/",S2.size)
            print("(S,K) :")
            print(np.array([sqrt(S2[idx]),K[idx]]).T)
        b1[idx] = np.ones_like(a13[idx])
        self.parameters["B1"] = b1
        # b2 coefficient :
        denom1,denom2 =  -a13*Eta2,m.c44*Eta2_square - a33
        test = abs(denom1) > abs(denom2)
        test[:,self.__idx_k_eq_0] = False # for k == 0
        b2 = np.empty_like(S2)
        idx = np.where(test)
        b2[idx] = (m.c11*Eta2_square[idx]-a44[idx])/denom1[idx]
        test0 = (~test)&(denom2==0)
        test0[:,self.__idx_k_eq_0] = True # for k == 0
        self.test0 = test0
        idx = np.where( (~test)&(~test0) )
        b2[idx] = a13[idx]*Eta2[idx]/denom2[idx]
        idx = np.where(test0)
        nb0 = len(idx[0])
        if nb0 != len(self.__idx_k_eq_0)*shp[0] : # Should be for k=0 only
            print("Case 0 for eta2:",len(idx[0]),"/",S2.size,\
                  "!= {}x{}".format(len(self.__idx_k_eq_0),shp[0]))
            print("(S,K) :")
            print(np.array([sqrt(S2[idx]),K[idx]]).T)
        b2[idx] = np.ones_like(a13[idx])
        self.parameters["B2"] = b2 
    #-------------------------------------
    def field_components(self,r,r0,S2,K,N=None) :
        shp = S2.shape
        FC = np.zeros( shp+(6,6), dtype = np.complex128)
        # (Ur,Ua,Uz,Srr,Sra,Srz),(Outgoing,Ingoing)
        dr = r - r0
        if N is None : # Axisymmetrical
            Eta_r = self.Eta*r
        else : # 3D
            ones_N = np.ones( (shp[-1],), dtype = np.complex128)
            Eta_r = np.einsum("ijl,k->ijkl",self.Eta*r,ones_N)
        Eta_dr = self.Eta*dr          
        # Outgoing waves:
        #    K(n,eta*r)*exp(eta*r0) = kve(n,eta*r)*exp(-eta*dr)
        # Ingoing waves:
        #    I(n,eta*r)*exp(-eta.real*r0) = ive(n,eta*r)*exp(eta.real*dr)
        Eplus,Emoins = np.exp(-Eta_dr),np.exp(Eta_dr.real)
        B1,B2 = self.parameters["B1"],self.parameters["B2"]
        if N is None : # Axisymmetrical            
            K0,K1,K2 = Eplus*kve(0,Eta_r),Eplus*kve(1,Eta_r),\
                       Eplus*kve(2,Eta_r)
            I0,I1,I2 = Emoins*ive(0,Eta_r),Emoins*ive(1,Eta_r),\
                       Emoins*ive(2,Eta_r)
            # Displacements
            M = np.array([[1,1,0],[0,0,1],[0,0,0]],dtype=np.complex128)
            FC[...,:3,:3] = np.einsum("ijl,kl->ijkl",K1,-M)
            FC[...,2,0] = 1j*B1*K0[...,0]
            FC[...,2,1] = 1j*B2*K0[...,1]
            FC[...,:3,3:] = np.einsum("ijl,kl->ijkl",I1,M)
            FC[...,2,3] = 1j*B1*I0[...,0]
            FC[...,2,4] = 1j*B2*I0[...,1]
            # Radial Stresses
            FC[...,3:,:3] = self.mat.c66*\
                            np.einsum("ijl,kl->ijkl",self.Eta*K2,M)
            C0 = (self.mat.c11-self.mat.c66)*self.Eta[...,0] +\
                 self.mat.c13*B1*K
            C1 = (self.mat.c11-self.mat.c66)*self.Eta[...,1] +\
                 self.mat.c13*B2*K
            FC[...,3,0] += C0*K0[...,0]
            FC[...,3,1] += C1*K0[...,1]
            D0 = 1j*self.mat.c44*(B1*self.Eta[...,0]-K)
            D1 = 1j*self.mat.c44*(B2*self.Eta[...,1]-K)
            FC[...,5,0] = -D0*K1[...,0]
            FC[...,5,1] = -D1*K1[...,1]
            FC[...,3:,-3:] = self.mat.c66*\
                            np.einsum("ijl,kl->ijkl",self.Eta*I2,M)
            FC[...,3,3] += C0*I0[...,0]
            FC[...,3,4] += C1*I0[...,1]
            FC[...,5,3] = D0*I1[...,0]
            FC[...,5,4] = D1*I1[...,1]
            # k = 0
            for no in self.__idx_k_eq_0 : # 0 or 1 value
                FC[:,no,::3,1::3] = 0
                FC[:,no,1::3,1::3] = 0
                FC[:,no,2,1] = K0[:,no,1] 
                FC[:,no,2,4] = I0[:,no,1] 
                FC[:,no,5,1] = -self.mat.c44*self.Eta[:,no,1]*K1[:,no,1] 
                FC[:,no,5,4] = self.mat.c44*self.Eta[:,no,1]*I1[:,no,1]
        else : # 3D
            N3 = np.einsum("ijk,l->ijkl",N,[1,1,1])
            idx = 'ijl,ijkl->ijkl'
            Km1,K0,Kp1 = np.einsum(idx,Eplus,kve(N3-1,Eta_r)),\
                         np.einsum(idx,Eplus,kve(N3,Eta_r)),\
                         np.einsum(idx,Eplus,kve(N3+1,Eta_r))
            Km2,Kp2 = np.einsum(idx,Eplus,kve(N3-2,Eta_r)),\
                      np.einsum(idx,Eplus,kve(N3+2,Eta_r))
            Im1,I0,Ip1 = np.einsum(idx,Emoins,ive(N3-1,Eta_r)),\
                         np.einsum(idx,Emoins,ive(N3,Eta_r)),\
                         np.einsum(idx,Emoins,ive(N3+1,Eta_r))
            Im2,Ip2 = np.einsum(idx,Emoins,ive(N3-2,Eta_r)),\
                      np.einsum(idx,Emoins,ive(N3+2,Eta_r))
            # Displacements
            Mm = 0.5*np.array([[1,1,1j],[-1j,-1j,1],[0,0,0]])
            Mp = 0.5*np.array([[1,1,-1j],[1j,1j,1],[0,0,0]])
            FC[...,:3,:3] = np.einsum("ijkm,lm->ijklm",Km1,-Mm) + \
                            np.einsum("ijkm,lm->ijklm",Kp1,-Mp)
            FC[...,2,0] = 1j*np.einsum("ij,ijk->ijk",B1,K0[...,0])
            FC[...,2,1] = 1j*np.einsum("ij,ijk->ijk",B2,K0[...,1])
            FC[...,:3,3:] = np.einsum("ijkm,lm->ijklm",Im1,Mm) + \
                            np.einsum("ijkm,lm->ijklm",Ip1,Mp)
            FC[...,2,3] = 1j*np.einsum("ij,ijk->ijk",B1,I0[...,0])
            FC[...,2,4] = 1j*np.einsum("ij,ijk->ijk",B2,I0[...,1])
            # Radial Stresses
            Eta1 = outer(self.Eta[...,0],ones_N)
            Eta2 = outer(self.Eta[...,1],ones_N)
            FC[...,3:,:3] = self.mat.c66*( \
                              np.einsum("ijkm,lm->ijklm",\
                               np.einsum(idx,self.Eta,Km2),Mm)+\
                              np.einsum("ijkm,lm->ijklm",\
                               np.einsum(idx,self.Eta,Kp2),Mp) )
            C0 = (self.mat.c11-self.mat.c66)*Eta1 +\
                 self.mat.c13*np.einsum("ij,ijk->ijk",B1,K)
            C1 = (self.mat.c11-self.mat.c66)*Eta2 +\
                 self.mat.c13*np.einsum("ij,ijk->ijk",B2,K)
            FC[...,3,0] += C0*K0[...,0]
            FC[...,3,1] += C1*K0[...,1]
            D0 = 0.5j*self.mat.c44*(np.einsum("ij,ijk->ijk",B1,Eta1)-K)
            D1 = 0.5j*self.mat.c44*(np.einsum("ij,ijk->ijk",B2,Eta2)-K)
            FC[...,5,0] = -D0*(Km1[...,0]+Kp1[...,0])
            FC[...,5,1] = -D1*(Km1[...,1]+Kp1[...,1])
            FC[...,5,2] = 0.5*self.mat.c44*K*(Kp1[...,2]-Km1[...,2])
            FC[...,3:,-3:] = self.mat.c66*( \
                              np.einsum("ijkm,lm->ijklm",\
                               np.einsum(idx,self.Eta,Im2),Mm)+\
                              np.einsum("ijkm,lm->ijklm",\
                               np.einsum(idx,self.Eta,Ip2),Mp) )
            FC[...,3,3] += C0*I0[...,0]
            FC[...,3,4] += C1*I0[...,1]
            FC[...,5,3] = D0*(Im1[...,0]+Ip1[...,0])
            FC[...,5,4] = D1*(Im1[...,1]+Ip1[...,1])
            FC[...,5,5] = 0.5*self.mat.c44*K*(Im1[...,2]-Ip1[...,2])
            # k = 0
            for no in self.__idx_k_eq_0 : # 0 or 1 value
                FC[:,no,:,::3,1::3] = 0
                FC[:,no,:,1::3,1::3] = 0
                FC[:,no,:,2,1] = K0[:,no,:,1] 
                FC[:,no,:,2,4] = I0[:,no,:,1] 
                FC[:,no,:,5,1] = -0.5*self.mat.c44*np.einsum("i,ij->ij", \
                                    self.Eta[:,no,1], \
                                    (Km1[:,no,:,1]+Kp1[:,no,:,1])) 
                FC[:,no,:,5,4] = 0.5*self.mat.c44*np.einsum("i,ij->ij", \
                                    self.Eta[:,no,1], \
                                    (Im1[:,no,:,1]+Ip1[:,no,:,1])) 
        return FC
        
    def sigmaThetaTheta(self,U,r,K,N=None) :
        """Returns the stress Saa, from the 6-dimensional field U,
           at r>0."""
        if N is None : # axisymmetric case 2D
            Saa = np.einsum("ijk,k->ij",U[...,:3],self.__CSaa_1sr)
            Saa /= r
            M = np.einsum("i,j->ij",K,self.__CSaa_k)
            Saa += np.einsum("ijk,jk->ij",U[...,:3],M)
            Saa += np.einsum("ijk,k->ij",U[...,-3:],self.__CSaa_Sr)
        else : # 3D
            Saa = np.einsum("ijkm,m->ijk",U[...,:3],self.__CSaa_1sr)
            M = np.einsum("i,j->ij",N,self.__CSaa_nsr)
            Saa += np.einsum("ijkm,km->ijk",U[...,:3],M)
            Saa /= r
            M = np.einsum("i,j->ij",K,self.__CSaa_k)
            Saa += np.einsum("ijkm,jm->ijk",U[...,:3],M)
            Saa += np.einsum("ijkm,m->ijk",U[...,-3:],self.__CSaa_Sr)        
        return Saa
    
    def sigmaZZ(self,U,r,K,N=None) :
        """Returns the stress Szz, from the 6-dimensional field U,
           at r>0."""
        if N is None : # axisymmetric case 2D
            Szz = np.einsum("ijk,k->ij",U[...,:3],self.__CSzz_1sr)
            Szz /= r
            M = np.einsum("i,j->ij",K,self.__CSzz_k)
            Szz += np.einsum("ijk,jk->ij",U[...,:3],M)
            Szz += np.einsum("ijk,k->ij",U[...,-3:],self.__CSzz_Sr)
        else : # 3D
            Szz = np.einsum("ijkm,m->ijk",U[...,:3],self.__CSzz_1sr)
            M = np.einsum("i,j->ij",N,self.__CSzz_nsr)
            Szz += np.einsum("ijkm,km->ijk",U[...,:3],M)
            Szz /= r
            M = np.einsum("i,j->ij",K,self.__CSzz_k)
            Szz += np.einsum("ijkm,jm->ijk",U[...,:3],M)
            Szz += np.einsum("ijkm,m->ijk",U[...,-3:],self.__CSzz_Sr)        
        return Szz
        
    def sigmaThetaZ(self,U,r,K,N=None) :
        """Returns the stress Saz, from the 6-dimensional field U,
           at r>0."""
        if N is None : # axisymmetric case 2D
            Saz = np.einsum("ijk,k->ij",U[...,:3],self.__CSaz_1sr)
            Saz /= r
            M = np.einsum("i,j->ij",K,self.__CSaz_k)
            Saz += np.einsum("ijk,jk->ij",U[...,:3],M)
            Saz += np.einsum("ijk,k->ij",U[...,-3:],self.__CSaz_Sr)
        else : # 3D
            Saz = np.einsum("ijkm,m->ijk",U[...,:3],self.__CSaz_1sr)
            M = np.einsum("i,j->ij",N,self.__CSaz_nsr)
            Saz += np.einsum("ijkm,km->ijk",U[...,:3],M)
            Saz /= r
            M = np.einsum("i,j->ij",K,self.__CSaz_k)
            Saz += np.einsum("ijkm,jm->ijk",U[...,:3],M)
            Saz += np.einsum("ijkm,m->ijk",U[...,-3:],self.__CSaz_Sr)        
        return Saz
# Abbreviation:
USTubTIE = USTubularTransverselyIsotropicElastic
#=========== Tests =========================================================
if __name__ == "__main__" :
    from SpaceGridClasses import Space2DGrid
    from TimeGridClass import TimeGrid
    # np.set_printoptions(precision=2)
    flu = Fluid({"rho":1000.0,"c":1500.0},"Water")
    print(flu)
    USflu = USTubularMaterial(flu)
    alu = IsotropicElasticSolid({"rho":2700.0,"Young modulus":6.9e10,\
                                 "Poisson ratio":0.3},"Aluminum")
    print(alu)
    USalu = USTubularMaterial(alu)
    crbepx = TransverselyIsotropicElasticSolid({"rho":1560.0,\
                   "c11":1.4e10, "c12": 8.0e9, "c33": 8.7e10,\
                   "c13": 9.0e9, "c44": 4.7e9}, "Carbon-Epoxy")
    print(crbepx)
    UScrbepx = USTubularMaterial(crbepx)
    print(USTubularMat.ListOfOrientedMaterials)
    # Time grid
    tg = TimeGrid(1.2e-5,1e-6)
    Vs = tg.S
    # Space grid
    gd = Space2DGrid(12,6,5e-3,np.pi/3)
    Vk,Vn = gd.Kx,gd.Ky
    # 3D or axisymmetrical
    choix = 0
    if choix == 0 : # 3D
        S2,K,N = np.meshgrid(Vs**2,Vk,Vn,indexing="ij")
    elif choix == 1 : # Axisymmetrical
        N = None
        Vn = None
        S2,K = np.meshgrid(Vs**2,Vk,indexing="ij")
    K_square = K**2
    # Updates and field computation
    USflu.update(S2,K,K_square,N)
    FC1 = USflu.field_components(10.5e-3,10.0e-3,S2,K,N)
    USalu.update(S2,K,K_square,N)
    FC2 = USalu.field_components(10.5e-3,11.0e-3,S2,K,N)
    Ualu = FC2[...,0] - 1j*FC2[...,1] +3*FC2[...,2]
    Saa = USalu.sigmaThetaTheta(Ualu,5e-3,Vk,Vn)
    Szz = USalu.sigmaZZ(Ualu,5e-3,Vk,Vn)
    Saz = USalu.sigmaThetaZ(Ualu,5e-3,Vk,Vn)
