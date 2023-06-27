# Version 0.97 - 2022, September, 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
outer = np.multiply.outer
import scipy.sparse as sprs
from scipy.linalg import block_diag
import sys, os
if __name__ == "__main__" :
    cwd = os.getcwd()
    while "TraFiC" not in os.path.basename(cwd) :
        cwd = os.path.dirname(cwd)
    sys.path.append(cwd)
    import TraFiC_init
from Plane_Guided_Mode import Plane_Guided_Mode
from MaterialClasses import *
from USMaterialClasses import USMaterial # For postprocessing
#=========================================================================
#==   FDScheme                                                          ==
#=========================================================================
class FDScheme:
    """Finite difference scheme of even order n."""
    def __init__(self,order) :
        if order%2 == 1 :
            order += 1
            msg = "[FDScheme::FDScheme] Warning: odd order -> even order"
            print(msg,order)
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
            D1.append(np.linalg.solve(M,B).round(14))
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
            
            D2.append(np.linalg.solve(M,B).round(14))
        self.__D2 = np.array(D2)
        # Derivatives of order greater than 2 are not taken into account
        # here
    @property
    def order(self) :
        """Order of the Finite Difference Scheme."""
        return self.__order
    @property
    def m(self) :
        """Half-order of the Finite Difference Scheme."""
        return self.__order//2
    @property
    def D1(self) :
        """(order+1)-by-(order+1) matrix of the coefficients of
           the Finite Difference Scheme for the first derivative.
           Each row #i gives the coefficients with respect to
           the value indexes : (n-i,...n+order-i). """
        return self.__D1
    @property
    def d100(self) :
        """Coefficient d_00^[1]. """
        return self.__D1[0,0]
    def D1up(self,d=1) :
        M = np.einsum("ij,kl->ikjl",self.__D1[:self.m+1,:],np.eye(d))
        M.shape = ( d*(self.m+1), d*(self.order+1) )
        return M
    def D1down(self,d=1) :
        M = np.einsum("ij,kl->ikjl",self.__D1[-self.m-1:,:],np.eye(d))
        M.shape = ( d*(self.m+1), d*(self.order+1) )
        return M
    def D1center(self,d=1) :
        M = np.einsum("j,kl->kjl",self.__D1[self.m,:],np.eye(d))
        M.shape = ( d, d*(self.order+1) )
        return M    
    def L1up(self,d=1) :
        M = np.einsum("j,kl->kjl",self.__D1[0,1:],np.eye(d))
        M.shape = ( d, d*self.order )
        return M    
    def L1down(self,d=1) :
        M = np.einsum("j,kl->kjl",self.__D1[-1,:-1],np.eye(d))
        M.shape = ( d, d*self.order )
        return M    
    @property
    def D2(self) : 
        """(order+2)-by-(order+2) matrix of the coefficients of
           the Finite Difference Scheme for the second derivative.
           Each row #i gives the coefficients with respect to
           the value indexes : (n-i,...n+order+1-i). """
        return self.__D2
    def D2up(self,d=1) :
        M = np.einsum("ij,kl->ikjl",self.__D2[:self.m+1,:],np.eye(d))
        M.shape = ( d*(self.m+1), d*(self.order+2) )
        return M
    def D2down(self,d=1) :
        M = np.einsum("ij,kl->ikjl",self.__D2[-self.m-1:,:],np.eye(d))
        M.shape = ( d*(self.m+1), d*(self.order+2) )
        return M
    def D2center(self,d=1) :
        M = np.einsum("j,kl->kjl",self.__D2[self.m,:-1],np.eye(d))
        M.shape = ( d, d*(self.order+1) )
        return M    
#=========================================================================
#==   DiscretizedLayer (function)                                       ==
#=========================================================================
def DiscretizedLayer(material, e, nb, fds) :
    """ Discretized layer of medium material and thickness e,
        discretized into nb sub-intervals of thickness e/nb.
        'fds' is the Finite Difference Scheme."""
    # The number of subintervals has to be greater than fds.order + 2 
    if nb < fds.order + 3 : nb = fds.order + 3
    # Fluid layer
    if isinstance(material,Fluid) :
        mat = material
        class_name = FluidDiscretizedLayer
        # Elastic Solid (Stiffnesses can be complexes)
    elif isinstance(material,IsotropicElasticSolid) :
        mat = material.export().export()
        class_name = SolidDiscretizedLayer
    elif isinstance(material,TransverselyIsotropicElasticSolid) :
        mat = material.export()
        class_name = SolidDiscretizedLayer
    elif isinstance(material,AnisotropicElasticSolid) :
        mat = material
        class_name = SolidDiscretizedLayer
    else :
        print("[DiscretizedLayer : Material of type " + \
              f"'{type(material).__name__}'\n\t" + \
              "cannot be taken into account.")
        return None
    return class_name(mat, e, nb, fds)
def print_matrix_structure(sparse_matrix) :
    T = np.empty( sparse_matrix.shape, dtype = str )
    T[:,:] = "-"
    T[sparse_matrix.nonzero()] = "x"
    print("\n".join([" ".join(L) for L in T.tolist() ]))
#=========================================================================
#==   AbstractDiscretizedLayer                                          ==
#=========================================================================
class GeneralDiscretizedLayer :
    # Parameter to allows sparse matrices of not 
    SPARSE = False
    #--------------------------------------------------
    def __init__(self, material, e, nb, fds) :
        self.__fds = fds
        self.__n = nb 
        self.__Z = np.linspace(0,e,nb+1)
        self.__Z.setflags(write=False) # read-only vector
        self.__h = e/nb
        self.__mat = material
        self.__dim = None
        self.__N = None
    #--------------------------------------------------
    @property
    def material(self) : return self.__mat
    @property
    def n(self) :
        """Number of sub-intervals."""
        return self.__n
    @property
    def thickness(self) :
        """Thickness of the layer."""
        return self.Z[-1]
    @property
    def e(self) :
        """Thickness of the layer."""
        return self.Z[-1]
    @property
    def fds(self) :
        """Finite difference scheme."""
        return self.__fds
    @property
    def Z(self) :
        """Discretized z-positions."""
        return self.__Z
    @property
    def h(self) :
        """Length of each sub-interval."""
        return self.__h
    @property
    def state_dim(self) :
        """Dimension of the local state vector."""
        return self.__dim
    @property
    def d(self) :
        """Dimension of the local state vector."""
        return self.__dim
    def _set_state_dim(self, new_dim) :
        """Set the dimension of the local state vector
           (called by the derived classes)."""
        self.__dim = new_dim
        self.__N = self.__dim * (self.__n - 1)
    @property
    def global_state_dim(self) :
        """Dimension of the global state vector."""
        return self.__N
    @property
    def N(self) :
        """Dimension of the global state vector."""
        return self.__N
    #--------------------------------------------------
    def __str__(self) :
        txt = f"Discretized layer of thickness {1e3*self.e:.3f} mm"
        txt += f"\n and material '{self.material.name}', with "
        txt += f"{self.n} sub-intervals"
        return txt
    #--------------------------------------------------
    def matrices_fixed_wavenumber(self, K, Nu=0.0 ) :
        """Returns matrices M,C0,Cn such that the eigenvalue
           problem to solve in the layer is :
                          i*w*W = M@W + C0@w0 + Cn@wn.
        """
        d,fds,N = self.d,self.__fds,self.N
        M0 = self.M0(K,Nu)
        M1 = self.M1(K,Nu) / self.h
        M2 = self.M2(K,Nu) / self.h**2
        return self.__global_matrices(fds,d,N,M0,M1,M2)
    #--------------------------------------------------
    def matrices_fixed_frequency(self, F, Nu=0.0 ) :
        """Returns matrices M,C0,Cn such that the eigenvalue
           problem to solve in the layer is :
                          -i*k*W = M@W + C0@w0 + Cn@wn.
        """
        d,fds,N = self.d,self.__fds,self.N
        W = 2*np.pi*F
        M0 = self.N0(W,Nu)
        M1 = self.N1(W,Nu) / self.h
        M2 = self.N2(W,Nu) / self.h**2 
        return self.__global_matrices(fds,d,N,M0,M1,M2)
    #--------------------------------------------------
    def matrices_fixed_slowness(self, S ) :
        """Returns matrices M,C0,Cn such that the eigenvalue
           problem to solve in the layer is :
                          i*w*W = M@W + C0@w0 + Cn@wn.
        """
        d,fds,N = self.d,self.__fds,self.N
        M0,M1,M2 = self.P0(S),self.P1(S),self.P2(S)
        for e in [M0,M1,M2] :
            if e is None : # Degenerate case
                return None
        M1 /= self.h
        M2 /= self.h**2 
        return self.__global_matrices(fds,d,N,M0,M1,M2)
    #--------------------------------------------------
    @staticmethod
    def ein_subs(d) :
        """Returns subscripts s for numpy.einsum(s,u,v) or
           numpy.einsum(s,M,u) :
                 u.v, M u, M^T u, outer(u,v) and diag(u)*M.
        """
        p = ""
        for no in range(105,105+d) :
            p += chr(no)
        i,j = chr(no+1),chr(no+2)
        pi,pj = p+i,p+j
        ps = pi+","+pi+"->"+p
        pij,pji = pi+j,pj+i
        pm = pij+","+pj+"->"+pi
        pt = pji+","+pj+"->"+pi
        po = pi+","+pj+"->"+pij
        pd = pij+","+pi+"->"+pij
        return ps, pm, pt, po, pd    
    #--------------------------------------------------
    @staticmethod
    def __global_matrices(fds,d,N,M0,M1,M2):
        D1u,D1d,D1c = fds.D1up(d),fds.D1down(d),fds.D1center(d)
        D2u,D2d,D2c = fds.D2up(d),fds.D2down(d),fds.D2center(d)
        r,l = D1u[d:,d:].shape
        _,j = D1c.shape 
        M = M0.copy()
        # First derivatives
        M[:r,:l] += M1[:r,:r]@D1u[d:,d:]
        for ir,ic in zip(range(r,N-r,d),range(0,N-2*r,d)) :
            M[ir:ir+d,ic:ic+j] += M1[ir:ir+d,ir:ir+d]@D1c
        M[-r:,-l:] += M1[-r:,-r:]@D1d[:-d,:-d]
        C0,Cn = M1[:r,:r]@D1u[d:,:d],M1[-r:,-r:]@D1d[:-d,-d:]
        # Second derivatives
        l += d
        M[:r,:l] += M2[:r,:r]@D2u[d:,d:]
        for ir,ic in zip(range(r,N-r,d),range(0,N-2*r,d)) :
            M[ir:ir+d,ic:ic+j] += M2[ir:ir+d,ir:ir+d]@D2c
        M[-r:,-l:] += M2[-r:,-r:]@D2d[:-d,:-d]
        C0 += M2[:r,:r]@D2u[d:,:d]
        Cn += M2[-r:,-r:]@D2d[:-d,-d:]
        return M,C0,Cn
    #--------------------------------------------------
    @staticmethod
    def homogenized_parameters(K,W,Nu) :
        homogenized_array = GeneralDiscretizedLayer.homogenized_array
        msg = "GeneralDiscretizedLayer.homogenized_parameters:"
        msg += "\n\tError:\n\t\t"
        k_dim,w_dim,nu_dim = [ np.ndim(X) for X in (K,W,Nu) ]
        k_shp,w_shp,nu_shp = [ np.shape(X) for X in (K,W,Nu) ]
        if k_dim > w_dim :
            if k_shp[k_dim-w_dim:] != w_shp :
                msg += ("K {} and W {} have incompatibles shapes." +
                        "").format(k_shp,w_shp)
                raise ValueError(msg)
            W = homogenized_array(k_shp[:k_dim-w_dim],W)
            w_dim,w_shp = k_dim,k_shp
            K = np.array(K)
        elif k_dim < w_dim :
            if w_shp[w_dim-k_dim:] != k_shp :
                msg += ("K {} and W {} have incompatibles shapes." +
                        "").format(k_shp,w_shp)
                raise ValueError(msg)
            K = homogenized_array(w_shp[:w_dim-k_dim],K)
            k_dim,k_shp = w_dim,w_shp
            W = np.array(W)
        else : # k_dim = w_dim
            if w_shp != k_shp :
                msg += ("K {} and W {} have incompatibles shapes." +
                        "").format(k_shp,w_shp)
                raise ValueError(msg)
            if k_dim > 0 :
                K,W = np.array(K),np.array(W)
        # K and W have now the same shape
        if nu_dim > k_dim :
            msg += ("K {} and Nu {} have incompatibles shapes." +
                    "").format(k_shp,nu_shp)
            raise ValueError(msg)
        elif nu_dim < k_dim :  
            if k_shp[k_dim-nu_dim:] != nu_shp :
                msg += ("K {} and Nu {} have incompatibles shapes." +
                        "").format(k_shp,nu_shp)
                raise ValueError(msg)
            Nu = homogenized_array(k_shp[:k_dim-nu_dim],Nu)
        else : # nu_dim = k_dim
            if k_shp != nu_shp :
                msg += ("K {} and Nu {} have incompatibles shapes." +
                        "").format(k_shp,nu_shp)
                raise ValueError(msg)
            if nu_dim > 0 :
                Nu = np.array(Nu)
        return K,W,Nu 
    #--------------------------------------------------
    @staticmethod
    def homogenized_array(shp,V) :
        V_shp = np.shape(V)
        V = np.array(V).flatten()
        ones = np.ones( shp, dtype=V.dtype )
        ones = ones.flatten()
        V = np.einsum("i,j->ij" ,ones, V)
        V.shape = shp + V_shp
        return V 
#=========================================================================
#==   FluidDiscretizedLayer                                             ==
#=========================================================================
class FluidDiscretizedLayer(GeneralDiscretizedLayer) :

    def __init__(self, material, e, nb, fds) :
        GeneralDiscretizedLayer.__init__(self, material, e, nb, fds)
        self._set_state_dim(2)
    #--------------------------------------------------
    # Fixed wavenumber
    #--------------------------------------------------
    def M0(self, K, Nu=0.0) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[I,Ip1] = -1.0/self.material.rho
        M[Ip1,I] = self.material.rho*self.material.c**2*(K**2+Nu**2)
        return M 
    #--------------------------------------------------
    def M1(self, K, Nu=0.0) :
        """Matrix of coefficients of w'(z) (zero for a fluid)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        return M
    #--------------------------------------------------
    def M2(self, K, Nu=0.0) :
        """Matrix of coefficients of w''(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[Ip1,I] = -self.material.rho*self.material.c**2
        return M   
    #-------------------------------------------------- 
    # Fixed frequency 
    #--------------------------------------------------
    def N0(self, W, Nu=0.0) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[I,Ip1] = 1.0
        M[Ip1,I] = Nu**2 - W**2/self.material.c**2
        return M
    #--------------------------------------------------
    def N1(self, W, Nu=0.0) :
        """Matrix of coefficients of w'(z) (zero for a fluid)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        return M
    #--------------------------------------------------
    def N2(self, W, Nu=0.0) :
        """Matrix of coefficients of w''(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[Ip1,I] = -1.0
        return M
    #-------------------------------------------------- 
    # Fixed slowness
    #--------------------------------------------------
    def P0(self, S) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        rho = self.material.rho
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[I,Ip1] = -1.0/rho
        return M
    #--------------------------------------------------
    def P1(self, S) :
        """Matrix of coefficients of w'(z) (zero for a fluid)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        return M
    #--------------------------------------------------
    def P2(self, S) :
        """Matrix of coefficients of w''(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        rho,c = self.material.rho,self.material.c
        sc = S*c
        if abs(sc-1) < 1e-8 : # Degenerate case
            return None
        I,Ip1 = np.arange(0,self.N,2),np.arange(1,self.N,2)
        M[Ip1,I] = rho*c**2/(sc**2-1)
        return M  
    #--------------------------------------------------
    def partial_waves(self, K, W, Nu=0.0) :
        """Returns the vertical wavenumbers ±sqrt(W²/c²-K²-Nu²)."""
        K,W,Nu = self.homogenized_parameters(K,W,Nu)
        Kz = -1.0j * np.sqrt( (1.0+0.0j) * \
                              (K**2+Nu**2-(W/self.material.c)**2) )
        Kappa = np.empty( np.shape(K)+(2,), dtype = np.complex128)
        Kappa[...,0],Kappa[...,1] = Kz,-Kz
        return Kappa
    #--------------------------------------------------
    def mode_shape(self, disc_shape, K, W, Nu=0.0, left=False, \
                   right=False) :
        """Returns the vertical wavenumbers 'Kz' and the amplitudes
           'A' of the partial waves contributing to the mode of
           discretized shape 'disc_shape'.
           For fluids, any scalar field f at position z is:
               f(z) = A[0]*exp(-i*Kz[0]*z) + A[1]*exp(i*Kz[1]*(e-z))
           For solids, the displacement field u at position z is:
               u(z) = A[:,:3]@exp(-i*Kz[:3]*z) +
                      A[:,-3:]@exp(i*Kz[-3:]*(e-z))."""
        msg = "FluidDiscretizedLayer.mode_shape:\n\tError:"
        K,W,Nu = self.homogenized_parameters(K,W,Nu)
        ps,pm,_,_,_ = self.ein_subs( np.ndim(K) )
        e = self.e # thickness
        # positions of inner nodes, which can include the bounds
        if left :
            if right :
                Z = self.Z[:]
                nb_nodes = self.n + 1
            else :
                Z = self.Z[:-1]
                nb_nodes = self.n
        else :
            if right :
                Z = self.Z[1:]
                nb_nodes = self.n
            else :
                Z = self.Z[1:-1]
                nb_nodes = self.n - 1
        Kz = self.partial_waves(K, W, Nu)
        K_shp = np.shape(K)
        shp = K_shp + (nb_nodes,)
        if np.shape(disc_shape) != shp :
            msg += " incompatible shape between\n\t\t" + \
                   "W, K, Nu {} -> {} and\n\t\t".format(K_shp,shp) +\
                   "discretized mode shape {}".format( \
                       disc_shape.shape)
            raise ValueError(msg)
        A = np.empty( shp+(2,), dtype=np.complex128 )
        A[...,0] = np.exp(-1.0j*np.multiply.outer(Kz[...,0],Z))
        A[...,1] = np.exp(1.0j*np.multiply.outer(Kz[...,1],(e-Z)))
        A_star = np.conjugate(np.swapaxes(A,-1,-2))
        M = A_star@A
        B = np.einsum(pm, A_star, disc_shape)        
        nrm = np.sqrt( np.einsum(ps, np.conjugate(disc_shape), \
                                     disc_shape).real )
        X = np.linalg.solve(M,B)
        err = np.einsum(pm, A, X) - disc_shape
        err = np.sqrt( np.einsum(ps, np.conjugate(err), err).real )
        return Kz, X, err/nrm
    #--------------------------------------------------
    def shape_function(self, field, k, f, Kz, C, nu=0, zL=0.0) :
        """k is the wavenumber in the x-direction, f the nonzero
           frequency, Kz a (2,) vector, C a (2,) vector of coefficients,
           nu the wavenumber in the y-direction, zL the z-origin.
           Returns a ufunc which gives the shape of the field
           at any vertical position.
        """
        if field not in ("Ux", "Uy", "Uz", "Sxx", "Sxy", "Sxz", \
                         "Syy", "Syz", "Szz", "Phi", "Psi", "P", \
                         "dUx/dz", "dUy/dz", "dUz/dz", \
                         "dSxz/dz", "dSyz/dz", "dSzz/dz") :
            print(f"Unknown field '{field}'")
            return np.zeros_like
        if field in ("Sxy","Sxz","Syz","dSxz/dz","dSyz/dz") :
            return np.zeros_like
        def fonc_phi(Vz, kz=Kz, c=C, z0=zL, z1=zL+self.e):
            Z = np.clip(Vz, z0, z1)
            Vphi = c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                   c[1]*np.exp(-1.0j*kz[1]*(Z-z1))
            return ((Vz>z0)&(Vz<=z1))*Vphi
        if field == "Phi" :
            return fonc_phi
        w = 2.0*np.pi*f # nonzero frequency
        iw = 1j*w
        if field == "Psi" :
            def fonc_psi(Vz, iw=iw, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return Vphi/iw
            return fonc_psi
        if field == "Ux" :
            def fonc_Ux(Vz, ksw=k/w, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -ksw*Vphi
            return fonc_Ux
        if field == "Uy" :
            def fonc_Uy(Vz, nsw=nu/w, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -nsw*Vphi
            return fonc_Uy
        if field == "Uz" :
            def fonc_Uz(Vz, w=w, kz=Kz, c=C, z0=zL, z1=zL+self.e):
                Z = np.clip(Vz, z0, z1)
                Uz = -( kz[0]*c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                        kz[1]*c[1]*np.exp(-1.0j*kz[1]*(Z-z1)) ) / w
                return ((Vz>z0)&(Vz<=z1))*Uz
            return fonc_Uz
        if field == "P" :
            rho = self.material.rho
            def fonc_P(Vz, iwrho=iw*rho, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -iwrho*Vphi
            return fonc_P
        if field in ("Sxx","Syy","Szz") :
            rho = self.material.rho
            def fonc_SL(Vz, iwrho=iw*rho, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return iwrho*Vphi
            return fonc_SL
        if field == "dUx/dz" :
            def fonc_dUx(Vz, ksw=k/w, kz=Kz, c=C, z0=zL, z1=zL+self.e):
                Z = np.clip(Vz, z0, z1)
                mwUz =  kz[0]*c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                        kz[1]*c[1]*np.exp(-1.0j*kz[1]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*1j*ksw*mwUz 
            return fonc_dUx
        if field == "dUy/dz" :
            def fonc_dUy(Vz, nsw=nu/w, kz=Kz, c=C, z0=zL, z1=zL+self.e):
                Z = np.clip(Vz, z0, z1)
                mwUz =  kz[0]*c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                        kz[1]*c[1]*np.exp(-1.0j*kz[1]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*1j*nsw*mwUz
            return fonc_dUy
        if field == "dUz/dz" :
            def fonc_dUz(Vz, w=w, kz=Kz, c=C, z0=zL, z1=zL+self.e):
                Z = np.clip(Vz, z0, z1)
                miwdUz = kz[0]**2*c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                         kz[1]**2*c[1]*np.exp(-1.0j*kz[1]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*1j*miwdUz/w
            return fonc_dUz
        if field == "dSzz/dz" :
            rho = self.material.rho
            def fonc_dSzz(Vz, wrho=w*rho, kz=Kz, c=C, z0=zL, \
                          z1=zL+self.e):
                Z = np.clip(Vz, z0, z1)
                idVphi = c[0]*kz[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                         c[1]*kz[1]*np.exp(-1.0j*kz[1]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*idVphi*wrho
            return fonc_dSzz
        print(f"Program error: unexpected field '{field}'")
        return np.zeros_like               
#=========================================================================
#==   SolidDiscretizedLayer                                             ==
#=========================================================================
class SolidDiscretizedLayer(GeneralDiscretizedLayer) :

    def __init__(self, material, e, nb, fds) :
        GeneralDiscretizedLayer.__init__(self, material, e, nb, fds)
        self._set_state_dim(6)
        if isinstance(material, AnisotropicElasticSolid) :
            if GeneralDiscretizedLayer.SPARSE :
                zero_matrix = sprs.lil_matrix
                matrix = sprs.lil_matrix
            else :
                zero_matrix = np.zeros
                matrix = np.array
            self.__O3 = lambda z,ZM=zero_matrix : \
                        ZM( (3,3), dtype=np.complex)
            self.__rho = lambda z : self.material.rho
            # Stresses coefficients in the x-direction
            self.__Cxx = lambda z,M=matrix : M(self.material.lol)
            self.__Hxx = self.__O3
            self.__Cxy = lambda z,M=matrix : M(self.material.lom)
            self.__Hxy = self.__O3
            self.__Cxz = lambda z,M=matrix : M(self.material.lon)
            self.__Hxz = self.__O3
            # Stresses coefficients in the y-direction
            self.__Cyx = lambda z,M=matrix : M(self.material.mol)
            self.__Hyx = self.__O3
            self.__Cyy = lambda z,M=matrix : M(self.material.mom)
            self.__Hyy = self.__O3
            self.__Cyz = lambda z,M=matrix : M(self.material.mon)
            self.__Hyz = self.__O3
            # Stresses coefficients in the z-direction
            self.__Czx = lambda z,M=matrix : M(self.material.nol)
            self.__Hzx = self.__O3
            self.__Czy = lambda z,M=matrix : M(self.material.nom)
            self.__Hzy = self.__O3
            self.__Czz = lambda z,M=matrix : M(self.material.non)
            self.__Hzz = self.__O3
            # First derivatives of stresses coefficients
            self.__Czx_p = self.__O3
            self.__Hzx_p = self.__O3
            self.__Czy_p = self.__O3
            self.__Hzy_p = self.__O3
            self.__Czz_p = self.__O3
            self.__Hzz_p = self.__O3 
            # US Speeds in the x-direction  (fixed-slowness problem)
            VCX = np.sqrt(np.linalg.eigvals(self.material.lol) / \
                          self.material.rho)
            self.__x_speeds = VCX
        else : # To be completed
            pass
        self.__Kxx = lambda z,w : self.__Cxx(z) + 1.0j*w*self.__Hxx(z)
        self.__Kxy = lambda z,w : self.__Cxy(z) + 1.0j*w*self.__Hxy(z)
        self.__Kxz = lambda z,w : self.__Cxz(z) + 1.0j*w*self.__Hxz(z)
        self.__Kyx = lambda z,w : self.__Cyx(z) + 1.0j*w*self.__Hyx(z)
        self.__Kyy = lambda z,w : self.__Cyy(z) + 1.0j*w*self.__Hyy(z)
        self.__Kyz = lambda z,w : self.__Cyz(z) + 1.0j*w*self.__Hyz(z)
        self.__Kzx = lambda z,w : self.__Czx(z) + 1.0j*w*self.__Hzx(z)
        self.__Kzy = lambda z,w : self.__Czy(z) + 1.0j*w*self.__Hzy(z)
        self.__Kzz = lambda z,w : self.__Czz(z) + 1.0j*w*self.__Hzz(z)
        self.__Kzx_p = \
                lambda z,w : self.__Czx_p(z) + 1.0j*w*self.__Hzx_p(z)
        self.__Kzy_p = \
                lambda z,w : self.__Czy_p(z) + 1.0j*w*self.__Hzy_p(z)
        self.__Kzz_p = \
                lambda z,w : self.__Czz_p(z) + 1.0j*w*self.__Hzz_p(z)
    #-------------------------------------------------- 
    # Fixed wavenumber
    #--------------------------------------------------
    def M0(self, K, Nu=0.0) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.block_diag( (self.n-1)* \
                                 [sprs.eye(6,k=3,dtype=np.complex)], \
                                 "lil")
        else :
            M = block_diag( *((self.n-1)* \
                                 [np.eye(6,k=3,dtype=np.complex)]))
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            rho = self.__rho(z)
            Sxx,Sxy,Syx = self.__Cxx(z),self.__Cxy(z),self.__Cyx(z)
            Syy,Tzx,Tzy = self.__Cyy(z),self.__Czx_p(z),self.__Czy(z)
            M[m:r,l:m] = -( K*(K*Sxx+Nu*(Sxy+Syx)+1.0j*Tzx) + \
                            Nu*(Nu*Syy+1.0j*Tzy) )/rho
            Sxx,Sxy,Syx = self.__Hxx(z),self.__Hxy(z),self.__Hyx(z)
            Syy,Tzx,Tzy = self.__Hyy(z),self.__Hzx_p(z),self.__Hzy(z)
            M[m:r,m:r] = -( K*(K*Sxx+Nu*(Sxy+Syx)+1.0j*Tzx) + \
                            Nu*(Nu*Syy+1.0j*Tzy) )/rho
        return M
    #--------------------------------------------------
    def M1(self, K, Nu=0.0) :
        """Matrix of coefficients of w'(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            rho = self.__rho(z)
            Sxz,Szx,Syz = self.__Cxz(z),self.__Czx(z),self.__Cyz(z)
            Szy,Tzz     = self.__Czy(z),self.__Czz_p(z)
            M[m:r,l:m] = (-1.0j*( K*(Sxz+Szx) + Nu*(Syz+Szy) ) + \
                          Tzz)/rho
            Sxz,Szx,Syz = self.__Hxz(z),self.__Hzx(z),self.__Hyz(z)
            Szy,Tzz     = self.__Hzy(z),self.__Hzz_p(z)
            M[m:r,m:r] = (-1.0j*( K*(Sxz+Szx) + Nu*(Syz+Szy) ) + \
                          Tzz)/rho
        return M
    #--------------------------------------------------
    def M2(self, K, Nu=0.0) :
        """Matrix of coefficients of w''(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            rho = self.__rho(z)
            M[m:r,l:m] = self.__Czz(z)/rho
            M[m:r,m:r] = self.__Hzz(z)/rho
        return M   
    #--------------------------------------------------
    def A0_B0_W(self, K, Nu=0.0) :
        """Matrices A0 and B0 for the left bound."""
        d00,ush = self.fds.d100,1.0/self.h
        Czx,Czy,Czz = self.__Czx(0.0),self.__Czy(0.0),self.__Czz(0.0)
        M_to_inv = -ush*d00*Czz + 1.0j*(K*Czx+Nu*Czy)
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        M = np.linalg.inv(M_to_inv)
        return ush*M@Czz, M   
    #--------------------------------------------------
    def An_Bn_W(self, K, Nu=0.0) :
        """Matrices An and Bn for the right bound."""
        d00,e,ush = self.fds.d100,self.thickness,1.0/self.h
        Czx,Czy,Czz = self.__Czx(e),self.__Czy(e),self.__Czz(e)
        M_to_inv = ush*d00*Czz + 1.0j*(K*Czx+Nu*Czy)
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        M = np.linalg.inv(M_to_inv)
        return ush*M@Czz, M   
    #--------------------------------------------------
    def IC0_IM0_W(self, K, Nu=0.0) :
        """Matrices IC0 and IM0 for the left bound."""
        d00,ush = self.fds.d100,1.0/self.h
        Czx,Czy,Czz = self.__Czx(0.0),self.__Czy(0.0),self.__Czz(0.0)
        M = -ush*d00*Czz + 1.0j*(K*Czx+Nu*Czy)
        return ush*Czz, M   
    #--------------------------------------------------
    def ICn_IMn_W(self, K, Nu=0.0) :
        """Matrices ICn and IMn for the right bound."""
        d00,e,ush = self.fds.d100,self.thickness,1.0/self.h
        Czx,Czy,Czz = self.__Czx(e),self.__Czy(e),self.__Czz(e)
        M = ush*d00*Czz + 1.0j*(K*Czx+Nu*Czy)
        return ush*Czz, M 
    #-------------------------------------------------- 
    # Fixed frequency  
    #--------------------------------------------------
    def N0(self, W, Nu=0.0) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.block_diag( (self.n-1)* \
                                 [sprs.eye(6,k=3,dtype=np.complex)], \
                                 "lil")
        else :
            M = block_diag( *((self.n-1)* \
                                 [np.eye(6,k=3,dtype=np.complex)]) )
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            rho = self.__rho(z)
            Cxx,Cxy = self.__Kxx(z,W),self.__Kxy(z,W)
            Cyx,Cyy = self.__Kyx(z,W),self.__Kyy(z,W)
            Czx_p,Czy_p = self.__Kzx_p(z,W),self.__Kzy_p(z,W)
            M_to_inv = Cxx
            if GeneralDiscretizedLayer.SPARSE :
                M_to_inv = M_to_inv.toarray()
            Cxx_inv = np.linalg.inv(M_to_inv)
            M[m:r,l:m] = Nu*Cxx_inv@(Nu*Cyy+1.0j*Czy_p) \
                         - rho*W**2*Cxx_inv
            M[m:r,m:r] = Cxx_inv@(1.0j*Nu*(Cxy+Cyx)-Czx_p)
        return M
    #--------------------------------------------------
    def N1(self, W, Nu=0.0) :
        """Matrix of coefficients of w'(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)      
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            Cxx,Cxz = self.__Kxx(z,W),self.__Kxz(z,W)
            Cyz,Czy = self.__Kyz(z,W),self.__Kzy(z,W)
            Czx,Czz_p = self.__Kzx(z,W),self.__Kzz_p(z,W)
            M_to_inv = Cxx
            if GeneralDiscretizedLayer.SPARSE :
                M_to_inv = M_to_inv.toarray()
            Cxx_inv = np.linalg.inv(M_to_inv)
            M[m:r,l:m] = Cxx_inv@(1.0j*Nu*(Cyz+Czy)-Czz_p)
            M[m:r,m:r] = -Cxx_inv@(Cxz+Czx)
        return M 
    #--------------------------------------------------
    def N2(self, W, Nu=0.0) :
        """Matrix of coefficients of w''(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)        
        for i,z in enumerate(self.Z[1:-1]) :
            l = 6*i
            m,r = l+3,l+6
            Cxx,Czz = self.__Kxx(z,W),self.__Kzz(z,W)
            M_to_inv = Cxx
            if GeneralDiscretizedLayer.SPARSE :
                M_to_inv = M_to_inv.toarray()
            Cxx_inv = np.linalg.inv(M_to_inv)
            M[m:r,l:m] = -Cxx_inv@Czz
        return M 
    #--------------------------------------------------
    def Matrices0(self, W, Nu=0.0) :
        """Matrices A0, B0, Czx, X0, Y0=X0m1 and diagonal of the 
           inverses of the nonzero eigenvalues for the left bound."""
        d00,ush = self.fds.d100,1.0/self.h
        Czx,Czy = self.__Kzx(0.0,W),self.__Kzy(0.0,W)
        Czz = self.__Kzz(0.0,W)
        M_to_inv = -ush*d00*Czz + 1.0j*Nu*Czy
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bo = np.linalg.inv(M_to_inv)
        M = Bo@Czx
        vp,Xo = np.linalg.eig(M)
        abs_vp = np.abs(vp)
        max_vp = abs_vp.max()
        nonzero_vp = (abs_vp >= 1e-10*max_vp)*1
        rank = nonzero_vp.sum()
        vp *= nonzero_vp
        if nonzero_vp[rank:].sum() > 0 : # unsorted eigenvalues
            Lmi = [ (m,i) for i,m in enumerate(abs_vp) ]
            Lmi.sort(reverse=True)
            indexes = np.array([ c[1] for c in Lmi ])
            vp = vp[indexes]
            Xo = Xo[:,indexes]
        Do,Xom1 = np.diag(vp),np.linalg.inv(Xo)
        if not np.allclose(M,Xo@Do@Xom1) :
            print("Warning : non-diagonalizable matrix \n{M}")            
        return ush*Bo@Czz, Bo, Czx, Xo, Xom1, np.diag(1.0/vp[:rank])
    #--------------------------------------------------
    def Matricesn(self, W, Nu=0.0) :
        """Matrices An, Bn, Czx, Xn, Yn=Xnm1 and diagonal of the
           inverses of the nonzero eigenvalues for the right bound."""
        d00,e,ush = self.fds.d100,self.thickness,1.0/self.h
        Czx,Czy = self.__Kzx(e,W),self.__Kzy(e,W)
        Czz = self.__Kzz(e,W)
        M_to_inv = ush*d00*Czz + 1.0j*Nu*Czy
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bn = np.linalg.inv(M_to_inv)
        M = Bn@Czx
        vp,Xn = np.linalg.eig(M)
        abs_vp = np.abs(vp)
        max_vp = abs_vp.max()
        nonzero_vp = (abs_vp >= 1e-10*max_vp)*1
        rank = nonzero_vp.sum()
        vp *= nonzero_vp
        if nonzero_vp[rank:].sum() > 0 : # unsorted eigenvalues
            Lmi = [ (m,i) for i,m in enumerate(abs_vp) ]
            Lmi.sort(reverse=True)
            indexes = np.array([ c[1] for c in Lmi ])
            vp = vp[indexes]
            Xn = Xn[:,indexes]
        Dn,Xnm1 = np.diag(vp),np.linalg.inv(Xn)
        if not np.allclose(M,Xn@Dn@Xnm1) :
            print("Warning : non-diagonalizable matrix \n{M}")
        return ush*Bn@Czz, Bn, Czx, Xn, Xnm1, np.diag(1.0/vp[:rank])
    #--------------------------------------------------
    def MatricesI(self, rhs_layer, W, Nu=0.0) :
        """Matrices Ass+, Ass-, Bss, Mss, Xss, Yss=Xss^-1 and diagonal
           of the inverses of the nonzero eigenvalues at the interface
           between the current lhs layer and and rhs_layer."""
        d00,ushL,ushR = self.fds.d100,1.0/self.h,1.0/rhs_layer.h
        CzxL,CzyL = self.__Kzx(0.0,W),self.__Kzy(0.0,W)
        CzzL = self.__Kzz(0.0,W)
        CzxR,CzyR = rhs_layer.__Kzx(0.0,W),rhs_layer.__Kzy(0.0,W)
        CzzR = rhs_layer.__Kzz(0.0,W)
            # CzyL == CzyR : no interfacial state vector
        M_to_inv = -d00*(ushR*CzzR+ushL*CzzL) + 1.0j*Nu*(CzyR-CzyL)        
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bss = np.linalg.inv(M_to_inv)
        AssL,AssR = ushL*Bss@CzzL,ushR*Bss@CzzR
        diff_Czx = CzxR-CzxL
        if np.abs(diff_Czx).max() <= 1e-10*min(np.abs(CzxL).max(),\
                                               np.abs(CzxR).max()) :
            return AssL, AssR, Bss, None, None, None
        else :
            Mss = Bss@diff_Czx
            vp,Xss = np.linalg.eig(Mss)
            abs_vp = np.abs(vp)
            max_vp = abs_vp.max()
            nonzero_vp = (abs_vp >= 1e-10*max_vp)*1
            rank = nonzero_vp.sum()
            vp *= nonzero_vp
            if nonzero_vp[rank:].sum() > 0 : # unsorted eigenvalues
                Lmi = [ (m,i) for i,m in enumerate(abs_vp) ]
                Lmi.sort(reverse=True)
                indexes = np.array([ c[1] for c in Lmi ])
                vp = vp[indexes]
                Xss = Xss[:,indexes]
            Dss,Yss = np.diag(vp),np.linalg.inv(Xss)
            if not np.allclose(Mss,Xss@Dss@Yss) :
                print("Warning : non-diagonalizable matrix \n{M}")            
        return AssL, AssR, Bss, Xss, Yss, np.diag(1.0/vp[:rank]) 
    #-------------------------------------------------- 
    # Fixed slowness
    #--------------------------------------------------
    def P0(self, S) :
        """Matrix of coefficients of w(z)."""
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        I,Ip3 = np.arange(0,self.N,6), np.arange(3,self.N,6)
        for d in (0,1,2) :
            M[I+d,Ip3+d] = 1.0
        return M
    #--------------------------------------------------
    def P1(self, S) :
        """Matrix of coefficients of w'(z)."""
        Cxx,Cxz,Czx = self.__Cxx(0),self.__Cxz(0),self.__Czx(0)
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
            Cxx,Cxz,Czx = [sm.toarray() for sm in [Cxx,Cxz,Czx]]
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        if np.abs(S*self.__x_speeds-1).min() < 1e-5 :
            # Degenerate case : S coincide with 1/c_i
            return None
        Q = np.linalg.inv(S**2*Cxx-self.material.rho*np.eye(3))
        Q = S*Q.dot(Cxz+Czx)
        for i in np.arange(3,self.N,6) :
            j = i+3
            M[i:j,i:j] += Q
        return M
    #--------------------------------------------------
    def P2(self, S) :
        """Matrix of coefficients of w''(z)."""
        Cxx,Czz = self.__Cxx(0),self.__Czz(0)
        if GeneralDiscretizedLayer.SPARSE :
            M = sprs.lil_matrix( (self.N,self.N), dtype=np.complex)
            Cxx,Czz = [sm.toarray() for sm in [Cxx,Czz]]
        else :
            M = np.zeros( (self.N,self.N), dtype=np.complex)
        if np.abs(S*self.__x_speeds-1).min() < 1e-5 :
            # Degenerate case: S ~ 1/c_i
            return None
        Q = np.linalg.inv(S**2*Cxx-self.material.rho*np.eye(3))
        Q = -Q.dot(Czz)
        for i in np.arange(0,self.N,6) :
            j,k = i+3,i+6
            M[j:k,i:j] += Q
        return M
    #--------------------------------------------------
    def FS_matrices_0(self, S, left_fluid=None, sign=1) :
        """Matrices Aur, Auu, Avr, Avv, Brr, Brv for the left bound."""
        musd00,L03,h = -1.0/self.fds.d100,self.fds.L1up(3),self.h
        Czx,Czz = self.__Czx(0.0),self.__Czz(0.0)
        M_to_inv = Czz
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        B0 = h*musd00*np.linalg.inv(M_to_inv)
        M = (B0@Czx).astype(np.complex128)
        if left_fluid is not None :
            c0,rho0 = left_fluid.c,left_fluid.rho
            if abs(S*c0-1.0) < 1e-5 : # Degenerate case: S ~ 1/c0
                return None
            tau0 = np.sqrt(complex(1.0/c0**2-S**2))
            if abs(tau0.real)<1e-10*abs(tau0.imag) :
                tau0 = -1.0j*abs(tau0.imag)
            M[:,-1] += sign*rho0/(S*tau0)*B0[:,-1]
        Lmbda,X0T,X0P,Y0T,Y0P = self.diagonalize(M)
        Aur = -S*X0T@Lmbda
        Auu = musd00*L03
        Avr = X0T
        Avv = X0P@Y0P@Auu
        Brr = (-1.0/S)*np.linalg.inv(Lmbda)
        Brv = -Brr@Y0T@Auu
        return Aur, Auu, Avr, Avv, Brr, Brv
    #--------------------------------------------------
    def FS_matrices_n(self, S, right_fluid=None, sign=1) :
        """Matrices Aur, Auu, Avr, Avv, Brr, Brv for the right bound."""
        usd00,Ln3,h = 1.0/self.fds.d100,self.fds.L1down(3),self.h
        Czx,Czz = self.__Czx(self.e),self.__Czz(self.e)
        M_to_inv = Czz
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bn = h*usd00*np.linalg.inv(M_to_inv)
        M = (Bn@Czx).astype(np.complex128)
        if right_fluid is not None :
            cn,rhon = right_fluid.c,right_fluid.rho
            if abs(S*cn-1.0) < 1e-5 : # Degenerate case: S ~ 1/cn
                return None
            taun = np.sqrt(complex(1.0/cn**2-S**2))
            if abs(taun.real)<1e-10*abs(taun.imag) :
                taun = -1.0j*abs(taun.imag)
            M[:,-1] += -sign*rhon/(S*taun)*Bn[:,-1]
        Lmbda,XnT,XnP,YnT,YnP = self.diagonalize(M)
        Aur = -S*XnT@Lmbda
        Auu = usd00*Ln3
        Avr = XnT
        Avv = XnP@YnP@Auu
        Brr = (-1.0/S)*np.linalg.inv(Lmbda)
        Brv = -Brr@YnT@Auu
        return Aur, Auu, Avr, Avv, Brr, Brv
    #--------------------------------------------------
    def matrices_fluid_at_right(self, right_layer, threshold_ratio=1e-8) :
        d00 = self.fds.d100
        hL = self.h
        CzxL,CzzL = self.__Czx(self.e), self.__Czz(self.e)
        M_to_inv = (-d00 / hL) * CzzL
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bn = np.linalg.inv(M_to_inv) # 3-by-3 matrix
        Bstar = np.linalg.inv(M_to_inv[:2,:2]) # 2-by-2 matrix
        Czxstar,Czzstar = CzxL[:2,:2], CzzL[:2,:2]
        if np.abs(Czxstar).max() < threshold_ratio*np.abs(CzzL).max() :
            # Matrice nulle
            return 5*(None,)+(Bstar,Bn,CzxL,CzzL)
        Lmbda,XT,XP,YT,YP = self.diagonalize(Bstar@Czxstar)
        return Lmbda,XT,XP,YT,YP,Bstar,Bn,CzxR,CzzR
    #--------------------------------------------------
    def matrices_fluid_at_left(self, left_layer, threshold_ratio=1e-8) :
        d00 = self.fds.d100
        hR = self.h
        CzxR,CzzR = self.__Czx(0.0), self.__Czz(0.0)
        M_to_inv = (-d00 / hR) * CzzR
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        B0 = np.linalg.inv(M_to_inv) # 3-by-3 matrix
        Bstar = np.linalg.inv(M_to_inv[:2,:2]) # 2-by-2 matrix
        Czxstar,Czzstar = CzxR[:2,:2], CzzR[:2,:2]
        if np.abs(Czxstar).max() < threshold_ratio*np.abs(CzzR).max() :
            # Matrice nulle
            return 5*(None,)+(Bstar,B0,CzxR,CzzR)
        Lmbda,XT,XP,YT,YP = self.diagonalize(Bstar@Czxstar)
        return Lmbda,XT,XP,YT,YP,Bstar,B0,CzxR,CzzR
    #--------------------------------------------------
    def matrices_solid_at_left(self,left_layer, threshold_ratio=1e-8) :
        d00,Ln3,L03 = self.fds.d100,self.fds.L1down(3),self.fds.L1up(3)
        hL,hR = left_layer.h,self.h
        CzxL,CzzL = left_layer.__Czx(left_layer.e), \
                    left_layer.__Czz(left_layer.e)
        CzxR,CzzR = self.__Czx(0.0), self.__Czz(0.0)
        AL,AR = CzzL/hL,CzzR/hR
        M_to_inv = -d00 * (AL + AR)
        if GeneralDiscretizedLayer.SPARSE :
            M_to_inv = M_to_inv.toarray()
        Bss = np.linalg.inv(M_to_inv)
        mALL,ALR =  -Bss@AL@Ln3,Bss@AR@L03
        diff_Czx = CzxR-CzxL
        if np.abs(diff_Czx).max() < threshold_ratio*np.abs(CzxR).max() :
            # Matrice nulle
            return 5*(None,)+(mALL,ALR)
        Lmbda,XT,XP,YT,YP = self.diagonalize(Bss@diff_Czx)
        return Lmbda,XT,XP,YT,YP,mALL,ALR 
    #--------------------------------------------------
    def partial_waves(self, K, W, Nu=0.0) :
        """Returns the 6 vertical wavenumbers and the 
           displacement polarizations."""
        m = self.material
        K,W,Nu = self.homogenized_parameters(K,W,Nu)
        M = np.zeros( np.shape(K)+(6,6), dtype=np.complex128 )
        M[...,:3,-3:] = np.eye(3)
        nonm1 = np.linalg.inv(m.non)
        Ak2,Aknu,Anu2 = nonm1@m.lol,nonm1@(m.lom+m.mol),nonm1@m.mom
        M[...,-3:,:3] = np.multiply.outer(-m.rho*W**2,nonm1) + \
                        np.multiply.outer(K**2,Ak2) + \
                        np.multiply.outer(K*Nu,Aknu) + \
                        np.multiply.outer(Nu**2,Anu2)
        Bk,Bnu = -1.0j*nonm1@(m.lon+m.nol),-1.0j*nonm1@(m.mon+m.nom)
        M[...,-3:,-3:] = np.multiply.outer(K,Bk) + \
                         np.multiply.outer(Nu,Bnu)
        Kappa, P = np.linalg.eig(M)
        # Sorting with respect to vertical wavenumbers:
        indexes = np.argsort(np.round(Kappa,0))
        Kappa = -1.0j*np.take_along_axis(Kappa, indexes, -1)
        Kappa = Kappa[...,[3,4,5,2,1,0]]
        polar = P[...,:3,:].swapaxes(-1,-2)
        for i in (0,1,2) :
            polar[...,i] = np.take_along_axis(polar[...,i], indexes, -1)
        # Normalization of the mode shapes
        polar /= np.multiply.outer(np.linalg.norm(polar,axis=-1),[1,1,1])
        P = polar.swapaxes(-1,-2)[...,[3,4,5,2,1,0]]
        return Kappa,P
    #--------------------------------------------------
    def mode_shape(self, disc_shape, K, W, Nu=0.0) :
        """Returns the vertical wavenumbers 'Kz' and the amplitudes
           'A' of the partial waves contributing to the mode of
           discretized shape 'disc_shape'.
           For fluids, any scalar field f at position z is:
               f(z) = A[0]*exp(-i*Kz[0]*z) + A[1]*exp(i*Kz[1]*(e-z))
           For solids, the displacement field u at position z is:
               u(z) = A[:,:3]@exp(-i*Kz[:3]*z) +
                      A[:,-3:]@exp(i*Kz[-3:]*(e-z))."""
        msg = "SolidDiscretizedLayer.mode_shape:\n\tError:"
        K,W,Nu = self.homogenized_parameters(K,W,Nu)
        ps,pm,_,_,pd = self.ein_subs( np.ndim(K) )
        Z,e = self.Z[1:-1],self.e # positions of inner nodes, thickness
        Kz,P = self.partial_waves(K, W, Nu)
        K_shp = np.shape(K)
        shp = K_shp + (3*(self.n-1),)
        if np.shape(disc_shape) != K_shp + (3*(self.n-1),) :
            msg += " incompatible shape between\n\t\t" + \
                   "W, K, Nu {} -> {} and\n\t\t".format(K_shp,shp) +\
                   "discretized mode shape {}".format( \
                       disc_shape.shape)
            raise ValueError(msg)
        E = np.empty( K_shp + (6,self.n-1), dtype=np.complex128 )
        for j in (0,1,2) :
            E[...,j,:] = np.exp(-1.0j*np.multiply.outer(Kz[...,j],Z))
        for j in (3,4,5) :
            E[...,j,:] = np.exp(1.0j*np.multiply.outer(Kz[...,j],(e-Z)))
        P = np.swapaxes(P,-1,-2)
        _,_,_,po,_ = self.ein_subs( np.ndim(K)+1 )
        At = np.einsum(po, E, P)
        At.shape = At.shape[:-2] + (At.shape[-2]*At.shape[-1],)
        A_star, A = np.conjugate(At), np.swapaxes(At,-1,-2)
        M = A_star@A
        B = np.einsum(pm, A_star, disc_shape)
        nrm = np.sqrt( np.einsum(ps, np.conjugate(disc_shape), \
                                     disc_shape).real )
        X = np.linalg.solve(M,B)
        err = np.einsum(pm, A, X) - disc_shape
        err = np.sqrt( np.einsum(ps, np.conjugate(err), err).real )
        C = np.einsum(pd, P, X).swapaxes(-1,-2)
        return Kz, C, err/nrm
    #--------------------------------------------------
    def shape_function(self, field, k, f, Kz, C, nu=0, zL=0.0) :
        """k is the wavenumber in the x-direction, w the frequency,
           Kz a (6,) vector, C a (3,6) matrix of coefficients,
           nu the wavenumber in the y-direction, zL the z-origin.
           Returns a ufunc which gives the shape of the field
           at any vertical position.
        """
        # Note that w is not used in this method. It will be
        #  used if velocities are added.
        #++++++++++++++++++++++++++++++++++++++++++++++
        if field in ("Ux","Uy","Uz") :
            def fonc_U(Vz, c, kz=Kz, z0=zL, z1=zL+self.e) :
                # c and kz are 6-dimensional vectors
                Z = np.clip(Vz, z0, z1)
                U = c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                    c[1]*np.exp(-1.0j*kz[1]*(Z-z0)) + \
                    c[2]*np.exp(-1.0j*kz[2]*(Z-z0)) + \
                    c[3]*np.exp(-1.0j*kz[3]*(Z-z1)) + \
                    c[4]*np.exp(-1.0j*kz[4]*(Z-z1)) + \
                    c[5]*np.exp(-1.0j*kz[5]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*U
            if field == "Ux" :
                def fonc_Ux(Vz, cx=C[0,:], f_U=fonc_U) :
                    return f_U(Vz, cx)                    
                return fonc_Ux
            if field == "Uy" :
                def fonc_Uy(Vz, cy=C[1,:], f_U=fonc_U) :
                    return f_U(Vz, cy)                    
                return fonc_Uy
            if field == "Uz" :
                def fonc_Uz(Vz, cz=C[2,:], f_U=fonc_U) :
                    return f_U(Vz, cz)                    
                return fonc_Uz
        #++++++++++++++++++++++++++++++++++++++++++++++
        if field in ("dUx/dz","dUy/dz","dUz/dz") :
            def fonc_dUdz(Vz, c, kz=Kz, z0=zL, z1=zL+self.e) :
                # c and kz are 6-dimensional vectors
                Z = np.clip(Vz, z0, z1)
                idU = kz[0]*c[0]*np.exp(-1.0j*kz[0]*(Z-z0)) + \
                      kz[1]*c[1]*np.exp(-1.0j*kz[1]*(Z-z0)) + \
                      kz[2]*c[2]*np.exp(-1.0j*kz[2]*(Z-z0)) + \
                      kz[3]*c[3]*np.exp(-1.0j*kz[3]*(Z-z1)) + \
                      kz[4]*c[4]*np.exp(-1.0j*kz[4]*(Z-z1)) + \
                      kz[5]*c[5]*np.exp(-1.0j*kz[5]*(Z-z1))
                return ((Vz>z0)&(Vz<=z1))*(-1j*idU)
            if field == "dUx/dz" :
                def fonc_dUx(Vz, cx=C[0,:], f_dU=fonc_dUdz) :
                    return f_dU(Vz, cx)                    
                return fonc_dUx
            if field == "dUy/dz" :
                def fonc_dUy(Vz, cy=C[1,:], f_dU=fonc_dUdz) :
                    return f_dU(Vz, cy)                    
                return fonc_dUy
            if field == "dUz/dz" :
                def fonc_dUz(Vz, cz=C[2,:], f_dU=fonc_dUdz) :
                    return f_dU(Vz, cz)                    
                return fonc_dUz
        #++++++++++++++++++++++++++++++++++++++++++++++
        if field in ("Sxx","Sxy","Sxz","Syy","Syz","Szz") :
            def fonc_Sig(Vz, du, dkzu, kz=Kz, c=C, z0=zL, z1=zL+self.e) :
                Z = np.clip(Vz, z0, z1)
                E = np.array([np.exp(-1.0j*kz[0]*(Z-z0)), \
                              np.exp(-1.0j*kz[1]*(Z-z0)), \
                              np.exp(-1.0j*kz[2]*(Z-z0)), \
                              np.exp(-1.0j*kz[3]*(Z-z1)), \
                              np.exp(-1.0j*kz[4]*(Z-z1)), \
                              np.exp(-1.0j*kz[5]*(Z-z1))])
                # E : 6-by-d matrix (d is the dimension of Vz)
                U = c@E # 3-by-d matrix
                kzU = np.einsum("j,ij->ij",kz,c)@E # 3-by-d matrix
                return -1.0j*((Vz>z0)&(Vz<=z1))*( du@U + dkzu@kzU )
            lol = self.material.lol
            lom = self.material.lom
            lon = self.material.lon
            if field == "Sxx" :
                du = k*lol[0,:] + nu*lom[0,:]
                dkzu = lon[0,:]
                def fonc_Sxx(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Sxx
            if field == "Sxy" :
                du = k*lol[1,:] + nu*lom[1,:]
                dkzu = lon[1,:]
                def fonc_Sxy(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Sxy
            if field == "Sxz" :
                du = k*lol[2,:] + nu*lom[2,:]
                dkzu = lon[2,:]
                def fonc_Sxz(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Sxz
            mol = self.material.mol
            mom = self.material.mom
            mon = self.material.mon
            if field == "Syy" :
                du = k*mol[1,:] + nu*mom[1,:]
                dkzu = mon[1,:]
                def fonc_Syy(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Syy
            if field == "Syz" :
                du = k*mol[2,:] + nu*mom[2,:]
                dkzu = mon[2,:]
                def fonc_Syz(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Syz
            nol = self.material.nol
            nom = self.material.nom
            non = self.material.non
            if field == "Szz" :
                du = k*nol[2,:] + nu*nom[2,:]
                dkzu = non[2,:]
                def fonc_Szz(Vz, du=du, dkzu=dkzu, f_Sig=fonc_Sig) :
                    return  f_Sig(Vz, du, dkzu)
                return fonc_Szz
        #++++++++++++++++++++++++++++++++++++++++++++++
        if field in ("dSxz/dz", "dSyz/dz", "dSzz/dz") :
            def fonc_dSig(Vz, du, dkzu, kz=Kz, c=C, z0=zL, \
                          z1=zL+self.e) :
                Z = np.clip(Vz, z0, z1)
                iE = np.array([kz[0]*np.exp(-1.0j*kz[0]*(Z-z0)), \
                               kz[1]*np.exp(-1.0j*kz[1]*(Z-z0)), \
                               kz[2]*np.exp(-1.0j*kz[2]*(Z-z0)), \
                               kz[3]*np.exp(-1.0j*kz[3]*(Z-z1)), \
                               kz[4]*np.exp(-1.0j*kz[4]*(Z-z1)), \
                               kz[5]*np.exp(-1.0j*kz[5]*(Z-z1))])
                # iE : 6-by-d matrix (d is the dimension of Vz)
                iU = c@iE # 3-by-d matrix
                ikzU = np.einsum("j,ij->ij",kz,c)@iE # 3-by-d matrix
                return ((Vz>z0)&(Vz<=z1))*( -du@iU - dkzu@ikzU )
            nol = self.material.nol
            nom = self.material.nom
            non = self.material.non
            if field == "dSxz/dz" :
                du = k*nol[0,:] + nu*nom[0,:]
                dkzu = non[0,:]
                def fonc_dSxz(Vz, du=du, dkzu=dkzu, f_dSig=fonc_dSig) :
                    return  f_dSig(Vz, du, dkzu)
                return fonc_dSxz
            if field == "dSyz/dz" :
                du = k*nol[1,:] + nu*nom[1,:]
                dkzu = non[1,:]
                def fonc_dSyz(Vz, du=du, dkzu=dkzu, f_dSig=fonc_dSig) :
                    return  f_dSig(Vz, du, dkzu)
                return fonc_dSyz
            if field == "dSzz/dz" :
                du = k*nol[2,:] + nu*nom[2,:]
                dkzu = non[2,:]
                def fonc_dSzz(Vz, du=du, dkzu=dkzu, f_dSig=fonc_dSig) :
                    return  f_dSig(Vz, du, dkzu)
                return fonc_dSzz
        #++++++++++++++++++++++++++++++++++++++++++++++
        print(f"Program error: unexpected field '{field}'")
        return np.zeros_like
    #--------------------------------------------------
    @staticmethod
    def diagonalize(M, threshold_ratio=1e-8) :
        """M is a square matrix of dimension n.
           Return matrices Lambda,XT,XP,YT,YP.
           The rank r of M is the dimension of the diagonal matrix Lambda.
           XT : n-by-r ; XP : n-by-(n-r) ; YT : r-by-n ; YP : (n-r)-by-n.
           M = XT@Lambda@YT ; XT@YT + XP@YP = In ;
           YT@XT = Ir ; YP@XP = I(n-r) ; YT@XP = 0 ; YP@XT = 0.
        """
        Lmbda,X = np.linalg.eig(M)
        lambda_abs = np.abs(Lmbda)
        threshold = threshold_ratio*lambda_abs.max()
        indexes_T = np.where( lambda_abs >  threshold)[0]
        indexes_P = np.where( lambda_abs <= threshold)[0]
        XT,XP = X[:,indexes_T],X[:,indexes_P]
        Y = np.linalg.inv(X)
        YT,YP = Y[indexes_T,:],Y[indexes_P,:]
        Lmbda = np.diag(Lmbda[indexes_T])
        return Lmbda,XT,XP,YT,YP
#==========================================================================
#==   DiscretizedMultilayerPlate                                         ==
#==========================================================================
class DiscretizedMultilayerPlate :  
    #--------------------------------------------------
    def __init__(self,material,e,nb,fds_order=8) :
        """First, a discretized monolayered plate is built, before
           appending other layers (from left to right)."""
        self.__fds = FDScheme(fds_order)
        self.__e = e # Thickness of the plate
        first_layer = DiscretizedLayer(material,e,nb,self.fds)
        self.__layers = [first_layer]
        self.__N = first_layer.N # Dimension of the global matrix
                                 # Before taking into account immersing
                                 # fluids
        self.__level = 0         # 0 : Vacuum / vacuum
                                 # 1 : Vacuum / fluid, fluid / Vacuum or
                                 #     Fluid 1 / fluid 1
                                 # 2 : Fluid 1 / fluid 2
        self.__L = 0 # Left-hand side :
                     #    0 : Vaccum
                     #    2 : Fluid in contact with a fluid
                     #    4 : Fluid in contact with a solid
        self.__R = 0 # Right-hand side (0, 2 or 4) too
        self.__update_T() # Dimension of the global matrix for the
                          # eigenvalue formulation with respect to i*omega
                          # Level 0 : T = N
                          # Level 1 : T = 2*N + L + R
                          # Level 2 : T = 4*N + 2*L + 2*R
        self.__left_fluid = None  # Vacuum by default
        self.__right_fluid = None # Vacuum by default
    #--------------------------------------------------
    def __str__(self) :
        nb_lay = self.nb_layers
        dblline,line = 50*"=",50*"-"
        txt = dblline + f"\n Plate made of {nb_lay} layer"
        if nb_lay > 1 : txt += "s"
        txt += " : \n" + line + "\n "
        if self.__left_fluid is None :
            txt += "(Vacuum)"
        elif self.__left_fluid == "Wall" :
            txt += "Wall"
        else :
            txt += f"Half-space of '{self.__left_fluid.name}'"
        txt += "\n" + line + "\n "
        for lay in self.layers :
            txt += lay.__str__()
            txt += "\n" + line + "\n "
        if self.__right_fluid is None :
            txt += "(Vacuum)"
        elif self.__right_fluid == "Wall" :
            txt += "Wall"
        else :
            txt += f"Half-space of '{self.__right_fluid.name}'"
        return txt + "\n" + dblline
    #--------------------------------------------------
    def __update_T(self) :
        if self.__level == 0 :
            self.__T = self.__N
        elif self.__level == 1 : 
            self.__T = 2*self.__N + self.__L + self.__R
        else : # self.__level == 2
            self.__T = 4*self.__N + 2*(self.__L + self.__R)
    #--------------------------------------------------
    @property
    def fds(self) : return self.__fds
    @property
    def layers(self) : return tuple(self.__layers)
    @property
    def e(self) : return self.__e
    @property
    def thickness(self) : return self.__e
    @property
    def nb_layers(self) : return len(self.__layers) 
    @property
    def node_positions(self) :
        """Vector of node positions (without bounds and interfaces)."""
        LZ,z0 = [],0.0
        for lay in self.layers :
            LZ.extend( [ z0 + z for z in lay.Z[1:-1] ] )
            z0 += lay.Z[-1]
        return np.array(LZ)
    @property
    def dim(self) : return self.__T   
    @property
    def T(self) : return self.__T   
    @property
    def N(self) : return self.__N   
    @property
    def L(self) : return self.__L   
    @property
    def R(self) : return self.__R   
    @property
    def level(self) : return self.__level   
    @property
    def left_fluid(self) :
        """Fluid for z < 0 (None if vacuum)."""
        return self.__left_fluid
    @property
    def right_fluid(self) :
        """Fluid for z > e (None if vacuum)."""
        return self.__right_fluid 
    #--------------------------------------------------
    def add_discretizedLayer(self,material,e,nb,position=-1) :
        """Add a discretized layer at given position (between -n
          and n, where n denotes the previous number of layers).
               position = 0 : first layer
               position = 1 : second layer (just behind the first layer)
               position = -1 or n : last layer
               position = -2 or n-1 : just before the last layer.
        """
        new_layer = DiscretizedLayer(material,e,nb,self.fds)
        self.__e += new_layer.e
        n = len(self.__layers)
        if position < -n-1 :
            msg = "DiscretizedMultilayerPlate.add_discretizedLayer ::"
            msg += f"\n\tWarning: position {position} < -{n+1}"
            msg += f"\n\t         layer added in first position."
            print(msg)
            position = 0
        elif position > n :
            msg = "DiscretizedMultilayerPlate.add_discretizedLayer ::"
            msg += f"\n\tWarning: position {position} > {n}"
            msg += f"\n\t         layer added in last position."
            print(msg)
            position = n            
        elif position < 0 :
            position += n+1
        self.__layers.insert(position,new_layer)
        self.__N += new_layer.N
        if position == 0 : # first layer
            if self.__L == 2 : # Fluid/Fluid previously
                if new_layer.state_dim == 6 : # Fluid/Solid now
                    self.__N += 2 # additional state vector Solid0/Fluid1
                    self.__L = 4
                    self.__update_T()
            elif self.__L == 4 : # Fluid/Solid previously
                if new_layer.state_dim == 2 : # Fluid/Fluid now
                    self.__N += 2 # additional state vector Fluid0/Solid1
                    self.__L = 2
                    self.__update_T()
            else : # Vacuum or Wall
                cd = (new_layer.state_dim,self.layers[1].state_dim)
                if cd in [(2,6),(6,2)]: # Fluid0/Solid1 or Solid0/Fluid1
                    self.__N += 2 # additional state vector 
        elif position == n : # last layer
            if self.__R == 2 : # Fluid/Fluid previously
                if new_layer.state_dim == 6 : # Solid/Fluid now
                    self.__N += 2 # additional state vector Flu-2/Sol-1
                    self.__R = 4
                    self.__update_T()
            elif self.__R == 4 : # Solid/Fluid previously
                if new_layer.state_dim == 2 : # Fluid/Fluid now
                    self.__N += 2 # additional state vector Sol-2/Flu-1
                    self.__R = 2
                    self.__update_T()
            else : # Vacuum or Wall
                cd = (self.layers[-2].state_dim,new_layer.state_dim)
                if cd in [(2,6),(6,2)]: # Fluid-2/Solid-1, Solid-2/Fluid-1
                    self.__N += 2 # additional state vector 
        else : # layer inside
            sdp,sdc,sdn = [self.__layers[position+i].state_dim \
                                                for i in (-1,0,1)]
            if sdp == 2 :
                if sdn == 2 :
                    if sdc == 6 : # Fluid/Solid/Fluid
                        self.__N += 4 # 2 additional interfacial state vec.
                    else :
                        pass # No change: Fluid/Fluid/Fluid
                else : # sdn == 6
                    pass # No change: Fluid/Solid/Solid
                         #            or Fluid/Fluid/Solid
            else :  # sdp == 6
                if sdn == 2 :
                    pass # No change: Solid/Solid/Fluid
                         #            or Solid/Fluid/Fluid
                else :  # sdn == 6
                    if sdc == 2 : # Solid/Fluid/Solid
                        self.__N += 4 # 2 additional interfacial state vec.
                    else :
                        pass # No change: Solid/Solid/Solid
    #--------------------------------------------------
    def set_left_fluid(self,material) :
        """Add a fluid at the left of the first layer."""
        if material is None :
            if self.__left_fluid is None :
                return # Nothing changes
            self.__L = 0
            if self.__right_fluid in (None,"Wall") :
                # Vacuum/Vacuum or Vacuum/Wall
                self.__level = 0
            else : # Right-hand side: fluid
                self.__level = 1
        elif material == "Wall" :
            if self.__layers[0].state_dim == 2 : # fluid first layer                
                if self.__left_fluid == "Wall" :
                    return # Nothing changes
                self.__L = 0
                if self.__right_fluid in (None,"Wall") :
                    # Wall/Vacuum or Wall/Wall
                    self.__level = 0
                else : # Right-hand side: fluid
                    self.__level = 1
            else :
                msg = "DiscretizedMultilayerPlate.set_left_fluid:\n\t" + \
                  "Cannot set left medium as 'Wall' if the" + \
                  "\n\t\tfirst layer is solid"
                print(msg)
                return
        elif not isinstance(material, Fluid) :
            msg = "DiscretizedMultilayerPlate.set_left_fluid:\n\t" + \
                  "Cannot set left medium as " + \
                 f"'{material.__class__.__name__}'"
            print(msg)
            return
        else : # material is a fluid
            if self.__layers[0].state_dim == 2 : # fluid first layer
                self.__L = 2
            else : # state_dim == 6 solid first layer
                self.__L = 4
            if self.__right_fluid in (None,"Wall") :
                # Right-hand side: vacuum or wall
                self.__level = 1            
            elif material.c == self.__right_fluid.c : # same sound speed
                self.__level = 1
            else :
                self.__level = 2
        self.__update_T()
        self.__left_fluid = material
    #--------------------------------------------------
    def set_right_fluid(self,material) :
        """Add a fluid at the right of the last layer."""
        if material is None :
            if self.__right_fluid is None :
                return # Nothing changes
            self.__R = 0
            if self.__left_fluid in (None,"Wall") :
                # Vacuum/vacuum or Wall/vacuum
                self.__level = 0
            else : # Left-hand side: fluid
                self.__level = 1
        elif material == "Wall" :
            if self.__layers[-1].state_dim == 2 : # fluid last layer
                if self.__right_fluid == "Wall" :
                    return # Nothing changes
                self.__R = 0
                if self.__left_fluid in (None,"Wall") :
                    # Vacuum/wall or Wall/wall
                    self.__level = 0
                else : # Left-hand side: fluid
                    self.__level = 1
            else :
                msg = "DiscretizedMultilayerPlate.set_left_fluid:\n\t" + \
                  "Cannot set left medium as 'Wall' if the" + \
                  "\n\t\tlast layer is solid"
                print(msg)
                return
        elif not isinstance(material, Fluid) :
            msg = "DiscretizedMultilayerPlate.set_right_fluid:\n\t" + \
                  "Cannot set right medium as " + \
                 f"'{material.__class__.__name__}'"
            print(msg)
            return
        else : # material is a fluid
            if self.__layers[-1].state_dim == 2 : # fluid last layer
                self.__R = 2
            else : # state_dim == 6 solid last layer
                self.__R = 4
            if self.__left_fluid in (None,"Wall") :
                # Right-hand side : vacuum or wall
                self.__level = 1
            elif material.c == self.__left_fluid.c : # same sound speed
                self.__level = 1
            else :
                self.__level = 2
        self.__update_T()
        self.__right_fluid = material
    #--------------------------------------------------
    # Fixed wavenumber
    #--------------------------------------------------
    def global_matrix_fixed_wavenumber(self, K, Nu=0.0, sign=1) :
        """Returns M, L_indexes, L_positions.
           M is the global matrix, the eigenvalues of which are i*omega.
           L_indexes is the list of indexes of ux/phi in the matrix,
           for each solid/fluid layer. If the plate is in contact with
           an external fluid at z=0, L_indexes contains N (index shift
           for kappa0*U) followed by [index of phi0], and L_positions
           contains [0]. If the plate is in contact with an external
           fluid at z=e, L_indexes contains Nstar (index shift
           for kappae*U) followed by [index of phie], and L_positions
           contains [e].
        """
        # 1 - global matrix for the multilayer plate only (N-by-N)
        shape,fds = (self.N,self.N), self.fds
        if GeneralDiscretizedLayer.SPARSE :
            G = sprs.lil_matrix(shape, dtype=np.complex)
        else :
            G = np.zeros(shape, dtype=np.complex)
        cur_layer = self.__layers[0]
        M,C0,Cn = cur_layer.matrices_fixed_wavenumber(K,Nu)
        C0_left = C0
        N,cur_mat = cur_layer.N,cur_layer.material
        #+++++++ Vacuum (or wall) at the left-hand side and first layer
        G[:N,:N] = M
        if self.left_fluid is None or self.left_fluid == "Wall":
            # Vacuum / Wall for fluid layer
            if isinstance(cur_mat,Fluid) : # Fluid layer
                if self.left_fluid == "Wall" :
                    Mw0 = -1.0/fds.d100 * C0@fds.L1up(2)
                    r,c = Mw0.shape
                    G[:r,:c] += Mw0
                else : # Vacuum
                    pass # Nothing to add in the global matrix G
            else : # Solid layer, self.left_fluid is necessary None
                A0,_ = cur_layer.A0_B0_W(K,Nu)
                if GeneralDiscretizedLayer.SPARSE :
                    Mw0 = C0@sprs.block_diag([A0,A0],"lil")@fds.L1up(6)
                else :
                    Mw0 = C0@block_diag(A0,A0)@fds.L1up(6)
                r,c = Mw0.shape
                G[:r,:c] += Mw0
        im1,i0,i1 = 0,0,N
        #+++++++ For post-processing of mode shapes
        mem_indexes, mem_pos = [],[]
        cum_pos = 0.0
        #+++++++ Other layers
        prev_layer,prev_mat = cur_layer,cur_mat
        for cur_layer in self.__layers[1:] :
            N,cur_mat = cur_layer.N,cur_layer.material
            M,C0,Cn_next = cur_layer.matrices_fixed_wavenumber(K,Nu)
            rhoL,rhoR = prev_mat.rho,cur_mat.rho
            hL,hR = prev_layer.h,cur_layer.h
            if isinstance(prev_mat,Fluid) :
                if isinstance(cur_mat,Fluid) : # Fluid/Fluid
                    i2,i3 = i1,i1+N
                    cf = 1.0/(fds.d100*(hL*rhoL+hR*rhoR))
                    ML,MR = cf*hR*fds.L1down(2),-cf*hL*fds.L1up(2)
                    jLmin,jLmax = i1-ML.shape[1],i1
                    jRmin,jRmax = i2,i2+MR.shape[1]
                    # wnL
                    iLmin,iLmax = i1-Cn.shape[0],i1
                        # WL
                    G[iLmin:iLmax,jLmin:jLmax] += \
                                    Cn@np.array([[rhoR,0],[0,rhoR]])@ML
                        # WR
                    G[iLmin:iLmax,jRmin:jRmax] += \
                                    Cn@np.array([[rhoR,0],[0,rhoL]])@MR
                    # w0R
                    iRmin,iRmax = i2,i2+C0.shape[0]
                        # WL
                    G[iRmin:iRmax,jLmin:jLmax] += \
                                    C0@np.array([[rhoL,0],[0,rhoR]])@ML
                        # WR
                    G[iRmin:iRmax,jRmin:jRmax] += \
                                    C0@np.array([[rhoL,0],[0,rhoL]])@MR
                else : # Fluid/Solid
                    i_phin,i_pn = i1,i1+1 # Additional interfacial
                                          # state vector                     
                    i2,i3 = i1+2,i1+2+N
                    A0,B0 = cur_layer.A0_B0_W(K,Nu)
                    # i*w*phin
                    G[i_phin,i_pn] = -1.0/prev_mat.rho
                    # i*w*pn
                    usbeta = 1.0/B0[-1,-1]
                        # PhiR
                    LPhiL = (usbeta/prev_layer.h)*fds.L1down(1)
                    JminL,JmaxL = i1-2*LPhiL.shape[1],i1
                    G[i_pn:i_pn+1,JminL:JmaxL:2] = LPhiL
                        # phin
                    cf = -fds.d100/prev_layer.h
                    G[i_pn,i_phin] = usbeta*cf
                        # VL
                    A0L03 = A0@fds.L1up(3)
                    LV = -usbeta*A0L03[-1:,:]
                    JminR,JmaxR = i2,i2+2*LV.shape[1]
                    for j in (0,1,2) :
                        G[i_pn:i_pn+1,JminR+3+j:JmaxR:6] = LV[:,j::3]
                    # wn = (phin,pn)
                    G[i1-Cn.shape[0]:i1,i_phin:i_phin+2] = Cn
                    # w0
                    Imin,Imax = i2,i2+C0.shape[0]
                        # u0
                    MU = C0[:,0:3]@A0L03
                    for j in (0,1,2) :
                        G[Imin:Imax,JminR+j:JmaxR:6] += MU[:,j::3]
                        # u0z from pn
                    G[Imin:Imax,i_pn:i_pn+1] = C0[:,0:3]@B0[:,-1:]
                        # v0x & v0y
                    MV = C0[:,3:5]@A0L03[:2,:]
                    for j in (0,1,2) :
                        G[Imin:Imax,JminR+3+j:JmaxR:6] += MV[:,j::3]
                        # v0z
                    MPhiL = (1.0/prev_layer.h)*C0[:,5:]@fds.L1down(1)
                    G[Imin:Imax,JminL:JmaxL:2] = MPhiL
                    G[Imin:Imax,i_phin:i_phin+1] = cf*C0[:,5:]
                #+++++++ For post-processing of mode shapes
                if im1 == i0 :
                    if i1 == i2 :
                        mem_indexes.append(np.arange(i0,i1,2))
                        mem_pos.append(prev_layer.Z[1:-1] + cum_pos)
                    else : # i1 < i2
                        mem_indexes.append(np.arange(i0,i2,2))
                        mem_pos.append(prev_layer.Z[1:] + cum_pos)
                else : # im1 < i0
                    if i1 == i2 :
                        mem_indexes.append(np.arange(im1,i1,2))
                        mem_pos.append(prev_layer.Z[:-1] + cum_pos)
                    else : # i1 < i2
                        mem_indexes.append(np.arange(im1,i2,2))
                        mem_pos.append(prev_layer.Z[:] + cum_pos)                       
            else :
                if isinstance(cur_mat,Fluid) : # Solid/Fluid 
                    i_phi0,i_p0 = i1,i1+1 # Additional interfacial
                                          # state vector                     
                    i2,i3 = i1+2,i1+2+N
                    An,Bn = prev_layer.An_Bn_W(K,Nu)
                    # i*w*phi0
                    G[i_phi0,i_p0] = -1.0/cur_mat.rho
                    # i*w*p0
                    usbeta = 1.0/Bn[-1,-1]
                        # VL
                    AnLn3 = An@fds.L1down(3)
                    LV = -usbeta*AnLn3[-1:,:]
                    JminL,JmaxL = i1-2*LV.shape[1],i1
                    for j in (0,1,2) :
                        G[i_p0:i_p0+1,JminL+3+j:JmaxL:6] = LV[:,j::3]
                        # phi0
                    cf = fds.d100/cur_layer.h
                    G[i_p0,i_phi0] = usbeta*cf
                        # PhiR
                    LPhiR = (usbeta/cur_layer.h)*fds.L1up(1)
                    JminR,JmaxR = i2,i2+2*LPhiR.shape[1]
                    G[i_p0:i_p0+1,JminR:JmaxR:2] = LPhiR
                    # wn
                    Imin,Imax = i1-Cn.shape[0],i1
                        # un
                    MU = Cn[:,0:3]@AnLn3
                    for j in (0,1,2) :
                        G[Imin:Imax,JminL+j:JmaxL:6] += MU[:,j::3]
                        # unz from p0
                    G[Imin:Imax,i_p0:i_p0+1] = Cn[:,0:3]@Bn[:,-1:]
                        # vnx & vny
                    MV = Cn[:,3:5]@AnLn3[:2,:]
                    for j in (0,1,2) :
                        G[Imin:Imax,JminL+3+j:JmaxL:6] += MV[:,j::3]
                        # vnz
                    G[Imin:Imax,i_phi0:i_phi0+1] = cf*Cn[:,5:]
                    MPhiR = (1.0/cur_layer.h)*Cn[:,5:]@fds.L1up(1)
                    G[Imin:Imax,JminR:JmaxR:2] = MPhiR
                    # w0 = (phi0,p0)
                    G[i2:i2+C0.shape[0],i_phi0:i_phi0+2] = C0
                    hL,hR = prev_layer.h, cur_layer.h
                    
                else : # Solid/Solid                    
                    i2,i3 = i1,i1+N # No interfacial state vector
                    CL,ML = prev_layer.ICn_IMn_W(K,Nu)
                    CR,MR = cur_layer.IC0_IM0_W(K,Nu)
                    Mss = ML - MR
                    if GeneralDiscretizedLayer.SPARSE :
                        Mss = Mss.toarray()
                    Bss = np.linalg.inv(Mss)
                    AL,AR = Bss@CL,Bss@CR
                    L1n,L10 = fds.L1down(6),fds.L1up(6)
                    if GeneralDiscretizedLayer.SPARSE :
                        ML = sprs.block_diag([AL,AL],"lil")@L1n
                        MR = sprs.block_diag([AR,AR],"lil")@L10
                    else :
                        ML = block_diag(AL,AL)@L1n
                        MR = block_diag(AR,AR)@L10
                    IminL,ImaxL = i1-Cn.shape[0],i1
                    IminR,ImaxR = i2,i2+C0.shape[0]
                    JminL,JmaxL = i1-ML.shape[1],i1
                    JminR,JmaxR = i2,i2+MR.shape[1]
                    G[IminL:ImaxL,JminL:JmaxL] += Cn@ML
                    G[IminL:ImaxL,JminR:JmaxR] += -Cn@MR
                    G[IminR:ImaxR,JminL:JmaxL] += C0@ML
                    G[IminR:ImaxR,JminR:JmaxR] += -C0@MR
                #+++++++ For post-processing of mode shapes
                mem_indexes.append(np.arange(i0,i1,6))
                mem_pos.append(prev_layer.Z[1:-1] + cum_pos)
                
            G[i2:i3,i2:i3] = G[i2:i3,i2:i3] + M[:,:]
            #+++ updates for the next iteration +++
            cum_pos += prev_layer.e
            im1,i0,i1,prev_layer,prev_mat = i1,i2,i3,cur_layer,cur_mat
            Cn = Cn_next
        Cn_right = Cn
        #+++++++ For post-processing of mode shapes
        if isinstance(prev_mat,Fluid) :
            if im1 == i0 :
                mem_indexes.append(np.arange(i0,i1,2))
                mem_pos.append(prev_layer.Z[1:-1] + cum_pos)
            else : # im1 < i0
                mem_indexes.append(np.arange(im1,i1,2))
                mem_pos.append(prev_layer.Z[:-1] + cum_pos)
        else :
            mem_indexes.append(np.arange(i0,i1,6))
            mem_pos.append(prev_layer.Z[1:-1] + cum_pos)           
        #+++++++ Vacuum (or wall) at the right-hand side
        if self.right_fluid is None or self.right_fluid == "Wall":
            # Vacuum / Wall for fluid layer
            if isinstance(cur_mat,Fluid) : # Fluid layer
                if self.right_fluid == "Wall" :
                    Mwn = 1.0/fds.d100 * Cn@fds.L1down(2)
                    r,c = Mwn.shape
                    G[-r:,-c:] += Mwn
                else : # Vacuum
                    pass # Nothing to add in the global matrix G
            else : # Solid layer, self.right_fluid is necessary None
                An,_ = cur_layer.An_Bn_W(K,Nu)
                if GeneralDiscretizedLayer.SPARSE :
                    Mwn = Cn@sprs.block_diag([An,An],"lil")@fds.L1down(6)
                else :
                    Mwn = Cn@block_diag(An,An)@fds.L1down(6)
                r,c = Mwn.shape
                G[-r:,-c:] += Mwn
        #+++++++
        # 2 - global matrix for the immersed multilayer plate (T-by-T)
        if self.__level == 0 : # Vacuum (or wall)/vacuum (or wall)
            return G,mem_indexes,mem_pos
        L,R,N,T = self.L,self.R,self.N,self.T
        if GeneralDiscretizedLayer.SPARSE :
            H = sprs.lil_matrix( (T,T), dtype=np.complex)
        else :
            H = np.zeros( (T,T), dtype=np.complex)
        H[:N,:N] = G[:,:].copy()       # W
        H[N:2*N,N:2*N] = G[:,:].copy() # i*kappa0*W (ou i*kappae*W)
        # Left-hand side
        lay0 = self.__layers[0]
        if L == 2 : # First layer is a fluid in contact with a fluid
            I_phi0,I_p0 = 2*N,2*N+1 # Indexes of phi0 and p0
            mem_indexes.extend([N,[I_phi0]]) ; mem_pos.append([0])
            rho, rho0 = lay0.material.rho, self.__left_fluid.rho
            ratio_rho = rho0/rho
            d00, ush = fds.d100, 1.0/lay0.h
            K0 = rho0 * self.__left_fluid.c**2 # rho0*c0**2
            # i*omega*phi0
            H[I_phi0,I_p0] = -1.0/rho0
            # i*omega*p0
                # phi0
            rr0 = ratio_rho*d00*ush
            H[I_p0,I_phi0] = K0*(K**2+Nu**2-rr0**2)
                # Phi
            Mp0ik0Phi = -K0*ush*fds.L1up(1)
            Mp0Phi = rr0*Mp0ik0Phi
            Jmax = 2*Mp0Phi.shape[1]
            H[I_p0,0:Jmax:2] = Mp0Phi[0,::]
                # i*kappa0*Phi
            Jmin2,Jmax2 = N,N+Jmax
            H[I_p0,Jmin2:Jmax2:2] = Mp0ik0Phi[0,::]
            # C0@w0
            Imax,_ = C0_left.shape
            Mw0r0 = C0_left@np.array([[ratio_rho,0],[0,1]])
            H[:Imax,I_phi0:I_phi0+2] = Mw0r0                    
            # C0@(i*kappa0*w0)
                # (phi0,p0)
            Imin2,Imax2 = N,N+Imax
            H[Imin2:Imax2,I_phi0:I_phi0+2] = rr0*Mw0r0
                # W
            H[Imin2:Imax2,:Jmax] = ratio_rho*ush*C0_left@fds.L1up(2)
        elif L == 4 : # First layer is a solid in contact with a fluid
            A0,B0 = lay0.A0_B0_W(K,Nu)
            un_sur_beta = 1.0/B0[2,2]
            A0L1 = A0@fds.L1up(3)
            a = A0L1[2,:] # vector
            # Indexes of p0, psi0, phi0, v0z
            I_p0,I_psi0,I_phi0,I_v0z = 2*N,2*N+1,2*N+2,2*N+3
            mem_indexes.extend([N,[I_phi0]]) ; mem_pos.append([0])
            # i*omega*p0
            H[I_p0,I_v0z] = un_sur_beta # v0z/beta
                # -a.V/beta :
            Jmax = 2*a.shape[0]
            H[I_p0,3:Jmax:6] = -un_sur_beta*a[ ::3]
            H[I_p0,4:Jmax:6] = -un_sur_beta*a[1::3]
            H[I_p0,5:Jmax:6] = -un_sur_beta*a[2::3]
            # i*omega*psi0
            H[I_psi0,I_phi0] = 1.0  
            # i*omega*phi0
            H[I_phi0,I_p0] = -1.0/self.__left_fluid.rho  
            # i*omega*v0z
            usbr = un_sur_beta/self.__left_fluid.rho
            rhoc2 = self.__left_fluid.rho*self.__left_fluid.c**2
            H[I_v0z,I_p0] = usbr/rhoc2
            K2pNu2 = K**2 + Nu**2
            H[I_v0z,I_psi0] = -usbr*K2pNu2
                # a.(i*kappa*U)/(beta*rho0)
            Jmin,JmaxK = N,N+Jmax
            ua = usbr*a
            H[I_v0z,Jmin  :JmaxK:6] = ua[ ::3] 
            H[I_v0z,Jmin+1:JmaxK:6] = ua[1::3]
            H[I_v0z,Jmin+2:JmaxK:6] = ua[2::3]
            # u0
                # p0*B0@n
            Imax,_ = C0_left.shape
            if GeneralDiscretizedLayer.SPARSE :
                H[:Imax,I_p0] = C0_left[:,:3]@B0[:,-1:]
            else :
                H[:Imax,I_p0] = C0_left[:,:3]@B0[:,-1]
                # A0@L0@U
            WU = C0_left[:,:3]@A0L1
            Jmax = 2*WU.shape[1]
            H[:Imax, :Jmax:6] += WU[:, ::3]
            H[:Imax,1:Jmax:6] += WU[:,1::3]
            H[:Imax,2:Jmax:6] += WU[:,2::3]
            # v0
            if GeneralDiscretizedLayer.SPARSE :
                I12 = sprs.lil_matrix( (2,3), dtype = np.complex )
            else :
                I12 = np.zeros( (2,3), dtype = np.complex )
            I12[0,0],I12[1,1] = 1.0,1.0
            I12A0L1 = I12@A0L1
                # v0z*n 
            if GeneralDiscretizedLayer.SPARSE :           
                H[:Imax,I_v0z] = C0_left[:,-1:]
            else :         
                H[:Imax,I_v0z] = C0_left[:,-1]
                # I12@A0@L0*V
            WV = C0_left[:,-3:-1]@I12A0L1
            H[:Imax,3:Jmax:6] += WV[:, ::3]
            H[:Imax,4:Jmax:6] += WV[:,1::3]
            H[:Imax,5:Jmax:6] += WV[:,2::3]
            # i*kappa*u0
            Imin,ImaxK = N,N+Imax
            Jmin,JmaxK = N,N+Jmax
                # ((k2+Nu2)*psi0 - p0/(rho0*c2))*n
            if GeneralDiscretizedLayer.SPARSE :
                H[Imin:ImaxK,I_psi0] = K2pNu2*C0_left[:,2:3]
                H[Imin:ImaxK,I_p0] = -1.0/rhoc2*C0_left[:,2:3]
            else :
                H[Imin:ImaxK,I_psi0] = K2pNu2*C0_left[:,2]
                H[Imin:ImaxK,I_p0] = -1.0/rhoc2*C0_left[:,2]
                # I12*A0*L0*(i*kappa*U)
            WUk = C0_left[:,:2]@I12A0L1
            H[Imin:ImaxK,Jmin  :JmaxK:6] += WUk[:, ::3]
            H[Imin:ImaxK,Jmin+1:JmaxK:6] += WUk[:,1::3]
            H[Imin:ImaxK,Jmin+2:JmaxK:6] += WUk[:,2::3]
            # i*kappa*v0
            usbrhoc2 = un_sur_beta/rhoc2
            if GeneralDiscretizedLayer.SPARSE :
                # (k2+Nu2)*phi0
                H[Imin:ImaxK,I_phi0] = K2pNu2*C0_left[:,-1:]
                # - v0z/(beta*rho0*c2)*n
                H[Imin:ImaxK,I_v0z] = -usbrhoc2*C0_left[:,-1:]
            else :
                H[Imin:ImaxK,I_phi0] = K2pNu2*C0_left[:,-1]
                H[Imin:ImaxK,I_v0z] = -usbrhoc2*C0_left[:,-1]
                # 1/(beta*rho0*c2)*(a.V)*n
            aVn = usbrhoc2*C0_left[:,-1:]@[a]
            H[Imin:ImaxK,3:Jmax:6] += aVn[:, ::3]
            H[Imin:ImaxK,4:Jmax:6] += aVn[:,1::3]
            H[Imin:ImaxK,5:Jmax:6] += aVn[:,2::3]
                # I12*A0*L0*(i*kappa*V)
            WVk = C0_left[:,-3:-1]@I12A0L1
            H[Imin:ImaxK,Jmin+3:JmaxK:6] += WVk[:, ::3]
            H[Imin:ImaxK,Jmin+4:JmaxK:6] += WVk[:,1::3]
            H[Imin:ImaxK,Jmin+5:JmaxK:6] += WVk[:,2::3]
        # Right-hand side
        if self.__level == 2 : # Two different fluids
            Nstar = 2*N+L
            # i*kappae*W
            H[Nstar:2*Nstar,Nstar:2*Nstar] = H[:Nstar,:Nstar].copy()
        else :
            Nstar = N
        layn = self.__layers[-1]
        #=======================
        if R == 2 : # Last layer is a fluid in contact with a fluid
            if self.__level == 1 :
                I_phie,I_pe = T-2,T-1 # Indexes of phie and pe
            else : # self.__level == 2
                I_phie,I_pe = T-4,T-3 # Indexes of phie and pe 
            mem_indexes.extend([Nstar,[I_phie]])
            mem_pos.append([self.e])           
            rho, rhoe = layn.material.rho, self.__right_fluid.rho
            ratio_rho = rhoe/rho
            d00, ush = fds.d100, 1.0/layn.h
            Ke = rhoe * self.__right_fluid.c**2 # rhoe*ce**2
            # i*omega*phie
            H[I_phie,I_pe] = -1.0/rhoe
            # i*omega*pe
                # phie
            rre = ratio_rho*d00*ush
            H[I_pe,I_phie] = Ke*(K**2+Nu**2-rre**2)
                # Phi
            MpeikePhi = Ke*ush*fds.L1down(1)
            MpePhi = rre*MpeikePhi
            Jmin,Jmax = N-2*MpePhi.shape[1],N
            H[I_pe,Jmin:Jmax:2] = MpePhi[0,::]
                # i*sign*kappae*Phi
            Jmin2,Jmax2 = Nstar+Jmin,Nstar+Jmax
            H[I_pe,Jmin2:Jmax2:2] = sign*MpeikePhi[0,::]
            # Cn@wn
            Imin,Imax = N-Cn_right.shape[0], N
            Mwere = Cn_right@np.array([[ratio_rho,0],[0,1]])
            H[Imin:Imax,I_phie:I_phie+2] = Mwere                    
            # Cn@(i*sign*kappae*wn)
                # (phie,pe)
            Imin2,Imax2 = Nstar+Imin,Nstar+Imax            
            H[Imin2:Imax2,I_phie:I_phie+2] = sign*rre*Mwere
                # W
            H[Imin2:Imax2,Jmin:Jmax] = \
                         -sign*ush*ratio_rho*Cn_right@fds.L1down(2)
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.__level == 2 : # 2 fluids with different sound speeds
                H[-2:,-2:] = H[-4:-2,-4:-2].copy()
                H[N:2*N,-2:] = H[:N,-4:-2].copy() 
                H[Nstar+N:Nstar+2*N,-2:] = H[Nstar:Nstar+N,-4:-2].copy()
                H[-2:,N:2*N] = H[-4:-2,:N].copy() 
                H[-2:,Nstar+N:Nstar+2*N] = H[-4:-2,Nstar:Nstar+N].copy()
                H[N:2*N,Nstar+N:Nstar+2*N] = H[:N,Nstar:Nstar+N].copy()
                H[Nstar+N:Nstar+2*N,N:2*N] = H[Nstar:Nstar+N,:N].copy()
        #=======================
        elif R == 4 : # Last layer is a solid in contact with a fluid  
            An,Bn = layn.An_Bn_W(K,Nu)
            un_sur_beta = 1.0/Bn[2,2]
            AnL1 = An@fds.L1down(3)
            a = AnL1[2,:] # vector
            # Indexes of pe, psie, phie, vez
            if self.__level == 1 :
                I_pe,I_psie,I_phie,I_vez = T-4,T-3,T-2,T-1
            else : # self.__level == 2
                I_pe,I_psie,I_phie,I_vez = T-8,T-7,T-6,T-5 
            mem_indexes.extend([Nstar,[I_phie]])
            mem_pos.append([self.e])           
            # i*omega*pe
            H[I_pe,I_vez] = un_sur_beta # vez/beta
                # -a.V/beta :
            Jmin,Jmax = N-2*a.shape[0],N
            H[I_pe,Jmin+3:Jmax:6] = -un_sur_beta*a[ ::3]
            H[I_pe,Jmin+4:Jmax:6] = -un_sur_beta*a[1::3]
            H[I_pe,Jmin+5:Jmax:6] = -un_sur_beta*a[2::3]
            # i*omega*psie
            H[I_psie,I_phie] = 1.0  
            # i*omega*phie
            H[I_phie,I_pe] = -1.0/self.__right_fluid.rho  
            # i*omega*vez
            usbr = un_sur_beta/self.__right_fluid.rho
            rhoc2 = self.__right_fluid.rho*self.__right_fluid.c**2
            H[I_vez,I_pe] = usbr/rhoc2
            K2pNu2 = K**2 + Nu**2
            H[I_vez,I_psie] = -usbr*K2pNu2
                # -sign*a.(i*kappa*U)/(beta*rhoe)
            JminK,JmaxK = Jmin+Nstar,Jmax+Nstar
            ua = -sign*usbr*a
            H[I_vez,JminK  :JmaxK:6] = ua[ ::3] 
            H[I_vez,JminK+1:JmaxK:6] = ua[1::3]
            H[I_vez,JminK+2:JmaxK:6] = ua[2::3]
            # ue
                # pe*Bn@n
            Imin,Imax = N-Cn_right.shape[0],N
            if GeneralDiscretizedLayer.SPARSE :
                H[Imin:Imax,I_pe] = Cn_right[:,:3]@Bn[:,-1:]
            else :
                H[Imin:Imax,I_pe] = Cn_right[:,:3]@Bn[:,-1]                
                # An@Ln@U
            WU = Cn_right[:,:3]@AnL1
            Jmin = Jmax - 2*WU.shape[1]
            H[Imin:Imax,Jmin  :Jmax:6] += WU[:, ::3]
            H[Imin:Imax,Jmin+1:Jmax:6] += WU[:,1::3]
            H[Imin:Imax,Jmin+2:Jmax:6] += WU[:,2::3]
            if self.__level == 2 :
                IminN, ImaxN = Imin+N, Imax+N
                JminN, JmaxN = Jmin+N, Jmax+N
                H[IminN:ImaxN,JminN  :JmaxN:6] += WU[:, ::3]
                H[IminN:ImaxN,JminN+1:JmaxN:6] += WU[:,1::3]
                H[IminN:ImaxN,JminN+2:JmaxN:6] += WU[:,2::3]
            # ve
            if GeneralDiscretizedLayer.SPARSE :
                I12 = sprs.lil_matrix( (2,3), dtype = np.complex )
            else :
                I12 = np.zeros( (2,3), dtype = np.complex )
            I12[0,0],I12[1,1] = 1.0,1.0
            I12AnL1 = I12@AnL1
                # vez*n   
            if GeneralDiscretizedLayer.SPARSE :         
                H[Imin:Imax,I_vez] = Cn_right[:,-1:]
            else :
                H[Imin:Imax,I_vez] = Cn_right[:,-1]
                # I12@An@Ln*V
            WV = Cn_right[:,-3:-1]@I12AnL1
            H[Imin:Imax,Jmin+3:Jmax:6] += WV[:, ::3]
            H[Imin:Imax,Jmin+4:Jmax:6] += WV[:,1::3]
            H[Imin:Imax,Jmin+5:Jmax:6] += WV[:,2::3]
            if self.__level == 2 :
                H[IminN:ImaxN,JminN+3:JmaxN:6] += WV[:, ::3]
                H[IminN:ImaxN,JminN+4:JmaxN:6] += WV[:,1::3]
                H[IminN:ImaxN,JminN+5:JmaxN:6] += WV[:,2::3]
            # i*kappa*ue
            IminK,ImaxK = Imin+Nstar,Imax+Nstar
                # -sign*((k2+Nu2)*psie - pe/(rhoe*c2))*n  
            if GeneralDiscretizedLayer.SPARSE : 
                H[IminK:ImaxK,I_psie] = -sign*K2pNu2*Cn_right[:,2:3]
                H[IminK:ImaxK,I_pe] = sign/rhoc2*Cn_right[:,2:3]
            else :
                H[IminK:ImaxK,I_psie] = -sign*K2pNu2*Cn_right[:,2]
                H[IminK:ImaxK,I_pe] = sign/rhoc2*Cn_right[:,2]
                # I12*An*Ln*(i*kappa*U)
            WUk = Cn_right[:,:2]@I12AnL1
            H[IminK:ImaxK,JminK  :JmaxK:6] += WUk[:, ::3]
            H[IminK:ImaxK,JminK+1:JmaxK:6] += WUk[:,1::3]
            H[IminK:ImaxK,JminK+2:JmaxK:6] += WUk[:,2::3]
            if self.__level == 2 :
                IminKN,ImaxKN = IminK+N,ImaxK+N
                JminKN,JmaxKN = JminK+N,JmaxK+N
                H[IminKN:ImaxKN,JminKN  :JmaxKN:6] += WUk[:, ::3]
                H[IminKN:ImaxKN,JminKN+1:JmaxKN:6] += WUk[:,1::3]
                H[IminKN:ImaxKN,JminKN+2:JmaxKN:6] += WUk[:,2::3]
            # i*kappa*ve
            usbrhoc2 = un_sur_beta/rhoc2 
            if GeneralDiscretizedLayer.SPARSE : 
                # -sign*(k2+Nu2)*phie
                H[IminK:ImaxK,I_phie] = -sign*K2pNu2*Cn_right[:,-1:]
                # sign*vez/(beta*rhoe*c2)*n
                H[IminK:ImaxK,I_vez] = sign*usbrhoc2*Cn_right[:,-1:]
            else :
                H[IminK:ImaxK,I_phie] = -sign*K2pNu2*Cn_right[:,-1]
                H[IminK:ImaxK,I_vez] = sign*usbrhoc2*Cn_right[:,-1]
                # -sign/(beta*rhoe*c2)*(a.V)*n
            aVn = -sign*usbrhoc2*Cn_right[:,-1:]@[a]
            H[IminK:ImaxK,Jmin+3:Jmax:6] += aVn[:, ::3]
            H[IminK:ImaxK,Jmin+4:Jmax:6] += aVn[:,1::3]
            H[IminK:ImaxK,Jmin+5:Jmax:6] += aVn[:,2::3]
                # I12*An*Ln*(i*kappa*V)
            # WV = Cn_right[:,-3:-1]@I12AnL1
            H[IminK:ImaxK,JminK+3:JmaxK:6] += WV[:, ::3]
            H[IminK:ImaxK,JminK+4:JmaxK:6] += WV[:,1::3]
            H[IminK:ImaxK,JminK+5:JmaxK:6] += WV[:,2::3]
            if self.__level == 2 :
                H[IminKN:ImaxKN,JminKN+3:JmaxKN:6] += WV[:, ::3]
                H[IminKN:ImaxKN,JminKN+4:JmaxKN:6] += WV[:,1::3]
                H[IminKN:ImaxKN,JminKN+5:JmaxKN:6] += WV[:,2::3]                
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.__level == 2 : # 2 fluids with different sound speeds
                H[-4:,-4:] = H[-8:-4,-8:-4].copy()
                H[N:2*N,-4:] = H[:N,-8:-4].copy() 
                H[Nstar+N:Nstar+2*N,-4:] = H[Nstar:Nstar+N,-8:-4].copy()
                H[-4:,N:2*N] = H[-8:-4,:N].copy() 
                H[-4:,Nstar+N:Nstar+2*N] = H[-8:-4,Nstar:Nstar+N].copy()
                H[N:2*N,Nstar+N:Nstar+2*N] = H[:N,Nstar:Nstar+N].copy()
                H[Nstar+N:Nstar+2*N,N:2*N] = H[Nstar:Nstar+N,:N].copy()
        #=======================
        return H,mem_indexes,mem_pos     
    #--------------------------------------------------
    def first_frequency_values(self, K, Nu=0.0, nb=20, \
                               check_opposite=True, \
                               with_mode_shapes=False,
                               mode_shapes=["Ux,Uy,Uz"]) :
        """K and Nu can be scalars or vectors.
           nb is the number of wanted frequency values"""
        is_array = True
        if len(np.shape(K)) > 0 :
            if len(np.shape(Nu)) == 0 : # Vector K and scalar Nu
                Nu = Nu*np.ones_like(K)
            else :            # Vector K and vector Nu
                pass 
        elif len(np.shape(Nu)) > 0 :    # Scalar K and vector Nu
            K = K*np.ones_like(Nu)
        else :                # Scalar K and scalar Nu
            is_array = False
        if is_array :
            k,nu = K[0],Nu[0]
            G,LI,LP = self.global_matrix_fixed_wavenumber(k,nu)
            if GeneralDiscretizedLayer.SPARSE : G = G.toarray()
            shp = G.shape
            LG = [G]
            for k,nu in zip(K[1:],Nu[1:]) :
                G,_,_ = self.global_matrix_fixed_wavenumber(k,nu)
                if GeneralDiscretizedLayer.SPARSE : G = G.toarray()
                if G.shape != shp :
                    print("DiscretizedMultilayerPlate." + \
                          "first_frequency_values:\n\p" + \
                          "Vectorization not possible!")
                    return [ self.first_frequency_values(k, nu, nb=nb, \
                                       check_opposite=check_opposite, \
                                       with_mode_shapes=with_mode_shapes,\
                                       mode_shapes=mode_shapes) \
                             for k,nu in zip(K,Nu) ]
                LG.append(G)
            G = np.array( LG ) # "Vector of matrices"
        else :
            G,LI,LP = self.global_matrix_fixed_wavenumber(K,Nu)
            if GeneralDiscretizedLayer.SPARSE : G = G.toarray() # Matrix
        if with_mode_shapes :
            iW, T_shapes = np.linalg.eig(G)
        else :
            iW, T_shapes = np.linalg.eigvals(G), None
        #+++++++ ICI +++++++ ICI +++++++ ICI +++++++ ICI +++++++ ICI +++++
        T_freq = (-0.5j/np.pi)*iW
        frequencies, mode_shapes, emax = \
                        self.take_first(T_freq, nb, epsilon=10.0,\
                                        check_opposite=check_opposite, \
                                        array=T_shapes)
        if check_opposite :
            if emax > 1e-8*np.abs(T_freq).mean() :
                msg = "DiscretizedMultilayerPlate." + \
                      "first_frequency_values :\n\t" + \
                      "opposites are not all present"
                print(msg)
        if with_mode_shapes :

            return frequencies, mode_shapes
        else :
            return frequencies # eigenfrequencies only
    #--------------------------------------------------
    # Fixed frequency
    #--------------------------------------------------
    def global_matrix_fixed_frequency(self, F, Nu=0.0, sign=1) :
        """Returns M, L_indexes, L_positions.
           M is the global matrix, the eigenvalues of which are -i*k_x.
           L_indexes is the list of indexes of ux/psi in the matrix,
           for each solid/fluid layer. If the plate is in contact with
           an external fluid at z=0, L_indexes contains N (index shift
           for kappa0*U) followed by [index of phi0], and L_positions
           contains [0]. If the plate is in contact with an external
           fluid at z=e, L_indexes contains Nstar (index shift
           for kappae*U) followed by [index of phie], and L_positions
           contains [e].
        """
        SPARSE = GeneralDiscretizedLayer.SPARSE
        # 1 - Global matrix for the multilayer plate without immersing
        #     fluids.
        W,fds = 2*np.pi*F,self.fds
        # 1.a - First loop without filling the global state matrix
        LM,LC0,LCn,idx_min,idx_max = [],[],[],[],[]
        # Internal parameters of the first layer
        cur_layer = self.__layers[0]
        M,C0,Cn = cur_layer.matrices_fixed_frequency(F,Nu)
        LM.append(M) ; LC0.append(C0) ; LCn.append(Cn)
        N,cur_mat = cur_layer.N,cur_layer.material
        cur_imin,cur_imax,cum_pos = 0,N,0.0
        # First layer parameters for external interface
        if isinstance(cur_mat,Fluid) : # Fluid layer
            # nothing changes
            #+++++++ For post-processing of mode shapes
            mem_indexes = [np.arange(cur_imin,cur_imax,2)]
        else : # Solid layer
            A0, B0, Czx0, X0, Y0, G0 = cur_layer.Matrices0(W,Nu)
            rank0,_ = G0.shape
            cur_imin += rank0 ; cur_imax += rank0
            #+++++++ For post-processing of mode shapes
            mem_indexes = [np.arange(cur_imin,cur_imax,6)]
        mem_pos = [cur_layer.Z[1:-1]]
        #+++++++
        idx_min.append(cur_imin) ; idx_max.append(cur_imax)
        interfacial_matrices = []
        prev_layer, prev_mat = cur_layer, cur_mat
        for cur_layer in self.__layers[1:] :
            # Internal parameters of the layer
            M,C0,Cn = cur_layer.matrices_fixed_frequency(F,Nu)
            LM.append(M) ; LC0.append(C0) ; LCn.append(Cn)
            N,cur_mat = cur_layer.N,cur_layer.material
            cur_imin,cur_imax = cur_imax,cur_imax+N
            if isinstance(prev_mat,Fluid) :
                if isinstance(cur_mat,Fluid) : # Fluid/Fluid
                    # No additional interfacial state vector
                    interfacial_matrices.append(None)
                else : # Fluid/Solid
                    AI,BI,CzxI,XI,YI,GI = cur_layer.Matrices0(W,Nu)
                    interfacial_matrices.append([AI,BI,CzxI,XI,YI,GI])
                    rank,_ = GI.shape
                    cur_imin += rank ; cur_imax += rank
            else :
                if isinstance(cur_mat,Fluid) : # Solid/Fluid
                    AI,BI,CzxI,XI,YI,GI = prev_layer.Matricesn(W,Nu)
                    interfacial_matrices.append([AI,BI,CzxI,XI,YI,GI])
                    rank,_ = GI.shape
                    cur_imin += rank ; cur_imax += rank
                else : # Solid/Solid
                    AIL,AIR,BI,XI,YI,GI = \
                                    prev_layer.MatricesI(cur_layer,W,Nu)
                    interfacial_matrices.append([AIL,AIR,BI,XI,YI,GI])
                    if GI is not None : 
                        rank,_ = GI.shape
                        cur_imin += rank ; cur_imax += rank
            #+++++++ For post-processing of mode shapes
            if isinstance(cur_mat,Fluid) :
                mem_indexes.append(np.arange(cur_imin,cur_imax,2))
            else :
                mem_indexes.append(np.arange(cur_imin,cur_imax,6))
            cum_pos += prev_layer.e
            mem_pos.append(cur_layer.Z[1:-1]+cum_pos)
            #+++++++
            idx_min.append(cur_imin) ; idx_max.append(cur_imax)
            prev_layer, prev_mat = cur_layer, cur_mat  
        Gsize = cur_imax # size of the global state matrix which 
                         # depends on the interfaces 
        # Last layer parameters for external interface
        if isinstance(cur_mat,Fluid) : # Fluid layer
            pass # nothing changes
        else : # Solid layer
            An, Bn, Czxn, Xn, Yn, Gn = cur_layer.Matricesn(W,Nu)
            rankn,_ = Gn.shape
            Gsize += rankn     
        # 1.b - Building the global state matrix without the interfaces
        shape = (Gsize,Gsize)
        if SPARSE :
            G = sprs.lil_matrix(shape, dtype=np.complex)
        else :
            G = np.zeros(shape, dtype=np.complex)
        for b,e,M in zip(idx_min,idx_max,LM) :
            G[b:e,b:e] = M
        # 1.c - Left interface (with the first layer)
        C0 = LC0[0]
        if isinstance(self.layers[0].material,Fluid) :
            if self.left_fluid is None : # Vacuum
                pass # Nothing to do (w0=0)
            elif self.left_fluid == "Wall" :
                addM = (-1.0/fds.d100)*C0@fds.L1up(2)
                nR,nC = addM.shape
                G[:nR,:nC] += addM
            else : # Immersing fluid at the left-hand side
                pass # Done below
        else : # First layer is solid, r0T is added as a state vector
            C0u,C0v = C0[:,:3],C0[:,-3:]
            X0T, X0P = X0[:,:rank0],X0[:,rank0:]
            Y0T, Y0P = Y0[:rank0,:],Y0[rank0:,:]
            # u0
                # A0 L0 U
            A0L3 = A0@fds.L1up(3)
            MU = C0u@A0L3
            nr,nc = MU.shape
            i_min,i_max = idx_min[0],idx_min[0]+nr
            j_min,j_max = i_min,i_min+2*nc
            G[i_min:i_max,j_min  :j_max:6] += MU[:, ::3]
            G[i_min:i_max,j_min+1:j_max:6] += MU[:,1::3]
            G[i_min:i_max,j_min+2:j_max:6] += MU[:,2::3]
                # B0 Czx v0 (modification of C0v)
            #C0v = C0v + C0u@B0@Czx0
                # We can also use: X0T Lambda0 r0T
            X0TLamb0 = X0T@np.linalg.inv(G0)
            G[i_min:i_max,:j_min] += C0u@X0TLamb0
            # -i k r0T (G0=Lambda0^-1)
                # G0 r0T    
            G[:i_min,:j_min] += G0
                # -G0 Y0T A0 L0 V
            MV = -G0@Y0T@A0L3
            G[:i_min,j_min+3:j_max:6] += MV[:, ::3]
            G[:i_min,j_min+4:j_max:6] += MV[:,1::3]
            G[:i_min,j_min+5:j_max:6] += MV[:,2::3]
            # v0
                # X0T r0T
            G[i_min:i_max,:j_min] += C0v@X0T
                # X0P r0P = X0P Y0P A0 L0 V
            HV = C0v@X0P@Y0P@A0L3
            G[i_min:i_max:,j_min+3:j_max:6] += HV[:, ::3]
            G[i_min:i_max:,j_min+4:j_max:6] += HV[:,1::3]
            G[i_min:i_max:,j_min+5:j_max:6] += HV[:,2::3]
        # 1.d - Right interface (with the last layer)
        Cn = LCn[-1]
        if isinstance(self.layers[-1].material,Fluid) :
            if self.right_fluid is None : # Vacuum
                pass # Nothing to do (wn=0)
            elif self.right_fluid == "Wall" :
                addM = (1.0/fds.d100)*Cn@fds.L1down(2)
                nR,nC = addM.shape
                G[-nR:,-nC:] += addM
            else : # Immersing fluid at the left-hand side
                pass # Done below
        else : # Last layer is solid, rnT is added as a state vector
            Cnu,Cnv = Cn[:,:3],Cn[:,-3:]
            XnT, XnP = Xn[:,:rankn],Xn[:,rankn:]
            YnT, YnP = Yn[:rankn,:],Yn[rankn:,:]
            # un
                # An Ln U
            AnL3 = An@fds.L1down(3)
            MU = Cnu@AnL3
            (nr,nc),i_max = MU.shape,idx_max[-1]
            i_min,i_xxx = i_max-nr,i_max+rankn
            j_min,j_max,j_xxx = i_max-2*nc,i_max,i_xxx
            G[i_min:i_max,j_min  :j_max:6] += MU[:, ::3]
            G[i_min:i_max,j_min+1:j_max:6] += MU[:,1::3]
            G[i_min:i_max,j_min+2:j_max:6] += MU[:,2::3]
                # Bn Czx vn (modification of Cnv)
            #Cnv = Cnv + Cnu@Bn@Czxn
                # We can also use: XnT Lambdan rnT
            XnTLambn = XnT@np.linalg.inv(Gn)
            G[i_min:i_max,j_max:j_xxx] += Cnu@XnTLambn
            # -i k rnT (Gn=Lambda_n^-1)
                # Gn rnT
            G[i_max:i_xxx,j_max:j_xxx] += Gn
                # -Gn YnT An Ln V
            MV = -Gn@YnT@AnL3
            G[i_max:i_xxx,j_min+3:j_max:6] += MV[:, ::3]
            G[i_max:i_xxx,j_min+4:j_max:6] += MV[:,1::3]
            G[i_max:i_xxx,j_min+5:j_max:6] += MV[:,2::3]
            # vn
                # XnT rnT
            G[i_min:i_max,j_max:j_xxx] += Cnv@XnT
                # XnP rnP = XnP YnP An Ln V
            HV = Cnv@XnP@YnP@AnL3
            G[i_min:i_max:,j_min+3:j_max:6] += HV[:, ::3]
            G[i_min:i_max:,j_min+4:j_max:6] += HV[:,1::3]
            G[i_min:i_max:,j_min+5:j_max:6] += HV[:,2::3]
        # 1.e - Internal interfaces
        prev_imin, prev_imax = idx_min[0], idx_max[0]
        prev_layer = self.__layers[0]
        prev_mat = prev_layer.material
        for cur_layer,iL,iR,CR,CL,LIM in zip( self.__layers[1:], \
                    idx_max[:-1], idx_min[1:], LC0[1:], LCn[:-1], \
                    interfacial_matrices) :
            cur_mat = cur_layer.material
            d00 = fds.d100
            hL,hR = prev_layer.h, cur_layer.h
            rhoL,rhoR = prev_mat.rho, cur_mat.rho
            if isinstance(prev_mat,Fluid) :
                if isinstance(cur_mat,Fluid) : # Fluid/Fluid
                    # No additional interfacial state vector
                    denom = 1.0/(d00*(rhoL*hL+rhoR*hR))
                    LL,LR = denom*hR*fds.L1down(2), -denom*hL*fds.L1up(2)
                    CL *= rhoR
                    MLL = CL@LL
                    nR,nC = MLL.shape
                    G[iL-nR:iL,iL-nC:iL] += MLL
                    MLR = CL@LR
                    nR,nC = MLR.shape
                    G[iL-nR:iL,iR:iR+nC] += MLR
                    CR *= rhoL
                    MRL = CR@LL
                    nR,nC = MRL.shape
                    G[iR:iR+nR,iL-nC:iL] += MRL
                    MRR = CR@LR
                    nR,nC = MRR.shape
                    G[iR:iR+nR,iR:iR+nC] += MRR
                else : # Fluid/Solid
                    CLpsi,CLunx = CL[:,:1],CL[:,-1:]
                    CRu,CRv = CR[:,:3],CR[:,-3:]
                    AI,BI,CzxI,XI,YI,GI = LIM
                    b0 = BI[:,-1:]
                    rk,_ = GI.shape # rk = iR-iL
                    XT,XP = XI[:,:rk],XI[:,rk:]
                    YT,YP = YI[:rk,:],YI[rk:,:]
                    w2r = rhoL*W**2                   
                    # Matrices
                    XYP = XP@YP
                    AL = AI@fds.L1up(3)
                    XTLbda = XT@np.linalg.inv(GI)
                    # Coefficients
                    coef_psi = 1.0/(d00+hL*w2r*BI[-1,-1])
                    coef_unx = 1.0/(d00+hL*w2r*(XYP@b0)[-1,-1])
                    # Row-matrix
                    Ln1 = fds.L1down(1)
                    #----------------------------------------
                    # psin-
                        # Psi-
                    RpsiPsi = coef_psi*Ln1
                    MpsiPsi = CLpsi@RpsiPsi
                    diL,djL = MpsiPsi.shape
                    imin,jmin = iL-diL, iL-2*djL
                    G[imin:iL,jmin:iL:2] += MpsiPsi
                        # rT
                    RpsirT = -hL*coef_psi*XTLbda[-1:,:]
                    MpsirT = CLpsi@RpsirT
                    G[imin:iL,iL:iR] += MpsirT
                        # U+
                    RpsiUR = -hL*coef_psi*AL[-1:,:]
                    MpsiUR = CLpsi@RpsiUR
                    _,djR = AL.shape
                    jmax = iR + 2*djR
                    G[imin:iL,iR  :jmax:6] += MpsiUR[:, ::3]
                    G[imin:iL,iR+1:jmax:6] += MpsiUR[:,1::3]
                    G[imin:iL,iR+2:jmax:6] += MpsiUR[:,2::3]   
                    #----------------------------------------
                    # unx-
                        # Ux-
                    RunxUx = coef_unx*Ln1
                    MunxUx = CLunx@RunxUx
                    G[imin:iL,jmin+1:iL:2] += MunxUx
                        # rT
                    RunxrT = -hL*coef_unx*XT[-1:,:]
                    MunxrT = CLunx@RunxrT
                    G[imin:iL,iL:iR] += MunxrT
                        # V+
                    RunxVR = -hL*coef_unx*(XYP@AL)[-1:,:]
                    MunxVR = CLunx@RunxVR
                    G[imin:iL,iR+3:jmax:6] += MunxVR[:, ::3]
                    G[imin:iL,iR+4:jmax:6] += MunxVR[:,1::3]
                    G[imin:iL,iR+5:jmax:6] += MunxVR[:,2::3]
                    #----------------------------------------
                    # -i k rT
                        # rT
                    G[iL:iR,iL:iR] = GI
                    mGYT = -GI@YT
                        # V+
                    MV = mGYT@AL
                    G[iL:iR,iR+3:jmax:6] += MV[:, ::3]
                    G[iL:iR,iR+4:jmax:6] += MV[:,1::3]
                    G[iL:iR,iR+5:jmax:6] += MV[:,2::3]
                        # unx-  
                    Munx = w2r*mGYT@b0
                            # Ux-
                    MUx = Munx@RunxUx
                    _,djL = MUx.shape
                    jmin = iL - 2*djL
                    G[iL:iR,jmin+1:iL:2] += MUx
                            # rT
                    MrT = Munx@RunxrT
                    G[iL:iR,iL:iR] += MrT
                            # V+
                    MVcor = Munx@RunxVR
                    G[iL:iR,iR+3:jmax:6] += MVcor[:, ::3]
                    G[iL:iR,iR+4:jmax:6] += MVcor[:,1::3]
                    G[iL:iR,iR+5:jmax:6] += MVcor[:,2::3]  
                    #----------------------------------------
                    # uR
                    diR,_ = CRu.shape
                    imax = iR+diR
                        # rT
                    MuRrT = CRu@XTLbda
                    G[iR:imax,iL:iR] += MuRrT
                        # U+
                    MuRUR = CRu@AL
                    G[iR:imax,iR  :jmax:6] += MuRUR[:, ::3]
                    G[iR:imax,iR+1:jmax:6] += MuRUR[:,1::3]
                    G[iR:imax,iR+2:jmax:6] += MuRUR[:,2::3]
                        # psin
                    MuRpsi = w2r*CRu@b0
                            # PsiL
                    G[iR:imax,jmin:iL:2] += MuRpsi@RpsiPsi
                            # rT
                    G[iR:imax,iL:iR] += MuRpsi@RpsirT
                            # U+
                    MuRURcor = MuRpsi@RpsiUR
                    G[iR:imax,iR  :jmax:6] += MuRURcor[:, ::3]
                    G[iR:imax,iR+1:jmax:6] += MuRURcor[:,1::3]
                    G[iR:imax,iR+2:jmax:6] += MuRURcor[:,2::3]
                    #----------------------------------------
                    # vR
                        # rT
                    MvRrT = CRv@XT
                    G[iR:imax,iL:iR] += MvRrT
                        # V+
                    CRvXYP = CRv@XYP
                    MvRVR = CRvXYP@AL
                    G[iR:imax,iR+3:jmax:6] += MvRVR[:, ::3]
                    G[iR:imax,iR+4:jmax:6] += MvRVR[:,1::3]
                    G[iR:imax,iR+5:jmax:6] += MvRVR[:,2::3]
                        # unx
                    MvRunx = w2r*CRvXYP@b0
                            # Ux
                    G[iR:imax,jmin+1:iL:2] += MvRunx@RunxUx
                            # rT
                    G[iR:imax,iL:iR] += MvRunx@RunxrT
                            # V+
                    MvRVRcor = MvRunx@RunxVR
                    G[iR:imax,iR+3:jmax:6] += MvRVRcor[:, ::3]
                    G[iR:imax,iR+4:jmax:6] += MvRVRcor[:,1::3]
                    G[iR:imax,iR+5:jmax:6] += MvRVRcor[:,2::3]
            else :
                if isinstance(cur_mat,Fluid) : # Solid/Fluid
                    CLu,CLv = CL[:,:3],CL[:,-3:]
                    CRpsi,CRu0x = CR[:,:1],CR[:,-1:]
                    AI,BI,CzxI,XI,YI,GI = LIM
                    bn = BI[:,-1:]
                    rk,_ = GI.shape # rk = iR-iL
                    XT,XP = XI[:,:rk],XI[:,rk:]
                    YT,YP = YI[:rk,:],YI[rk:,:]
                    w2r = rhoR*W**2                   
                    # Matrices
                    XYP = XP@YP
                    AL = AI@fds.L1down(3)
                    XTLbda = XT@np.linalg.inv(GI)
                    # Coefficients
                    coef_psi = 1.0/(-d00+hR*w2r*BI[-1,-1])
                    coef_u0x = 1.0/(-d00+hR*w2r*(XYP@bn)[-1,-1])
                    # Row-matrix
                    L01 = fds.L1up(1)
                    #----------------------------------------
                    # psi0+
                        # Psi+
                    RpsiPsi = coef_psi*L01
                    MpsiPsi = CRpsi@RpsiPsi
                    diR,djR = MpsiPsi.shape
                    imax,jmax = iR+diR, iR+2*djR
                    G[iR:imax,iR:jmax:2] += MpsiPsi
                        # rT
                    RpsirT = -hR*coef_psi*XTLbda[-1:,:]
                    MpsirT = CRpsi@RpsirT
                    G[iR:imax,iL:iR] += MpsirT
                        # U-
                    RpsiUL = -hR*coef_psi*AL[-1:,:]
                    MpsiUL = CRpsi@RpsiUL
                    _,djL = AL.shape
                    jmin = iL - 2*djL
                    G[iR:imax,jmin  :iL:6] += MpsiUL[:, ::3]
                    G[iR:imax,jmin+1:iL:6] += MpsiUL[:,1::3]
                    G[iR:imax,jmin+2:iL:6] += MpsiUL[:,2::3]   
                    #----------------------------------------
                    # u0x+
                        # Ux+
                    Ru0xUx = coef_u0x*L01
                    Mu0xUx = CRu0x@Ru0xUx
                    G[iR:imax,iR+1:jmax:2] += Mu0xUx
                        # rT
                    Ru0xrT = -hR*coef_u0x*XT[-1:,:]
                    Mu0xrT = CRu0x@Ru0xrT
                    G[iR:imax,iL:iR] += Mu0xrT
                        # V-
                    Ru0xVL = -hR*coef_u0x*(XYP@AL)[-1:,:]
                    Mu0xVL = CRu0x@Ru0xVL
                    G[iR:imax,jmin+3:iL:6] += Mu0xVL[:, ::3]
                    G[iR:imax,jmin+4:iL:6] += Mu0xVL[:,1::3]
                    G[iR:imax,jmin+5:iL:6] += Mu0xVL[:,2::3]
                    #----------------------------------------
                    # -i k rT
                        # rT
                    G[iL:iR,iL:iR] = GI
                    mGYT = -GI@YT
                        # V-
                    MV = mGYT@AL
                    G[iL:iR,jmin+3:iL:6] += MV[:, ::3]
                    G[iL:iR,jmin+4:iL:6] += MV[:,1::3]
                    G[iL:iR,jmin+5:iL:6] += MV[:,2::3]
                        # u0x+  
                    Mu0x = w2r*mGYT@bn
                            # Ux+
                    MUx = Mu0x@Ru0xUx
                    _,djR = MUx.shape
                    jmax = iR + 2*djR
                    G[iL:iR,iR+1:jmax:2] += MUx
                            # rT
                    MrT = Mu0x@Ru0xrT
                    G[iL:iR,iL:iR] += MrT
                            # V-
                    MVcor = Mu0x@Ru0xVL
                    G[iL:iR,jmin+3:iL:6] += MVcor[:, ::3]
                    G[iL:iR,jmin+4:iL:6] += MVcor[:,1::3]
                    G[iL:iR,jmin+5:iL:6] += MVcor[:,2::3]  
                    #----------------------------------------
                    # uL
                    diL,_ = CLu.shape
                    imin = iL-diL
                        # rT
                    MuLrT = CLu@XTLbda
                    G[imin:iL,iL:iR] += MuLrT
                        # U-
                    MuLUL = CLu@AL
                    G[imin:iL,jmin  :iL:6] += MuLUL[:, ::3]
                    G[imin:iL,jmin+1:iL:6] += MuLUL[:,1::3]
                    G[imin:iL,jmin+2:iL:6] += MuLUL[:,2::3]
                        # psi0
                    MuLpsi = w2r*CLu@bn
                            # PsiR
                    G[imin:iL,iR:jmax:2] += MuLpsi@RpsiPsi
                            # rT
                    G[imin:iL,iL:iR] += MuLpsi@RpsirT
                            # U-
                    MuLULcor = MuLpsi@RpsiUL
                    G[imin:iL,jmin  :iL:6] += MuLULcor[:, ::3]
                    G[imin:iL,jmin+1:iL:6] += MuLULcor[:,1::3]
                    G[imin:iL,jmin+2:iL:6] += MuLULcor[:,2::3]
                    #----------------------------------------
                    # vL
                        # rT
                    MvLrT = CLv@XT
                    G[imin:iL,iL:iR] += MvLrT
                        # V-
                    CLvXYP = CLv@XYP
                    MvLVL = CLvXYP@AL
                    G[imin:iL,jmin+3:iL:6] += MvLVL[:, ::3]
                    G[imin:iL,jmin+4:iL:6] += MvLVL[:,1::3]
                    G[imin:iL,jmin+5:iL:6] += MvLVL[:,2::3]
                        # u0x
                    MvLu0x = w2r*CLvXYP@bn
                            # Ux
                    G[imin:iL,iR+1:jmax:2] += MvLu0x@Ru0xUx
                            # rT
                    G[imin:iL,iL:iR] += MvLu0x@Ru0xrT
                            # V-
                    MvLVLcor = MvLu0x@Ru0xVL
                    G[imin:iL,jmin+3:iL:6] += MvLVLcor[:, ::3]
                    G[imin:iL,jmin+4:iL:6] += MvLVLcor[:,1::3]
                    G[imin:iL,jmin+5:iL:6] += MvLVLcor[:,2::3]
                else : # Solid/Solid
                    ALI,ARI,BI,XI,YI,GI = LIM
                    if GI is None : # No interfacial state vector iR=iL
                        AL6 = block_diag(ALI,ALI)
                        AR6 = block_diag(ARI,ARI)
                        mALn,AL0 = -AL6@fds.L1down(6),AR6@fds.L1up(6)
                        MLL,MLR = CL@mALn,CL@AL0
                        MRL,MRR = CR@mALn,CR@AL0
                        diL,djL = MLL.shape
                        imin,jmin = iL-diL, iL-djL
                        diR,djR = MRR.shape
                        imax,jmax = iR+diR, iR+djR
                        G[imin:iL,jmin:iL] += MLL
                        G[imin:iL,iR:jmax] += MLR
                        G[iR:imax,jmin:iL] += MRL
                        G[iR:imax,iR:jmax] += MRR
                    else :
                        CLu,CLv = CL[:,:3],CL[:,-3:]
                        CRu,CRv = CR[:,:3],CR[:,-3:]
                        mALn,AL0 = -ALI@fds.L1down(3),ARI@fds.L1up(3)
                        rk,_ = GI.shape # = iR - iL
                        XT,XP = XI[:,:rk],XI[:,rk:]
                        YT,YP = YI[:rk,:],YI[rk:,:]
                        XYP = XP@YP
                        mGYT = -GI@YT
                        XTLbda = XT@np.linalg.inv(GI)
                        #----------------------------------------
                        # -i k rT
                            # VL-
                        MrTVL = mGYT@mALn
                        _,djL = MrTVL.shape
                        jmin = iL - 2*djL
                        G[iL:iR,jmin+3:iL:6] += MrTVL[:, ::3]
                        G[iL:iR,jmin+4:iL:6] += MrTVL[:,1::3]
                        G[iL:iR,jmin+5:iL:6] += MrTVL[:,2::3]
                            # rT
                        G[iL:iR,iL:iR] += GI
                            # VR+
                        MrTVR = mGYT@AL0
                        _,djR = MrTVR.shape
                        jmax = iR + 2*djR
                        G[iL:iR,iR+3:jmax:6] += MrTVR[:, ::3]
                        G[iL:iR,iR+4:jmax:6] += MrTVR[:,1::3]
                        G[iL:iR,iR+5:jmax:6] += MrTVR[:,2::3] 
                        #----------------------------------------
                        # uss
                            # UL-
                        MULUL, MURUL = CLu@mALn,CRu@mALn
                        diL,_ = MULUL.shape
                        imin = iL - diL
                        G[imin:iL,jmin  :iL:6] += MULUL[:, ::3]
                        G[imin:iL,jmin+1:iL:6] += MULUL[:,1::3]
                        G[imin:iL,jmin+2:iL:6] += MULUL[:,2::3]
                        diR,_ = MURUL.shape
                        imax = iR + diR
                        G[iR:imax,jmin  :iL:6] += MURUL[:, ::3]
                        G[iR:imax,jmin+1:iL:6] += MURUL[:,1::3]
                        G[iR:imax,jmin+2:iL:6] += MURUL[:,2::3]
                            # rT
                        MULrT, MURrT = CLu@XTLbda,CRu@XTLbda
                        G[imin:iL,iL:iR] += MULrT
                        G[iR:imax,iL:iR] += MURrT
                            # UR+
                        MULUR, MURUR = CLu@AL0,CRu@AL0
                        G[imin:iL,iR  :jmax:6] += MULUR[:, ::3]
                        G[imin:iL,iR+1:jmax:6] += MULUR[:,1::3]
                        G[imin:iL,iR+2:jmax:6] += MULUR[:,2::3]
                        G[iR:imax,iR  :jmax:6] += MURUR[:, ::3]
                        G[iR:imax,iR+1:jmax:6] += MURUR[:,1::3]
                        G[iR:imax,iR+2:jmax:6] += MURUR[:,2::3]          
                        #----------------------------------------
                        # vss
                            # VL-
                        MVLVL, MVRVL = CLv@XYP@mALn,CRv@XYP@mALn
                        G[imin:iL,jmin+3:iL:6] += MVLVL[:, ::3]
                        G[imin:iL,jmin+4:iL:6] += MVLVL[:,1::3]
                        G[imin:iL,jmin+5:iL:6] += MVLVL[:,2::3]
                        G[iR:imax,jmin+3:iL:6] += MVRVL[:, ::3]
                        G[iR:imax,jmin+4:iL:6] += MVRVL[:,1::3]
                        G[iR:imax,jmin+5:iL:6] += MVRVL[:,2::3]
                            # rT
                        MVLrT, MVRrT = CLv@XT,CRv@XT
                        G[imin:iL,iL:iR] += MVLrT
                        G[iR:imax,iL:iR] += MVRrT                    
                            # VR+
                        MVLVR, MVRVR = CLv@XYP@AL0,CRv@XYP@AL0
                        G[imin:iL,iR+3:jmax:6] += MVLVR[:, ::3]
                        G[imin:iL,iR+4:jmax:6] += MVLVR[:,1::3]
                        G[imin:iL,iR+5:jmax:6] += MVLVR[:,2::3]
                        G[iR:imax,iR+3:jmax:6] += MVRVR[:, ::3]
                        G[iR:imax,iR+4:jmax:6] += MVRVR[:,1::3]
                        G[iR:imax,iR+5:jmax:6] += MVRVR[:,2::3]
                        #----------------------------------------
            # Next iteration
            prev_imin, prev_imax = cur_imin,cur_imax
            prev_layer, prev_mat = cur_layer, cur_mat
        # 2 - Global matrix for the multilayer plate including immersing
        #     fluids.
        if self.__level == 0 : # Vacuum/vacuum
            return G, mem_indexes, mem_pos
        L,R,N = self.L,self.R,Gsize
        if L > 0 : # Fluid at the left-hand side
            if R > 0 : # Fluid at the right-hand side
                if self.level == 2 : # Fluid 1 / Fluid 2
                    T = 4*N+8
                else :  # Fluid 1 / Fluid 1
                    T = 2*N+4
            else : # Fluid / Vacuum
                T = 2*N+2
        else : # Vacuum / Fluid 
            T = 2*N+2
        shape = (T,T)
        if SPARSE :
            H = sprs.lil_matrix(shape, dtype=np.complex)
        else :
            H = np.zeros(shape, dtype=np.complex)
        #-------
        H[:N,:N] = G.copy()       # [r0T] W [rnT] 
        H[N:2*N,N:2*N] = G.copy() # [i*kappa*r0T] i*kappa*W [i*kappa*rnT]
        if L > 0 : # Immersing fluid at the left-hand side
            ipsi0,iu0x = 2*N,2*N+1 # indexes of psi0 and u0x
            mem_indexes.extend([N,[ipsi0]]) ; mem_pos.append([0.0])
            lay_mat = self.__layers[0].material
            if isinstance(lay_mat,Fluid) : # First layer is fluid
                C0 = LC0[0]
                r0sr = self.__left_fluid.rho / lay_mat.rho
                ush = 1.0 / self.__layers[0].h
                cf = fds.d100 * r0sr * ush 
                # - i k psi0
                H[ipsi0,iu0x] += 1.0
                # - i k u0x
                    # psi0
                H[iu0x,ipsi0] += Nu**2 - (W/self.__left_fluid.c)**2 \
                                 - cf**2
                    # Psi
                mLsh = -fds.L1up(1) * ush
                _,dj = mLsh.shape
                jmax = 2*dj
                H[iu0x:iu0x+1,:jmax:2] += cf*mLsh
                    # i*kappa*Psi
                H[iu0x:iu0x+1,N:N+jmax:2] += mLsh
                # w0
                Mw0 = r0sr * C0
                imax,_ = Mw0.shape
                H[:imax,ipsi0:ipsi0+2] += Mw0
                # i*kappa*w0 # *** Correction 21-10-23 ***
                H[N:N+imax,ipsi0:ipsi0+2] += r0sr*cf*C0
                H[N:N+imax,:jmax] += r0sr*ush * C0@fds.L1up(2)
            else :  # First layer is solid
                C0u,C0v = LC0[0][:,:3],LC0[0][:,3:] # Coeff. of u0, v0
                nr,_ = C0u.shape
                i_min,i_max = idx_min[0],idx_min[0]+nr
                j_min,j_max = i_min,i_min+2*A0L3.shape[1]
                # rho0 omega^2 b0 (b0 is the last column of B0)
                rw2b0 = self.__left_fluid.rho*W**2*B0[:,2:]
                # Rows from 0 to N-1 : r0T and W
                # Corrective coefficient for u0: rw2b0 psi0
                MU = C0u@rw2b0
                H[i_min:i_max,ipsi0:ipsi0+1] += MU          
                # Corrective coefficient for v0: rw2b0 u0x
                MV = C0v@rw2b0
                H[i_min:i_max,iu0x:iu0x+1] +=  MV        
                # Corrective coefficient for -i k r0T: -G0 Y0T rw2b0 u0x
                H[:i_min,iu0x:iu0x+1] += -G0@Y0T@rw2b0
                # Rows from N to 2N-1 : i*kappa0*r0T and i*kappa0*W
                i_min2,i_max2 = i_min+N,i_max+N
                # Corrective coefficient for i*kappa0*u0: rw2b0 (n.u0)
                # n.u0 = (n.(X0T Lambda0 r0T)) + (n.(A0 L0 U))
                #        + (n.rw2b0) psi0
                c0uw2b0 = C0u@rw2b0
                nX0TLamb0 = X0TLamb0[-1:,:]
                nA0L3 = A0L3[-1:,:]
                nrw2b0 = rw2b0[-1:,:] # 1x1 array
                    # r0T
                H[i_min2:i_max2,:i_min] += c0uw2b0@nX0TLamb0
                    # U
                HU = c0uw2b0@nA0L3
                H[i_min2:i_max2,j_min  :j_max:6] += HU[:, ::3]
                H[i_min2:i_max2,j_min+1:j_max:6] += HU[:,1::3]
                H[i_min2:i_max2,j_min+2:j_max:6] += HU[:,2::3]
                    # psi0
                H[i_min2:i_max2,ipsi0:ipsi0+1] += c0uw2b0@nrw2b0          
                # Corrective coefficient for i*kappa0*v0: rw2b0 (n.v0)
                # n.v0 = (n.(X0T r0T)) + (n.(X0P Y0P A0 L0 V))
                #        + (n.X0P Y0P rw2b0)u0x
                c0vw2b0 = C0v@rw2b0
                nX0T = X0T[-1:,:]
                nX0PY0P = X0P[-1:,:]@Y0P
                nXYAL = nX0PY0P@A0L3
                nXYrw2b0 = nX0PY0P@rw2b0
                    # r0T
                H[i_min2:i_max2,:i_min] += c0vw2b0@nX0T
                    # V
                HV = c0vw2b0@nXYAL
                H[i_min2:i_max2,j_min+3:j_max:6] += HV[:, ::3]
                H[i_min2:i_max2,j_min+4:j_max:6] += HV[:,1::3]
                H[i_min2:i_max2,j_min+5:j_max:6] += HV[:,2::3]
                    # u0x
                H[i_min2:i_max2,iu0x:iu0x+1] += c0vw2b0@nXYrw2b0  
                # Corrective coefficient for -i k (i*kappa0*r0T):
                #                       -G0 Y0T rw2b0 (n.v0)
                mG0Y0Trw2b0 = -G0@Y0T@rw2b0
                    # r0T
                H[N:i_min2,:i_min] += mG0Y0Trw2b0@nX0T
                    # V
                GV = mG0Y0Trw2b0@nXYAL
                H[N:i_min2,j_min+3:j_max:6] += GV[:, ::3]
                H[N:i_min2,j_min+4:j_max:6] += GV[:,1::3]
                H[N:i_min2,j_min+5:j_max:6] += GV[:,2::3]
                    # u0x
                H[N:i_min2,iu0x:iu0x+1] += mG0Y0Trw2b0@nXYrw2b0 
                # Coefficient of -i k psi0
                H[ipsi0,iu0x] += 1
                # Coefficients of -i k u0x
                    # psi0: nu^2 - w^2/c0^2
                H[iu0x,ipsi0] += Nu**2-W**2/self.__left_fluid.c**2
                    # i*kappa0*r0T: -n.(X0T Lambda0)
                H[iu0x:iu0x+1,N:i_min2] += -nX0TLamb0
                    # i*kappa0*U: -n.(A0 L0)
                GU = -nA0L3
                j_min2,j_max2 = j_min+N,j_max+N
                H[iu0x:iu0x+1,j_min2  :j_max2:6] += GU[:, ::3]
                H[iu0x:iu0x+1,j_min2+1:j_max2:6] += GU[:,1::3]
                H[iu0x:iu0x+1,j_min2+2:j_max2:6] += GU[:,2::3]
                    # -(n.rw2b0)*(n.u0)
                mnrw2b0 = -rw2b0[-1,0] # scalar
                        # r0T
                H[iu0x:iu0x+1,:i_min] += mnrw2b0*nX0TLamb0
                        # U
                KU = mnrw2b0*nA0L3
                H[iu0x:iu0x+1,j_min  :j_max:6] += KU[:, ::3]
                H[iu0x:iu0x+1,j_min+1:j_max:6] += KU[:,1::3]
                H[iu0x:iu0x+1,j_min+2:j_max:6] += KU[:,2::3]
                        # psi0
                H[iu0x:iu0x+1,ipsi0:ipsi0+1] += mnrw2b0*nrw2b0        
        if R > 0 : # Immersing fluid at the right-hand side
            if self.__level == 2 : # 2 Fluids with different sound speeds
                Nstar = 2*N + 2
                H[Nstar:2*Nstar,Nstar:2*Nstar] = H[:Nstar,:Nstar].copy()
                ipsie,iuex = 2*Nstar,2*Nstar+1 # indexes of psie and uex
            else :
                Nstar = N
                if L > 0 : # 2 Fluids with the same sound speed
                    ipsie,iuex = 2*N+2,2*N+3 # indexes of psie and uex
                else : # Vacuum at the left-hand side
                    ipsie,iuex = 2*N,2*N+1
            lay_mat = self.__layers[-1].material
            mem_indexes.extend([Nstar,[ipsie]]) ; mem_pos.append([self.e])
            if isinstance(lay_mat,Fluid) : # Last layer is fluid
                Cn = LCn[-1]
                rnsr = self.__right_fluid.rho / lay_mat.rho
                ush = 1.0 / self.__layers[-1].h
                cf = -fds.d100 * rnsr * ush 
                # - i k psie
                H[ipsie,iuex] += 1.0
                # - i k uex
                    # psie
                H[iuex,ipsie] += Nu**2 - (W/self.__right_fluid.c)**2 - \
                                 cf**2
                    # Psi
                mLsh = -fds.L1down(1) * ush
                _,dj = mLsh.shape
                jmin = N - 2*dj
                H[iuex:iuex+1,jmin:N:2] += cf*mLsh
                    # -sign*i*kappa*Psi
                imax2 = N+Nstar
                jmin2,jmax2 = imax2-2*dj,imax2
                H[iuex:iuex+1,jmin2:jmax2:2] += -sign*mLsh
                # wn
                Mwn = rnsr * Cn
                di,_ = Mwn.shape
                imin,imin2 = N-di,imax2-di
                H[imin:N,ipsie:ipsie+2] += Mwn
                # -sign*i*kappa*wn *** Correction 21-10-23 ***
                H[imin2:imax2,ipsie:ipsie+2] += -sign*rnsr*cf*Cn
                H[imin2:imax2,jmin:N] += -sign*rnsr*ush * Cn@fds.L1down(2)
            else :  # First layer is solid
                Cnu,Cnv = LCn[-1][:,:3],LCn[-1][:,3:] # Coeff. of ue, ve
                nr,_ = Cnu.shape
                i_min,i_max = idx_max[-1]-nr,idx_max[-1]
                j_min,j_max = i_max-2*AnL3.shape[1],i_max
                # rhoe omega^2 bn (bn is the last column of Bn)
                rw2bn = self.__right_fluid.rho*W**2*Bn[:,2:]
                # Rows from i_max to N-1 : rnT
                # Corrective coefficient for un: rw2bn psie
                MU = Cnu@rw2bn
                H[i_min:i_max,ipsie:ipsie+1] += MU          
                # Corrective coefficient for vn: rw2bn uex
                MV = Cnv@rw2bn
                H[i_min:i_max,iuex:iuex+1] +=  MV        
                # Corrective coefficient for -i k rnT: -Gn YnT rw2bn uex
                H[i_max:N,iuex:iuex+1] += -Gn@YnT@rw2bn
                # Rows from N* to N*+N-1 : i*kappae*[r0T W rnT]
                i_min2,i_max2 = Nstar+i_min,Nstar+i_max
                # Corrective coefficient for i*kappae*ue:
                #                                   -sign*rw2bn (n.ue)
                # n.ue = (n.(XnT Lambdan rnT)) + (n.(An Ln U))
                #        + (n.rw2bn) psie
                cnuw2bn = -sign*Cnu@rw2bn
                nXnTLambn = XnTLambn[-1:,:]
                nAnL3 = AnL3[-1:,:]
                nrw2bn = rw2bn[-1:,:] # 1x1 array
                    # rnT
                H[i_min2:i_max2,i_max:N] += cnuw2bn@nXnTLambn
                    # U
                HU = cnuw2bn@nAnL3
                H[i_min2:i_max2,j_min  :j_max:6] += HU[:, ::3]
                H[i_min2:i_max2,j_min+1:j_max:6] += HU[:,1::3]
                H[i_min2:i_max2,j_min+2:j_max:6] += HU[:,2::3]
                    # psie
                H[i_min2:i_max2,ipsie:ipsie+1] += cnuw2bn@nrw2bn          
                # Corrective coefficient for i*kappae*vn:
                #                                   -sign*rw2bn (n.v0)
                # n.vn = (n.(XnT rnT)) + (n.(XnP YnP An Ln V))
                #        + (n.XnP YnP rw2bn)u0x
                cnvw2bn = -sign*Cnv@rw2bn
                nXnT = XnT[-1:,:]
                nXnPYnP = XnP[-1:,:]@YnP
                nXYAL = nXnPYnP@AnL3
                nXYrw2bn = nXnPYnP@rw2bn
                    # rnT
                H[i_min2:i_max2,i_max:N] += cnvw2bn@nXnT
                    # V
                HV = cnvw2bn@nXYAL
                H[i_min2:i_max2,j_min+3:j_max:6] += HV[:, ::3]
                H[i_min2:i_max2,j_min+4:j_max:6] += HV[:,1::3]
                H[i_min2:i_max2,j_min+5:j_max:6] += HV[:,2::3]
                    # uex
                H[i_min2:i_max2,iuex:iuex+1] += cnvw2bn@nXYrw2bn  
                # Corrective coefficient for -i k (i*kappae*rnT):
                #                       sign Gn YnT rw2bn (n.vn)
                mGnYnTrw2bn = sign*Gn@YnT@rw2bn
                    # rnT
                i_max3 = Nstar+N
                H[i_max2:i_max3,i_max:N] += mGnYnTrw2bn@nXnT
                    # V
                GV = mGnYnTrw2bn@nXYAL
                H[i_max2:i_max3,j_min+3:j_max:6] += GV[:, ::3]
                H[i_max2:i_max3,j_min+4:j_max:6] += GV[:,1::3]
                H[i_max2:i_max3,j_min+5:j_max:6] += GV[:,2::3]
                    # uex
                H[i_max2:i_max3,iuex:iuex+1] += mGnYnTrw2bn@nXYrw2bn 
                # Coefficient of -i k psie
                H[ipsie,iuex] += 1
                # Coefficients of -i k uex
                    # psie: nu^2 - w^2/ce^2
                H[iuex,ipsie] += Nu**2-W**2/self.__right_fluid.c**2
                    # i*kappae*rnT: sign * n.(XnT Lambdan)
                H[iuex:iuex+1,i_max2:i_max3] += sign*nXnTLambn
                    # i*kappae*U: sign * n.(An Ln)
                GU = sign*nAnL3
                j_min2,j_max2 = Nstar+j_min,Nstar+j_max
                H[iuex:iuex+1,j_min2  :j_max2:6] += GU[:, ::3]
                H[iuex:iuex+1,j_min2+1:j_max2:6] += GU[:,1::3]
                H[iuex:iuex+1,j_min2+2:j_max2:6] += GU[:,2::3]
                    # -(n.rw2bn)*(n.un)
                mnrw2bn = -rw2bn[-1,0] # scalar
                        # rnT
                H[iuex:iuex+1,i_max:N] += mnrw2bn*nXnTLambn
                        # U
                KU = mnrw2bn*nAnL3
                H[iuex:iuex+1,j_min  :j_max:6] += KU[:, ::3]
                H[iuex:iuex+1,j_min+1:j_max:6] += KU[:,1::3]
                H[iuex:iuex+1,j_min+2:j_max:6] += KU[:,2::3]
                        # psie
                H[iuex:iuex+1,ipsie:ipsie+1] += mnrw2bn*nrw2bn
            #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            if self.__level == 2 : # 2 fluids with different sound speeds
                H[-2:,-2:] = H[-4:-2,-4:-2].copy()
                H[N:2*N,-2:] = H[:N,-4:-2].copy() 
                H[Nstar+N:Nstar+2*N,-2:] = H[Nstar:Nstar+N,-4:-2].copy()
                H[-2:,N:2*N] = H[-4:-2,:N].copy() 
                H[-2:,Nstar+N:Nstar+2*N] = H[-4:-2,Nstar:Nstar+N].copy()
                H[N:2*N,Nstar+N:Nstar+2*N] = H[:N,Nstar:Nstar+N].copy()
                H[Nstar+N:Nstar+2*N,N:2*N] = H[Nstar:Nstar+N,:N].copy()
        #-------
        return H, mem_indexes, mem_pos
    #--------------------------------------------------
    def first_wavenumber_values(self, F, Nu=0.0, nb=20, \
                                check_opposite=True, \
                                with_mode_shapes=False) :
        """F and Nu can be scalars or vectors.
           nb is the number of wanted wavenumber values"""
        is_array = True
        if len(np.shape(F)) > 0 :
            if len(np.shape(Nu)) == 0 : # Vector F and scalar Nu
                Nu = Nu*np.ones_like(F)
            else :            # Vector F and vector Nu
                pass 
        elif len(np.shape(Nu)) > 0 :    # Scalar F and vector Nu
            F = F*np.ones_like(Nu)
        else :                # Scalar F and scalar Nu
            is_array = False
        if is_array :
            f,nu = F[0],Nu[0]
            G0,LI0,LP0 = self.global_matrix_fixed_frequency(f,nu)
            if GeneralDiscretizedLayer.SPARSE : G0 = G0.toarray()
            shp = G0.shape
            LG = [G0]
            for f,nu in zip(F[1:],Nu[1:]) :
                G,LI,_ = self.global_matrix_fixed_frequency(f,nu)
                if GeneralDiscretizedLayer.SPARSE : G = G.toarray()
                err = False
                if G.shape != shp :
                    err = True
                else :
                    for VI0,VI in zip(LI0,LI) :
                        if not np.allclose(VI0,VI) :
                            err = True
                            break
                if err :
                    print("GeneralDiscretizedLayer." + \
                          "first_wavenumber_values:\n\t" + \
                          "Vectorization not possible")
                    return [ self.first_wavenumber_values(f, nu, nb=np, \
                                check_opposite=check_opposite, \
                                with_mode_shapes=with_mode_shapes) \
                             for f,nu in zip(F,Nu) ]
                LG.append(G)
            G,LP = np.array(LG),LP0
        else :
            G,LI,LP = self.global_matrix_fixed_frequency(F,Nu)
            if GeneralDiscretizedLayer.SPARSE : G = G.toarray()
        if with_mode_shapes :
            miK, T_shapes = np.linalg.eig(G)
        else :
            miK, T_shapes = np.linalg.eigvals(G),None
        T_k = 1.0j*miK
        wavenumbers, mode_shapes, emax = \
                        self.take_first(T_k, nb, dtype="Wavenumber", \
                                        check_opposite=check_opposite, \
                                        array=T_shapes)
        if check_opposite :
            if emax > 1e-8*np.abs(T_k).mean() :
                msg = "DiscretizedMultilayerPlate." + \
                      "first_frequency_values :\n\t" + \
                      "opposites are not all present"
                print(msg)
        if with_mode_shapes : return wavenumbers, mode_shapes
        else : return wavenumbers # eigenwavenumbers only
    #--------------------------------------------------
    # Fixed slowness
    #--------------------------------------------------
    def global_matrix_fixed_slowness(self, S ,signs=[1,1], verbose=False):
        if verbose :
            prt = print
        else :
            prt = lambda *args : None
        fds = self.fds
        # 1 - Determining the size of the global matrix        
        # First layer
        cur_layer = self.__layers[0]
        matrices = cur_layer.matrices_fixed_slowness(S)
        if matrices is None : # Degenerate case
            prt("Degenerate case for the first layer")
            return None,None,None
        M,C0,Cn = matrices
        LM,LC0,LCn = [M],[C0],[None,Cn]
        i0,N,cur_mat = 0,cur_layer.N,cur_layer.material
        i1 = i0 + N
        mem_pos = [cur_layer.Z[1:-1]]
        cur_pos = cur_layer.Z[-1]
        #+++++++ Left-hand side and first layer
        if isinstance(cur_mat,Fluid) : # Fluid layer
            if self.left_fluid is None :
            # Vacuum for fluid layer
                interfaces = ["VF"]
            elif self.left_fluid == "Wall":
            # Wall for fluid layer
                interfaces = ["WF"]
            else : # External fluid at z=0
                if abs(S*self.left_fluid.c-1) < 1e-8 : # Degenerate case
                    prt("Degenerate case for the left interface")
                    return None,None,None                    
                interfaces = ["LF"]
                i0 += 1 ; i1 += 1
            mem_indexes = [np.arange(i0,i1,2)]
        else : # Solid layer
            matrices = cur_layer.FS_matrices_0( S, self.left_fluid, \
                                                signs[0])
            if matrices is None : # Degenerate case
                prt("Degenerate case for the left interface")
                return None,None,None
            Aur0, Auu0, Avr0, Avv0, Brr0, Brv0 = matrices
            interfaces = ["XS"]
            rank0 = Brr0.shape[0]
            i0 += rank0 ; i1 += rank0
            mem_indexes = [np.arange(i0,i1,6)]
        indexes = [0,i0,i1]
        prev_layer,prev_mat = cur_layer,cur_mat 
        #+++++++ Other layers
        interface_matrices = [None]
        for no_lay,cur_layer in enumerate(self.__layers[1:],2) :
            N,i0,cur_mat = cur_layer.N,indexes[-1],cur_layer.material
            i1 = i0 + N
            mem_pos.append(cur_layer.Z[1:-1]+cur_pos)
            cur_pos += cur_layer.Z[-1]
            matrices = cur_layer.matrices_fixed_slowness(S)
            if matrices is None : # Degenerate case
                prt(f"Degenerate case for layer #{no_lay}")
                return None,None,None
            M,C0,Cn = matrices
            LM.append(M) ; LC0.append(C0) ; LCn.append(Cn)
            if isinstance(cur_mat,Fluid) :
                if isinstance(prev_mat,Fluid) :
                    # Fluid/Fluid
                    interfaces.append("FF")
                    interface_matrices.append([])
                    # Unchanged i0 and i1
                else :
                    # Solid/Fluid
                    im = prev_layer.matrices_fluid_at_right(cur_layer)                                       
                    if im[0] is None : # no interfacial state vector for
                                       #  solid layer
                        rank = 2
                    else :
                        rank = 2 + im[0].shape[0]
                    interfaces.append("SF")
                    interface_matrices.append(im)
                    i0 += rank ; i1 += rank
                mem_indexes.append(np.arange(i0,i1,2))
            else :
                if isinstance(prev_mat,Fluid) :
                    # Fluid/Solid
                    im = cur_layer.matrices_fluid_at_left(prev_layer)                    
                    if im[0] is None : # no interfacial state vector for
                                       #  solid layer
                        rank = 2
                    else :
                        rank = 2 + im[0].shape[0]
                    interfaces.append("FS")
                    interface_matrices.append(im)
                    i0 += rank ; i1 += rank
                else :
                    # Solid/Solid
                    im = cur_layer.matrices_solid_at_left(prev_layer)
                    if im[0] is None : # no interfacial state vector
                        rank = 0
                    else :
                        rank = im[0].shape[0]
                    interfaces.append("SS")
                    interface_matrices.append(im)
                    i0 += rank ; i1 += rank
                mem_indexes.append(np.arange(i0,i1,2))                    
            indexes.extend( [i0,i1] )
            prev_layer,prev_mat = cur_layer,cur_mat
        #+++++++ Right-hand side and last layer  
        if isinstance(cur_mat,Fluid) : # Fluid layer
            if self.right_fluid is None :
            # Vacuum for fluid layer
                interfaces.append("FV")
                i1 = 0
            elif self.right_fluid == "Wall" :
            # Wall for fluid layer
                interfaces.append("FW")
                i1 = 0
            else : # External fluid at z=l
                if abs(S*self.right_fluid.c-1) < 1e-8 : # Degenerate case
                    prt("Degenerate case for the right interface")
                    return None,None,None 
                interfaces.append("FR")
                i1 = 1
        else : # Solid layer
            matrices = cur_layer.FS_matrices_n( S, self.right_fluid, \
                                                signs[1])
            if matrices is None : # Degenerate case
                prt("Degenerate case for the right interface")
                return None,None,None
            Aurn, Auun, Avrn, Avvn, Brrn, Brvn = matrices
            interfaces.append("SX")
            rankn = Brrn.shape[0]
            i1 = rankn
        indexes.append( indexes[-1]+i1 )
        # prt(indexes,interfaces) 
        # 2 - Building the global matrix
        shape = (indexes[-1],indexes[-1])
        if GeneralDiscretizedLayer.SPARSE :
            G = sprs.lil_matrix(shape, dtype=np.complex)
        else :
            G = np.zeros(shape, dtype=np.complex)
        for no,(i0,i1,i2,M,C0,Cn,itf,im) in enumerate( \
            zip(indexes[:-2:2], indexes[1:-1:2], indexes[2::2], \
                LM, LC0, LCn, interfaces,interface_matrices) ) :
            G[i1:i2,i1:i2] = M
        # +++ Left interface +++
            if itf == "XS" :   # Left/Solid
                # i w r0T
                    # Brr0
                G[i0:i1,i0:i1] += Brr0
                    # Brv0
                _,nc = Brv0.shape
                jmax = i1+2*nc
                G[i0:i1,i1+3:jmax:6] += Brv0[:, ::3]
                G[i0:i1,i1+4:jmax:6] += Brv0[:,1::3]
                G[i0:i1,i1+5:jmax:6] += Brv0[:,2::3]
                # C0[:,:3]@Aur0@r0T
                nr,_ = C0.shape
                imax = i1+nr
                G[i1:imax,i0:i1] += C0[:,:3]@Aur0
                # C0[:,-3:]@Avr0@r0T
                G[i1:imax,i0:i1] += C0[:,-3:]@Avr0                
                # C0[:,:3]@Auu0@U
                MU = C0[:,:3]@Auu0
                G[i1:imax,i1  :jmax:6] += MU[:, ::3]
                G[i1:imax,i1+1:jmax:6] += MU[:,1::3]
                G[i1:imax,i1+2:jmax:6] += MU[:,2::3]                
                # C0[:,-3:]@Avv0@V
                MV = C0[:,-3:]@Avv0
                G[i1:imax,i1+3:jmax:6] += MV[:, ::3]
                G[i1:imax,i1+4:jmax:6] += MV[:,1::3]
                G[i1:imax,i1+5:jmax:6] += MV[:,2::3]
            elif itf == "LF" : # Left fluid/Fluid
                ######################
                ### À MODIFIER !!! ###
                ######################
                c0,rho0 = self.left_fluid.c,self.left_fluid.rho
                h = self.layers[0].h
                t0 = signs[0]*np.sqrt(1.0/c0**2-S**2)
                di,_ = C0.shape
                # phi0
                G[1:1+di,:1] += C0[:,:1]
                # p0 = a00*phi0 + Aphi*Phi
                cpv = -rho0/(t0*h)
                a00,Aphi = cpv*self.fds.d100, cpv*self.fds.L1up(1)
                    # phi0
                G[1:1+di,:1] += a00*C0[:,-1:]
                    # Phi
                Aphi = C0[:,-1:]@Aphi
                _,dj = Aphi.shape
                G[1:1+di,1:1+2*dj:2] += Aphi
                # i*w*phi0
                cpv = -cpv/self.layers[0].material.rho
                    # phi0
                G[0,0] = cpv*self.fds.d100
                    # Phi
                G[:1,1:1+2*dj:2] = cpv*self.fds.L1up(1)
            elif itf == "VF" : # Vacuum/Fluid
                pass # w0 = 0, nothing to do
            elif itf == "WF" : # Wall/Fluid
                Mw0W = (-1.0/self.fds.d100)*C0@self.fds.L1up(2)
                imax,jmax = Mw0W.shape
                G[:imax,:jmax] += Mw0W
        # +++ Internal interface +++
            elif itf == "SS" : # Solid/Solid
                pass ## TO DO ##
            elif itf == "FF" : # Fluid/Fluid
                pass ## TO DO ##
            elif itf == "SF" : # Solid/Fluid
                pass ## TO DO ##
            elif itf == "FS" : # Fluid/Solid
                pass ## TO DO ##
            else :
                print(f"Unknown interface '{itf}'")
        # (end of loop in layers)
        # +++ Right interface +++
        Cn = LCn[-1]
        itf = interfaces[-1]
        i0,i1 = indexes[-2:]
        if itf == "SX" :   # Solid/Right
            # i w rnT
                # Brrn
            G[i0:i1,i0:i1] += Brrn
                # Brvn
            _,nc = Brvn.shape
            jmin = i0-2*nc
            G[i0:i1,jmin+3:i0:6] += Brvn[:, ::3]
            G[i0:i1,jmin+4:i0:6] += Brvn[:,1::3]
            G[i0:i1,jmin+5:i0:6] += Brvn[:,2::3]
            # Cn[:,:3]@Aurn@rnT
            nr,_ = Cn.shape
            imin = i0-nr
            G[imin:i0,i0:i1] += Cn[:,:3]@Aurn
            # Cn[:,-3:]@Avrn@rnT
            G[imin:i0,i0:i1] += Cn[:,-3:]@Avrn                
            # Cn[:,:3]@Auun@U
            MU = Cn[:,:3]@Auun
            G[imin:i0,jmin  :i0:6] += MU[:, ::3]
            G[imin:i0,jmin+1:i0:6] += MU[:,1::3]
            G[imin:i0,jmin+2:i0:6] += MU[:,2::3]                
            # Cn[:,-3:]@Avvn@V
            MV = Cn[:,-3:]@Avvn
            G[imin:i0,jmin+3:i0:6] += MV[:, ::3]
            G[imin:i0,jmin+4:i0:6] += MV[:,1::3]
            G[imin:i0,jmin+5:i0:6] += MV[:,2::3]
        elif itf == "FR" : # Fluid/Right fluid
                ######################
                ### À MODIFIER !!! ###
                ######################
            ce,rhoe = self.right_fluid.c,self.right_fluid.rho
            h = self.layers[-1].h
            te = signs[-1]*np.sqrt(1.0/ce**2-S**2)
            di,_ = Cn.shape
            # phie
            G[-1-di:-1,-1:] += Cn[:,:1]
            # pn = aee*phie + Aphi*Phi
            cpv = rhoe/(te*h)
            aee,Aphi = -cpv*self.fds.d100, cpv*self.fds.L1down(1)
                # phie
            G[-1-di:-1,-1:] += aee*Cn[:,-1:]
                # Phi
            Aphi = Cn[:,-1:]@Aphi
            _,dj = Aphi.shape
            G[-1-di:-1,-1-2*dj:-1:2] += Aphi
            # i*w*phie
            cpv = -cpv/self.layers[-1].material.rho
                # phie
            G[-1,-1] = -cpv*self.fds.d100
                # Phi
            G[-1:,-1-2*dj:-1:2] = cpv*self.fds.L1down(1)
        elif itf == "FV" : # Fluid/Vacuum
            pass # wn = 0, nothing to do
        elif itf == "FW" : # Fluid/Wall
            MwnW = (1.0/self.fds.d100)*Cn@self.fds.L1down(2)
            di,dj = MwnW.shape
            G[-di:,-dj:] += MwnW
        else :
            print(f"Unknown last interface '{itf}'") 
        #--------
        return G, mem_indexes, mem_pos    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def partial_waves(fluid, K, W, Nu=0.0) :
        """Returns the vertical wavenumbers ±sqrt(W²/c²-K²-Nu²)."""
        K,W,Nu = GeneralDiscretizedLayer.homogenized_parameters(K,W,Nu)
        Kz = -1.0j * np.sqrt( (1.0+0.0j) * \
                              (K**2+Nu**2-(W/fluid.c)**2) )
        Kappa = np.empty( np.shape(K)+(2,), dtype = np.complex128)
        Kappa[...,0],Kappa[...,1] = Kz,-Kz
        return Kappa      
    #--------------------------------------------------  
    def shape_function(self, field, k, f, mode, nu=0.0, dz=0.0) :
        if field not in ("Ux", "Uy", "Uz", "Sxx", "Sxy", "Sxz", "Syy", \
                         "Syz","Szz", "dUx/dz", "dUy/dz", "dUz/dz", \
                         "dSxz/dz", "dSyz/dz", "dSzz/dz" ) :
            print(f"Unknown field '{field}'")
            return np.zeros_like
        if self.left_fluid is None or self.left_fluid == "Wall" :
            shape_L = np.zeros_like
        else :
            k0,c0 = mode[-2]
            shape_L = self.side_shape_function(field, k, f, \
                                                k0, c0, "L", nu, dz)
        if self.right_fluid is None or self.right_fluid == "Wall"  :
            shape_R = np.zeros_like
        else :
            ke,ce = mode[-1]
            shape_R = self.side_shape_function(field, k, f, \
                                               ke, ce, "R", nu, dz)
        z0,L_shape = dz,[]
        for lay,(Kz,C,_) in zip(self.layers,mode) :
            L_shape.append( \
                lay.shape_function(field, k, f, Kz, C, nu, z0) )
            z0 += lay.e
        def fonc_shape(Vz, LS=L_shape, sL=shape_L, sR=shape_R) :
            shp = sL(Vz).astype(np.complex128) + sR(Vz)
            for fonc in LS :
                shp += fonc(Vz)
            return shp
        return fonc_shape     
    #--------------------------------------------------  
    def side_shape_function(self, field, k, f, Kz, C, side, \
                            nu=0.0, dz=0.0) :
        if field not in ("Ux", "Uy", "Uz", "Sxx", "Sxy", "Sxz", "Syy", \
                         "Syz", "Szz", "Phi", "Psi", "P", \
                         "dUx/dz", "dUy/dz", "dUz/dz", \
                         "dSxz/dz", "dSyz/dz", "dSzz/dz") :
            print(f"Unknown field '{field}'")
            return np.zeros_like
        if side == "L" :
            zL,zR = -np.inf, dz
            z0 = zR
            def clip(Vz,zr=z0) :
                return np.clip(Vz,-np.inf,zr)
            Kz = -Kz # stored Kz oriented with the decreasing z
            fluid = self.left_fluid
        elif side == "R" :
            zL,zR = dz+self.e, np.inf
            z0 = zL
            def clip(Vz,zl=z0) :
                return np.clip(Vz,zl,np.inf)
            fluid = self.right_fluid
        else :
            print(f"Unknown side '{side}'")
            return np.zeros_like
        #+++++++++    
        if field in ("Sxy", "Sxz", "Syz", "dSxz/dz", "dSyz/dz") :
            return np.zeros_like
        def fonc_phi(Vz, kz=Kz, c=C, z0=z0, zL=zL, zR=zR, clip=clip) :
            Z = clip(Vz)
            Vphi = c*np.exp(-1.0j*kz*(Z-z0)) 
            return ((Vz>zL)&(Vz<=zR))*Vphi
        if field == "Phi" :
            return fonc_phi
        w = 2.0*np.pi*f # non-zero
        iw = 1j*w
        if field == "Psi" :
            def fonc_psi(Vz, iw=iw, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return Vphi/iw
            return fonc_psi
        if field == "Ux" :
            def fonc_Ux(Vz, ksw=k/w, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -ksw*Vphi
            return fonc_Ux
        if field == "dUx/dz" :
            def fonc_dUx(Vz, kz=Kz, ksw=k/w, f_phi=fonc_phi):
                idVphi = kz*f_phi(Vz)
                return 1.0j*ksw*idVphi
            return fonc_dUx
        if field == "Uy" :
            def fonc_Uy(Vz, nsw=nu/w, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -nsw*Vphi
            return fonc_Uy
        if field == "dUy/dz" :
            def fonc_dUy(Vz, kz=Kz, nsw=nu/w, f_phi=fonc_phi):
                idVphi = kz*f_phi(Vz)
                return 1.0j*nsw*idVphi
            return fonc_dUy
        if field == "Uz" :
            def fonc_Uz(Vz, w=w, kz=Kz, c=C, z0=z0, zL=zL, zR=zR, \
                        clip=clip):
                Z = clip(Vz)
                Uz = - kz*c*np.exp(-1.0j*kz*(Z-z0)) / w
                return ((Vz>zL)&(Vz<=zR))*Uz
            return fonc_Uz
        if field == "dUz/dz" :
            def fonc_Uz(Vz, w=w, kz=Kz, c=C, z0=z0, zL=zL, zR=zR, \
                        clip=clip):
                Z = clip(Vz)
                dUz = 1.0j*kz**2*c*np.exp(-1.0j*kz*(Z-z0)) / w
                return ((Vz>zL)&(Vz<=zR))*dUz
            return fonc_Uz
        if field == "P" :
            rho = fluid.rho
            def fonc_P(Vz, iwrho=iw*rho, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return -iwrho*Vphi
            return fonc_P
        if field in ("Sxx","Syy","Szz") :
            rho = fluid.rho
            def fonc_SL(Vz, iwrho=iw*rho, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return iwrho*Vphi
            return fonc_SL
        if field == "dSzz/dz" :
            rho = fluid.rho
            def fonc_dSL(Vz, w_rho_kz=w*rho*Kz, f_phi=fonc_phi):
                Vphi = f_phi(Vz)
                return w_rho_kz*Vphi
            return fonc_dSL            
        print(f"Program error: unexpected field '{field}'")
        return np.zeros_like 
    #--------------------------------------------------    
    def modes_for_given_wavenumber(self, K, Nu=0.0, sign=1, \
                                   rel_err=1e-4, rel_kappa_err=1e-2, \
                                   normalized=True) :
        """WARNING: New Version since 2021, December 14."""
        if np.ndim(K) == 0 : # K is a single value
            big_G,LI,LP = self.global_matrix_fixed_wavenumber(K,Nu,sign)
        elif np.ndim(K) == 1 : # K is vector
            G,LI,LP = self.global_matrix_fixed_wavenumber(K[0],Nu,sign)
            big_G = np.empty( np.shape(K)+G.shape, dtype=np.complex128 )
            big_G[0,...] = G
            for i,k in enumerate(K[1:],1) :
                G,_,_ = self.global_matrix_fixed_wavenumber(k,Nu,sign)
                big_G[i,...] = G
        else :
            print("K cannot be an array with more than one index")
            return
        FDLay = FluidDiscretizedLayer
        iW, VP = np.linalg.eig(big_G)
        W = -1.0j*iW ; del iW
        VP = np.swapaxes( VP, -1, -2)
        if np.ndim(K) == 0 :
            indexes = np.where( W.round(0).real > 0 )[0]
            W,VP = W[indexes], VP[indexes]
            indexes = W.argsort()
            W,VP = W[indexes], VP[indexes]
            Kx = K*np.ones(len(indexes))
        else :
            k,w,vp = K[0],W[0],VP[0]
            indexes = np.where( w.round(0).real > 0 )[0]
            nb_idx = len(indexes)
            w,vp = w[indexes], vp[indexes]
            indexes = w.argsort()
            Tab_W = w[indexes]
            Tab_VP = vp[indexes]
            Tab_K = k*np.ones(nb_idx)
            for k,w,vp in zip(K[1:],W[1:],VP[1:]) :
                indexes = np.where( w.round(0).real > 0 )[0]
                nb_idx = len(indexes)
                w,vp = w[indexes], vp[indexes]
                indexes = w.argsort()
                Tab_W = np.concatenate([Tab_W, w[indexes]])
                Tab_VP = np.concatenate([Tab_VP, vp[indexes]], axis=0)
                Tab_K = np.concatenate([Tab_K, k*np.ones(nb_idx)])
            Kx,W,VP = Tab_K, Tab_W, Tab_VP
        # Kx and W are vectors, VP is a matrix
        print("Kx.shape:", Kx.shape, "; W.shape:", W.shape, \
              "; VP.shape:", VP.shape)
        # Loop on layers :
        modes = []
        prev_lay = None
        for no,(lay,next_lay,idx) in enumerate( \
                            zip(self.layers,self.layers[1:]+(None,),LI)) :
            if isinstance(lay, FDLay) :
                Phi = VP[:,idx]
                no_left = prev_lay is None or isinstance(prev_lay, FDLay)
                no_right = next_lay is None or isinstance(next_lay, FDLay)
                left,right = not no_left, not no_right
                Kz,C,RE = lay.mode_shape(Phi, Kx, W, Nu, \
                                         left=left, right=right)
            else :
                Ux,Uy,Uz = VP[:,idx],VP[:,idx+1],VP[:,idx+2]
                Ush = Ux.shape
                tab_U = np.empty( Ush[:-1]+(3*Ush[-1],), dtype=Ux.dtype)
                tab_U[...,::3],tab_U[...,1::3],tab_U[...,2::3] = Ux,Uy,Uz
                Kz,C,RE = lay.mode_shape(tab_U, Kx, W, Nu)
            indexes = np.where(RE<rel_err)[0]
            Kx,W,VP = Kx[indexes],W[indexes],VP[indexes]
            modes.append( [Kz[indexes],C[indexes],RE[indexes]] )
            for pm in modes[:no] :
                for i,e in enumerate(pm) :
                    pm[i] = e[indexes]
            prev_lay = lay
        nb_lay = self.nb_layers
        # Left-hand side
        vec_zero = np.zeros_like(W)
        if self.left_fluid is None :
            kz0,C0 = vec_zero,vec_zero
        else : # External fluid
            dec_idx0 = LI[nb_lay] # V size and beginning of kappa0*V
            idx_phi0 = LI[nb_lay+1][0]
            phi0 = VP[:, idx_phi0]
            vec_W = VP[:,:dec_idx0]
            vec_ikappa0_W = VP[:,dec_idx0:2*dec_idx0]
            vec_ikappa0_W = np.multiply.outer(vec_ikappa0_W,np.ones(2))
            pmkz0 = self.partial_waves(self.left_fluid, Kx, W, Nu)
            pmikz0_vec_W = np.einsum("ij,ik->ijk", vec_W, 1.0j*pmkz0)
            ecart = np.abs( vec_ikappa0_W - pmikz0_vec_W )
            test = (ecart <= (rel_kappa_err * np.abs( vec_ikappa0_W )))*1
            test = ( test.mean(axis=1) >= 0.3 )*1
            kz0 = (pmkz0*test).sum(axis=1)
            coef = test.sum(axis=1)
            if coef.max() == 2 :
                msg = "Error: two valid vertical wavenumbers"
                raise ValueError(msg)
            C0 = coef*phi0
        # Right-hand side
        if self.right_fluid is None :
            kze,Ce = vec_zero,vec_zero
        else : # External fluid
            dec_idx0 = LI[nb_lay] # V size
            dec_idxe = LI[-2] # Beginning of kappae*V
            idx_phie = LI[-1][0]
            phie = VP[..., idx_phie]
            vec_W = VP[:,:dec_idx0]
            vec_ikappae_W = VP[:,dec_idxe:dec_idxe+dec_idx0]
            vec_ikappae_W = np.multiply.outer(vec_ikappae_W,np.ones(2))
            pmkze = self.partial_waves(self.right_fluid, Kx, W, Nu)
            pmikze_vec_W = np.einsum("ij,ik->ijk", vec_W, 1.0j*pmkze)
            ecart = np.abs( vec_ikappae_W - pmikze_vec_W )
            test = (ecart <= (rel_kappa_err * np.abs( vec_ikappae_W )))*1
            test = ( test.mean(axis=1) >= 0.3 )*1
            kze = (pmkze*test).sum(axis=1)
            coef = test.sum(axis=1)
            if coef.max() == 2 :
                msg = "Error: two valid vertical wavenumbers"
                raise ValueError(msg)
            Ce = coef*phie
        # Kx and W are vectors, VP is a matrix
        print("Kx.shape:", Kx.shape, "; W.shape:", W.shape, \
              "; VP.shape:", VP.shape)
        print("K;", K)
        # Previous returned tuple : (K, 0.5*W/np.pi, modes)
        F = 0.5*W/np.pi
        if np.ndim(K) == 0 : # A single k value
            # List of Plane_Guided_Mode instances
            pgm = []
            k, nu = K, Nu
            for i,f in enumerate(F) :
                mode = [ (kz[i],C[i,:],err[i]) for kz,C,err in modes ]
                mode.extend( [(kz0[i],C0[i]), (kze[i],Ce[i]) ] )
                pgm.append( Plane_Guided_Mode(self, f, k, mode, nu,
                                              normalized=normalized) )
            return pgm
        # np.ndim(K) == 1
        L_pgm = []
        nu = Nu
        for k in K :
            indexes = np.where( np.isclose(Kx, k) )[0]
            pgm = [] 
            for i in indexes :
                mode = [ (kz[i],C[i,:],err[i]) for kz,C,err in modes ]
                mode.extend( \
                    [(kz0[i],C0[i]), (kze[i],Ce[i]) ] )
                pgm.append( Plane_Guided_Mode(self, F[i], k, mode, nu,
                                                normalized=normalized) )
            L_pgm.append(pgm)
        return pgm
    #--------------------------------------------------
    def modes_for_given_frequency(self, F, Nu=0.0, sign=1, \
                                  rel_err=1e-4, rel_kappa_err=1e-2, \
                                  normalized=True) :
        """WARNING: New Version since 2021, February 8."""
        try :
            W = 2*np.pi*F
        except :
            F = np.array(F)
            W = 2*np.pi*F
        if np.ndim(F) == 0 : # W is a single value
            G,LI,LP = self.global_matrix_fixed_frequency(F,Nu,sign)
        elif np.ndim(F) == 1 : # K is vector
            return [ self.modes_for_given_frequency( frq, \
                                    Nu, sign, rel_err, rel_kappa_err) \
                      for frq in F  ]
        else :
            print("F cannot be an array with more than one index")
            return
        FDLay = FluidDiscretizedLayer
        miK, VP = np.linalg.eig(G)
        VP = np.swapaxes( VP, -1, -2)
        K = 1.0j*miK # Vector of wavenumbers in the x-direction
        indexes = np.where( K.round(0).real > 0 )[0]
        K,VP = K[indexes], VP[indexes]
        indexes = K.argsort()
        K,VP = K[indexes], VP[indexes]
        iw = 1.0j*W # Scalar
        W = W*np.ones(len(indexes)) # Vector
        # Loop on layers :
        modes = []
        for no,(lay,idx) in enumerate(zip(self.layers,LI)) :
            if isinstance(lay, FDLay) : # Fluid layer
                Phi = iw * VP[:,idx] # Psi -> Phi = i w Psi
                Kz,C,RE = lay.mode_shape(Phi, K, W, Nu)
            else : # Solid layer
                Ux,Uy,Uz = VP[:,idx],VP[:,idx+1],VP[:,idx+2]
                Ush = Ux.shape
                tab_U = np.empty( Ush[:-1]+(3*Ush[-1],), dtype=Ux.dtype)
                tab_U[...,::3],tab_U[...,1::3],tab_U[...,2::3] = Ux,Uy,Uz
                Kz,C,RE = lay.mode_shape(tab_U, K, W, Nu)
            indexes = np.where(RE<rel_err)[0]
            K,W,VP = K[indexes],W[indexes],VP[indexes]
            modes.append( [Kz[indexes],C[indexes],RE[indexes]] )
            for pm in modes[:no] : # Previous layers:
                for i,e in enumerate(pm) :
                    pm[i] = e[indexes]
        nb_lay = self.nb_layers
        # Left-hand side
        vec_zero = np.zeros_like(K)
        if self.left_fluid is None or self.left_fluid == "Wall"  :
            kz0,C0 = vec_zero,vec_zero
        else : # External fluid
            dec_idx0 = LI[nb_lay] # V size and beginning of kappa0*V
            idx_psi0 = LI[nb_lay+1][0]
            phi0 = iw * VP[:, idx_psi0] # Psi -> Phi = i w Psi
            vec_W = VP[:,:dec_idx0]
            vec_ikappa0_W = VP[:,dec_idx0:2*dec_idx0]
            vec_ikappa0_W = np.multiply.outer(vec_ikappa0_W,np.ones(2))
            pmkz0 = self.partial_waves(self.left_fluid, K, W, Nu)
            pmikz0_vec_W = np.einsum("ij,ik->ijk", vec_W, 1.0j*pmkz0)
            ecart = np.abs( vec_ikappa0_W - pmikz0_vec_W )
            test = (ecart <= (rel_kappa_err * np.abs( vec_ikappa0_W )))*1
            test = ( test.mean(axis=1) >= 0.3 )*1
            kz0 = (pmkz0*test).sum(axis=1)
            coef = test.sum(axis=1)
            if coef.max() == 2 :
                msg = "Error: two valid vertical wavenumbers"
                raise ValueError(msg)
            C0 = coef*phi0
        # Right-hand side
        if self.right_fluid is None or self.right_fluid == "Wall" :
            kze,Ce = vec_zero,vec_zero
        else : # External fluid
            dec_idx0 = LI[nb_lay] # V size
            dec_idxe = LI[-2] # Beginning of kappae*V
            idx_psie = LI[-1][0]
            phie = iw * VP[:, idx_psie] # Psi -> Phi = i w Psi
            vec_W = VP[:,:dec_idx0]
            vec_ikappae_W = VP[:,dec_idxe:dec_idxe+dec_idx0]
            vec_ikappae_W = np.multiply.outer(vec_ikappae_W,np.ones(2))
            pmkze = self.partial_waves(self.right_fluid, K, W, Nu)
            pmikze_vec_W = np.einsum("ij,ik->ijk", vec_W, 1.0j*pmkze)
            ecart = np.abs( vec_ikappae_W - pmikze_vec_W )
            test = (ecart <= (rel_kappa_err * np.abs( vec_ikappae_W )))*1
            test = ( test.mean(axis=1) >= 0.3 )*1
            kze = (pmkze*test).sum(axis=1)
            coef = test.sum(axis=1)
            if coef.max() == 2 :
                msg = "Error: two valid vertical wavenumbers"
                raise ValueError(msg)
            Ce = coef*phie
            # modes.append([kze,Ce]) # useless
        # List of Plane_Guided_Mode instances
        mode_instances = []
        f, nu = F, Nu
        for i,kx in enumerate(K) :
            mode = [ (kz[i],C[i,:],err[i]) for kz,C,err in modes ]
            mode.extend( [(kz0[i],C0[i]), (kze[i],Ce[i]) ] )
            mode_instances.append( \
                            Plane_Guided_Mode(self, f, kx, mode, nu, \
                                              normalized=normalized) )
        return mode_instances           
    #--------------------------------------------------
    def modes_for_given_slowness(self, S, signs=[1,1], rel_arr=1e-3) :
        pass
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++   
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def first_frequency_values_for_given_slowness(self, S, nb=20,
                               sort_by_imag = True, threshold_imag=0.1, \
                               with_mode_shapes=False,verbose=False,
                               signs=(1,1)) :
        """S is the value of the slowness (scalar).
           nb is the number of wanted frequency values"""
        G,LI,LP = self.global_matrix_fixed_slowness(S, signs=signs, \
                                                       verbose=verbose)
        if G is None :
            if verbose :
                print(f"Degenerate case for 1/s ~ {1e-3/S:.5f} mm/µs")
            return
        if GeneralDiscretizedLayer.SPARSE :
            G = G.toarray()
        if with_mode_shapes :
            iW, T_shapes = np.linalg.eig(G)
        else :
            iW, T_shapes = np.linalg.eigvals(G), None
        T_freq = (-0.5j/np.pi)*iW
        result = self.take_first(T_freq, nb, epsilon=200.0,\
                                    check_opposite=False, \
                                    array=T_shapes, \
                                    by_imag=sort_by_imag, \
                                    threshold_imag=threshold_imag, \
                                    verbose=verbose
                                 )
        if result is None :
            return None
        frequencies, mode_shapes, emax = result
        if with_mode_shapes : return frequencies, mode_shapes
        else : return frequencies # eigenfrequencies only
    #--------------------------------------------------
    @staticmethod
    def take_first_indexes(A, nb, dtype="Frequency", epsilon=1e-3, \
                            by_imag=False, threshold_imag = 1.0, \
                            verbose=False) :
        TFI = DiscretizedMultilayerPlate.take_first_indexes
        if dtype.lower() in ("frequency","f","w","omega") :
            in_freq = True
        elif dtype.lower() in ("wavenumber","k") :
            in_freq = False
        else :
            msg = "DiscretizedMultilayerPlate.take_first_indexes:" + \
                 f"\n\tError: unknown '{dtype}' dtype"
            raise ValueError(msg)
        if verbose : print("take_first_indexes :\n\tin_freq =",in_freq)
        if A.ndim == 1 :
            if in_freq : # Frequencies
                if verbose :
                    print("\tby_imag = {}".format(by_imag))
                if by_imag :
                    A_real = A.real
                    max_real = A_real.max()
                    if verbose :
                        print(("\tepsilon ~ {:.3e}\tmax_real ~ {:.3e}" \
                               ).format(epsilon,max_real))
                    if epsilon > max_real :
                        if verbose : print("\tepsilon > max_real")
                        print("DiscretizedMultilayerPlate." + \
                              "take_first_indexes:\n\tWarning:" + \
                              f"max real value ~ {max_real:.2e}")
                        B = ( A_real >= max_real-epsilon)
                    else :
                        if verbose : print("\tepsilon <= max_real")
                        abs_A_imag = np.abs(A.imag)
                        B = (A_real >= epsilon)
                        indexes = np.where(B)[0]
                        min_imag = abs_A_imag[indexes].min()
                        if verbose :
                            print(("\tthreshold_imag ~ {:.3e}" + \
                                   "\tmin_imag ~ {:.3e}" \
                                   ).format(threshold_imag,min_imag))
                        if min_imag > threshold_imag :
                            print("DiscretizedMultilayerPlate." + \
                                  "take_first_indexes:\n\tWarning:" + \
                                  f"threshold {threshold_imag:.2e}" + \
                                   " too low (min imaginary value ~ "+ \
                                  f"{min_imag:.2e})")
                            threshold_imag = 1.1*min_imag
                        elif verbose :
                            print("\tmin_imag <= threshold_imag")
                        B = B & (abs_A_imag <= threshold_imag)
                else :
                    B = (A.real >= epsilon) & (A.imag >= -epsilon)
            else :  # Wavenumbers
                B = (A.real >= -epsilon) & (A.imag <= epsilon)
            idxp = np.where(B)[0]
            Aplus = A[idxp]
            if in_freq :  # Frequencies
                if by_imag :
                    L = [ (abs(z.imag),abs(z.real),i) for i,z in \
                          zip(idxp,Aplus) ]

                else :
                    L = [ (z.real,z.imag,i) for i,z in zip(idxp,Aplus) ]
            else :   # Wavenumbers 
                L = [ ((abs(z.imag)>epsilon)*abs(z.imag), \
                       (abs(z.real)>epsilon)*(-z.real),i) \
                                     for i,z in zip(idxp,Aplus) ]
            L.sort()
            return ( np.array([ trp[2] for trp in L[:nb] ]), )
        # V.ndim > 1 :
        T = []
        for i,S in enumerate(A) :
            L = [ e.tolist() for e in TFI(S, nb, dtype, epsilon, \
                                          by_imag, threshold_imag) ]
            T.append( [i*np.ones_like(L[-1])] + L )
        return tuple( np.array(T).swapaxes(0,1) )
    #--------------------------------------------------
    @staticmethod
    def take_first(T, nb, dtype="Frequency", epsilon=1e-3, \
                   check_opposite=True, array=None, \
                   by_imag=False, threshold_imag = 1.0, \
                   verbose = False ):
        TFI = DiscretizedMultilayerPlate.take_first_indexes
        if dtype.lower() in ("frequency","f","w","omega") :
            in_freq = True
        elif dtype.lower() in ("wavenumber","k") :
            in_freq = False
        else :
            msg = "DiscretizedMultilayerPlate.take_first:" + \
                 f"\n\tError: unknown '{dtype}' dtype"
            raise ValueError(msg)
        indexes = TFI(T, nb, dtype, epsilon, by_imag, threshold_imag, \
                      verbose=verbose)
        T_is_array = True
        if len(indexes) == 1 :
            T_is_array = False
        if len(indexes) == 0 :
            return None
        Tplus = T[indexes]
        if array is None :
            tab = None
        elif T_is_array :
            tab = array[ indexes[:-1]+(slice(None),)+indexes[-1:] ]
        else :
            tab = array[ (slice(None),)+indexes ]
        if check_opposite :
            if in_freq :
                Tmoins = T[ TFI(-T,nb,dtype,epsilon,verbose=verbose) ]
                emax = (Tplus+Tmoins).max()
            else :
                Tmoins = T[ TFI(-T.conjugate(),nb,dtype,epsilon,\
                                verbose=verbose) ]
                emax = (Tplus+Tmoins.conjugate()).max()
        else :
            emax = None
        return Tplus, tab, emax        
#=========================================================================
if __name__ == "__main__" :
    water = Fluid({"rho":1000.0,"c":1500.0},"Water")
    air = Fluid({"rho":1.3,"c":340.0},"Air")
    gaz = Fluid({"rho":5.0,"c":280.0},"Gaz")
    # Olivier Poncelet
    alu = IsotropicElasticSolid({"rho":2800.0, "cL":6.37e3, "cT":3.1e3}, \
                             "Aluminum")
    # Aditya Krishna
    #alu = IsotropicElasticSolid({"rho":2700.0, "Young modulus":6.9e10, \
    #                             "Poisson ratio":0.3}, "Aluminum")
    steel = IsotropicElasticSolid({"rho":7900.0, "c11":2.8e11, \
                                   "c44":8.0e10}, "Steel")
    # Thèse Guillaume Neau
    crbepx = AnisotropicElasticSolid({"rho":1560.0, \
                  "c11": 8.665e10, "c22": 1.350e10, "c33": 1.400e10, \
                  "c44":  2.72e09, "c55":  4.06e09, "c66":  4.70e09, \
                  "c12":  9.00e09, "c13":  6.40e09, "c23":  6.80e09, \
                  "c14": 0.0, "c15": 0.0, "c16": 0.0, "c24": 0.0, \
                  "c25": 0.0, "c26": 0.0, "c34": 0.0, "c35": 0.0, \
                  "c36": 0.0, "c45": 0.0, "c46": 0.0, "c56": 0.0}, \
                                        "Carbon-Epoxy")
    ###########################
    CASE = 1
    ###########################
    #======================================================================
    if CASE == 3 : # Basic tests for multilayers for fixed slowness
        thickness,slowness = 2e-3,0.25e-3
        my_plate = DiscretizedMultilayerPlate(water, thickness, 10, 6)
        #my_plate.add_discretizedLayer(crbepx, 1e-3, 4)
        #my_plate.set_left_fluid(gaz)
        my_plate.add_discretizedLayer(water, 0.5e-3, 10)
        #my_plate.set_right_fluid(air)
        #my_plate.set_left_fluid("Wall")
        my_plate.set_right_fluid("Wall")
        M,LI,LP = my_plate.global_matrix_fixed_slowness(slowness)
        print_matrix_structure(M)
        Lf = 1e-6*np.linalg.eigvals(M).imag/np.pi
        Lf.sort()
        print(Lf.round(5))
        #Vf = my_plate.first_frequency_values_for_given_slowness(slowness, \
        #                                threshold_imag=1e5)
        #Vf.sort()
        #print(Vf)
    #======================================================================
    if CASE == 2 : # Basic tests for multilayers for fixed frequency
        thickness,frequency = 2e-3,1e6
        subcase = "water"
                        # ("alu", "water", "water/air", "water/alu",
                        #  "carbon epoxy/alu", "carbon epoxy/water/alu",
                        #  "water/alu/vacuum")
        external_fluids = [air,gaz] # None, "Wall", Fluid
        if subcase == "alu" :
            # Monolayer
            my_plate1 = DiscretizedMultilayerPlate(alu, thickness, 40, 8)
            # Two layers
            my_plate2 = DiscretizedMultilayerPlate(alu, 0.7*thickness, \
                                                   25, 8)
            my_plate2.add_discretizedLayer(alu, 0.3*thickness, 15)
        elif subcase == "water" :
            # Monolayer
            my_plate1 = DiscretizedMultilayerPlate(water, thickness, \
                                                   100, 8)
            # Three layers
            my_plate2 = DiscretizedMultilayerPlate(water, 0.45*thickness, \
                                                   45, 8)
            my_plate2.add_discretizedLayer(water, 0.25*thickness, 25)
            my_plate2.add_discretizedLayer(water, 0.3*thickness, 30)
        elif subcase == "water/air" :
            # Water / Air
            my_plate1 = DiscretizedMultilayerPlate(water, thickness, 80, 8)
            my_plate1.add_discretizedLayer(air, 0.8*thickness, 50)
            # Air / Water
            my_plate2 = DiscretizedMultilayerPlate(air, 0.8*thickness, \
                                                   50, 8)
            my_plate2.add_discretizedLayer(water, thickness, 80)
        elif subcase == "water/alu" :
            # Water / Aluminum
            my_plate1 = DiscretizedMultilayerPlate(water, 0.7*thickness, \
                                                   40, 8)
            my_plate1.add_discretizedLayer(alu, thickness, 60)
            # Aluminum / Water
            my_plate2 = DiscretizedMultilayerPlate(alu, thickness, 60, 8)
            my_plate2.add_discretizedLayer(water, 0.7*thickness, 40)
        elif subcase == "carbon epoxy/alu" :
            # Aluminum / Carbon Epoxy
            my_plate1 = DiscretizedMultilayerPlate(alu, thickness, 42, 8)
            my_plate1.add_discretizedLayer(crbepx, 0.5*thickness, 28) 
            # Carbon Epoxy / Aluminum
            my_plate2 = DiscretizedMultilayerPlate(crbepx, 0.5*thickness,\
                                                   28, 8)
            my_plate2.add_discretizedLayer(alu, thickness, 42)
        elif subcase == "carbon epoxy/water/alu" :
            my_plate1 = DiscretizedMultilayerPlate(crbepx, 0.5*thickness,\
                                                   30, 8)
            my_plate1.add_discretizedLayer(water, 0.7*thickness, 20)
            my_plate1.add_discretizedLayer(alu, thickness, 32)
            my_plate2 = DiscretizedMultilayerPlate(alu, thickness, 32, 8)
            my_plate2.add_discretizedLayer(water, 0.7*thickness, 20)
            my_plate2.add_discretizedLayer(crbepx, 0.5*thickness, 30)
        elif subcase == "water/alu/vacuum" :
            my_plate1 = DiscretizedMultilayerPlate(alu, thickness, 40, 8)
            my_plate2 = DiscretizedMultilayerPlate(water, 0.1*thickness, \
                                                   50, 8)
            my_plate2.add_discretizedLayer(alu, thickness, 40)
        else :
            print(f"Unknwon subcase '{subcase}'")
        # left & right sides
        my_plate1.set_left_fluid(external_fluids[0])
        my_plate1.set_right_fluid(external_fluids[1])
        my_plate2.set_left_fluid(external_fluids[1])
        my_plate2.set_right_fluid(external_fluids[0])
        T1 = my_plate1.first_wavenumber_values(frequency)
        T2 = my_plate2.first_wavenumber_values(frequency)
        msg = "Comparison of wavenumbers :"
        if subcase == "water" and \
           set(external_fluids).issubset({None,"Wall"}) :
            nb, = T1.shape
            kc = 2*np.pi*frequency/water.c
            if my_plate1.left_fluid == "Wall" and \
               my_plate1.right_fluid == "Wall":
                VKx = [ -1j*np.sqrt( \
                            (-1+0j)*(kc**2 - (n*np.pi/my_plate1.e)**2) \
                          ) for n in range(nb) ]
            elif my_plate1.left_fluid is None and \
                 my_plate1.right_fluid is None :
                VKx = [ -1j*np.sqrt( \
                            (-1+0j)*(kc**2 - (n*np.pi/my_plate1.e)**2) \
                          ) for n in range(1,nb+1) ]
            else :
                VKx = [ -1j*np.sqrt( \
                            (-1+0j)*(kc**2 - ((n+0.5)*np.pi/my_plate1.e)**2) \
                          ) for n in range(nb) ]
            for kx1,kx2,kxth in zip(T1,T2,VKx) :
                msg+= "\n"
                for kx in kx1,kx2,kxth :
                    cv = "{:.3f}+({:.3f}j)".format(kx.real,kx.imag)
                    msg += "{:>25}".format(cv)
        else :
            for kx1,kx2 in zip(T1,T2) :
                msg+= "\n"
                for kx in kx1,kx2 :
                    cv = "{:.3f}+({:.3f}j)".format(kx.real,kx.imag)
                    msg += "{:>25}".format(cv)
        print(msg)
    #======================================================================
    if CASE == 1 : # Basic tests for multilayers for fixed wavenumber
        LT = []
        sub_cases = ("a","b","c","d")
        for c in sub_cases :
            if c in ("a","b") : # A single aluminum layer
                my_plate = DiscretizedMultilayerPlate(alu, 1.0e-3, 100, 8)
            else : # 3 aluminum layers
                my_plate = DiscretizedMultilayerPlate(alu, 0.6e-3, 60, 8)
                my_plate.add_discretizedLayer(alu, 0.3e-3, 28)
                my_plate.add_discretizedLayer(alu, 0.1e-3, 12)
            if c in ("a","c") : # A single water layer
                my_plate.add_discretizedLayer(water, 0.6e-3, 60)
            else :
                my_plate.add_discretizedLayer(water, 0.25e-3, 25)
                my_plate.add_discretizedLayer(water, 0.35e-3, 35)
            my_plate.set_left_fluid(water)
            #my_plate.set_right_fluid(water)
            pgm = my_plate.modes_for_given_wavenumber(1000.0)
            print(f"Case '{c}':\n\t" + \
                f"{[round(1e3*l.thickness,3) for l in my_plate.layers]}"+\
                 "\n\t" + \
                f"{[round(1e-3*l.material.rho,3) for l in my_plate.layers]}"+\
                 "\n===========")
            LT.append( [m.f for m in pgm] )
        n_min = min([len(r) for r in LT])
        LF = np.array([ r[:n_min] for r in LT ])
        F,S = LF.mean(axis=0),LF.std(axis=0)
        indexes = np.where(S < 1e-5)[0]
        Ff,Sf = F[indexes],S[indexes]
        print("Eigenfrequencies [MHz]     | " + \
              "Standard dev. on 4 cases:\n" + \
               "\n".join([f"{f:26.9f} | {s:.3e}" for f,s in zip(Ff,Sf)]))
