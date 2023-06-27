# Version 1.7 - 2023, April, 14
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
# ====== Initialization ====================================================
if __name__ == "__main__" : from TraFiC_init import *
# ====== Material Classes ==================================================
from MaterialClasses import  *
from USMaterialClasses import  *
# ====== Numerical tools ===================================================
import numpy as np
from numpy import sqrt, exp, pi, sin, cos, log, array, meshgrid, multiply
outer =  multiply.outer # (n,),(k,) -> (n,k)
#===========================================================================
class USLayer :
    """Basic class with 3 attributes."""
    def __init__(self,thickness,us_material) :
        self.__m = us_material
        self.__n = 2*self.usm.nb_pw # number of partial waves
        self.__h = thickness
    @property
    def us_material(self) : return self.__m
    @property
    def usm(self) : return self.__m
    @property
    def thickness(self) : return self.__h
    @property
    def h(self) : return self.__h
    @property
    def npw(self) : return self.__n
    def __str__(self,number=None) :
        piece = "" if number is None else " #{}".format(number)
        msg = "Layer"+piece+" of {} and of thickness {:.2f} mm".format(\
            self.usm.mat.name,self.h*1e3)
        msg += " ({} partial waves)".format(self.npw)
        return msg
#===========================================================================
class USMultilayeredPlate :
    """ US multilayered plate containing layers, each layer characterized by
        a thickness and an oriented material."""
    def __init__(self,thickness,material,angles=None) :
        """Creates a monolayer plate in Vacuum."""
        usmat = USMaterial(material,angles,self)
        self.__Lusm = [usmat] # List of US materials
        if isinstance(usmat,USAniE) :
            self.__ani = True
            self.__iso = False
        else :
            self.__ani = False
            self.__iso = True
        new_layer = USLayer(thickness,usmat)
        self.__lyrs = [ new_layer ] # List of layers
        self.__Lz = [ 0.0, thickness ] # List of interfaces positions
        self.__nbi = 2 # Number of interfaces
        self.__dim = new_layer.npw # Number of unknown coefficients
        self.__shp = () # Shape of the array of dim-by-dim matrices
        self.__M = array([]) # Array of dim-by-dim matrices
        self.__R = [new_layer.npw//2] # Row indexes
        self.__C = [0,new_layer.npw ] # Column indexes
        self.__Vs = None # Vector of s values
        self.__Vkx = None # Vector of Kx values
        self.__Vky = None # Vector of Ky values
        self.__ths = None # Top Half-Space (z<0)
        self.__bhs = None # Bottom Half-Space (z>zmax=self.Z[-1])
    #--------------------------------------------------------
    @property
    def dim(self) : return self.__dim
    @property
    def shape(self) :
        "S-by-K shape"
        return self.__shp[:-2]
    @property
    def M(self) :
        if self.__M is None :
            return None
        # return self.__M.copy() # too high cost
        self.__M.flags.writeable = False # to protect data
        return self.__M
    @property
    def layers(self) : return self.__lyrs
    @property
    def topUSmat(self) : return self.__ths
    @property
    def bottomUSmat(self) : return self.__bhs
    @property
    def Z(self) : return tuple(self.__Lz)
    @property
    def C(self) : return tuple(self.__C)
    @property
    def R(self) : return tuple(self.__R)
    @property
    def Vs(self) :
        if self.__Vs is None : return tuple()
        return tuple(self.__Vs.tolist())
    @property
    def Vkx(self) : 
        if self.__Vkx is None : return tuple()
        return tuple(self.__Vkx.tolist())
    @property
    def Vky(self) :
        if self.__Vky is None : return tuple()
        return tuple(self.__Vky.tolist()) 
    @property
    def nMBytes(self) :
        mbs = 0
        for usm in self.__Lusm :
            mbs += usm.nMBytes
        if self.M is not None :
            mbs += 2**-20*self.M.nbytes
        return mbs    
    @property
    def nBytes_max_per_value(self) :
        mbs = 0
        for usm in self.__Lusm :
            mbs += usm.nBytes_max_per_value
        # M matrix
        mbs += self.dim*self.dim*16 # size of np.complex128
        return mbs     
    @property
    def nbPartialWaves(self) :
        npw = 0
        for usm in self.__Lusm :
            npw += 2*usm.nb_pw
        return npw
    #--------------------------------------------------------
    def appendLayer(self,thickness,material,angles=None) :
        new_usm = USMaterial(material,angles,self)
        new_layer = USLayer(thickness,new_usm)
        prev_layer = self.__lyrs[-1]
        self.__lyrs.append(new_layer)
        self.__Lz.append(self.__Lz[-1]+thickness) 
        self.__dim += new_layer.npw # Number of unknown coefficients
        self.__nbi += 1 # Number of interfaces
        new_nb_eq = prev_layer.usm.nb_pw + new_layer.usm.nb_pw
        self.__R.append(self.__R[-1]+new_nb_eq)            
        self.__C.append(self.__C[-1]+new_layer.npw)
        if isinstance(new_usm,USAniE) : self.__ani = True
        else : self.__iso = True
        if new_usm not in self.__Lusm :
            self.__Lusm.append(new_usm)
            mat_to_update = new_usm
        else : mat_to_update = None
        if self.__Vs is not None : # Started computation
            Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
            self.update(Vs,Vkx,Vky,usmat=mat_to_update)
    #--------------------------------------------------------
    def setTopHalfSpace(self, material, angles=None) :
        """material is either a medium or None (vacuum) or 'Rigid Wall'."""
        if material is None : # dropping Top Half-Space
            if self.__ths is None : return # no change
            if self.__ths != "wall" :
                prev_nb_eq = self.__ths.nb_pw
                self.__dim -= prev_nb_eq
                self.__C = [ c - prev_nb_eq for c in self.__C ]
                self.__R = [ r - prev_nb_eq for r in self.__R ]
            self.__ths = None
            if self.__Vs is not None : # Started computation
                Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                self.update(Vs,Vkx,Vky,usmat=None)
        elif isinstance(material, str) :
            if "wall" in material.lower() :
                if self.__ths == "wall" : return # no change
                if self.layers[0].npw != 2 :
                    print( "USMultilayerdPlate.setTopHalfSpace:\n\p" + \
                          f"Impossible '{material}' condition: the " + \
                           "first layer is solid!" )
                    return                    
                if self.__ths is not None :
                    prev_nb_eq = self.__ths.nb_pw
                    self.__dim -= prev_nb_eq
                    self.__C = [ c - prev_nb_eq for c in self.__C ]
                    self.__R = [ r - prev_nb_eq for r in self.__R ]
                self.__ths = "wall"
                if self.__Vs is not None : # Started computation
                    Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                    self.update(Vs,Vkx,Vky,usmat=None)
            else :
                print( "USMultilayerdPlate.setTopHalfSpace:\n\p" + \
                      f"Unknown '{material}' condition!" )
                return
        else :
            new_usmat = USMaterial(material, angles, self)
            if new_usmat is self.__ths : return # no change
            if self.__ths is None or self.__ths == "wall" :
                # adding Top Half-Space
                prev_nb = 0
            else : # changing Top Half-Space
                prev_nb = self.__ths.nb_pw
            new_nb = new_usmat.nb_pw
            dnb = new_nb - prev_nb
            self.__dim += dnb
            self.__C = [ c + dnb for c in self.__C ]
            self.__R = [ r + dnb for r in self.__R ]
            self.__ths = new_usmat
            if new_usmat not in self.__Lusm :
                self.__Lusm.append(new_usmat)
                mat_to_update = new_usmat
            else : mat_to_update = None
            if self.__Vs is not None : # Started computation
                Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                self.update(Vs,Vkx,Vky,usmat=mat_to_update)
    #--------------------------------------------------------            
    def setBottomHalfSpace(self, material, angles=None) :
        """material is either a medium or None (vacuum) or 'Rigid Wall'."""
        if material is None : # dropping Bottom Half-Space
            if self.__bhs is None : return # no change
            if self.__bhs != "wall" :
                prev_nb_eq = self.__bhs.nb_pw
                self.__dim -= prev_nb_eq
            # self.__C and self.__R unchanged
            self.__bhs = None
            if self.__Vs is not None : # Started computation
                Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                self.update(Vs,Vkx,Vky,usmat=None)
        elif isinstance(material, str) :
            if "wall" in material.lower() :
                if self.__bhs == "wall" : return # no change
                if self.layers[-1].npw != 2 :
                    print( "USMultilayerdPlate.setBottomeHalfSpace: " + \
                          f"\n\pImpossible '{material}' condition: " + \
                           "the first layer is solid!" )
                    return 
                if self.__bhs is not None :
                    prev_nb_eq = self.__bhs.nb_pw
                    self.__dim -= prev_nb_eq
                # self.__C and self.__R unchanged
                self.__bhs = "wall"
                if self.__Vs is not None : # Started computation
                    Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                    self.update(Vs,Vkx,Vky,usmat=None)
            else :
                print( "USMultilayerdPlate.setBottomHalfSpace : \n\p" + \
                      f"Unknown '{material}' condition!" )
                return
        else :
            new_usmat = USMaterial(material,angles,self)
            if new_usmat is self.__bhs : return # no change
            if self.__bhs is None or self.__bhs == "wall":
                # adding Bottom Half-Space
                prev_nb = 0
            else : # changing Bottom Half-Space
                prev_nb = self.__bhs.nb_pw
            new_nb = new_usmat.nb_pw
            dnb = new_nb - prev_nb
            self.__dim += dnb
            # self.__C and self.__R unchanged
            self.__bhs = new_usmat
            if new_usmat not in self.__Lusm :
                self.__Lusm.append(new_usmat)
                mat_to_update = new_usmat
            else : mat_to_update = None
            if self.__Vs is not None : # Started computation
                Vs,Vkx,Vky = self.__Vs,self.__Vkx,self.__Vky
                self.update(Vs,Vkx,Vky,usmat=mat_to_update)
    #--------------------------------------------------------   
    def update(self,Vs,Vkx,Vky=None,usmat="All",buildM=True) :
        if (self.__Vs is not Vs) or (self.__Vkx is not Vkx) or \
           (self.__Vky is not Vky) : # detected changes
            usmat="All"
        twoD = Vky is None
        if usmat is not None : # one or more US material to update
            self.__Vs = array(Vs)
            self.__Vkx = array(Vkx)
            if twoD : # 2D
                self.__Vky = None
                S2,Kx = meshgrid(self.__Vs**2,self.__Vkx,indexing="ij")
                Kx2 = Kx**2
                if usmat == "All" :
                    for usm in self.__Lusm : usm.update(S2,Kx,Kx2)
                else : usmat.update(S2,Kx,Kx2)
            else : # 3D
                self.__Vky = array(Vky)
                S2,Kx,Ky = meshgrid(self.__Vs**2,self.__Vkx,self.__Vky,\
                                    indexing="ij")
                Kx2,Ky2 = Kx**2,Ky**2
                if usmat == "All" :
                    if self.__ani : Kxy = Kx*Ky
                    if self.__iso :
                        K2 = Kx2+Ky2
                        K = sqrt(K2)
                        Q = np.arctan2(Ky,Kx)
                    for usm in self.__Lusm :
                        if isinstance(usm,USAniE) : # anisotropic
                            usm.update(S2,Kx,Kx2,Ky,Ky2,Kxy)
                        else : # isotropic
                            usm.update(S2,K,K2,Q)
                elif isinstance(usmat,USAniE) : # anisotropic
                    Kxy = Kx*Ky
                    usmat.update(S2,Kx,Kx2,Ky,Ky2,Kxy)
                else : # isotropic
                    K2 = Kx2+Ky2
                    K = sqrt(K2)
                    Q = np.arctan2(Ky,Kx)
                    usmat.update(S2,K,K2,Q)            
            self.__shp = tuple( list(S2.shape)+[self.dim,self.dim] )
        else : # usmat == None and Vs,Vkx,Vky did not change
            self.__shp = tuple( list(self.__shp[:-2])+[self.dim,self.dim] )
        # Global matrix
        if buildM : self.rebuildM(forced=True)
        else : self.clearM()
    #--------------------------------------------------------   
    def clearSK(self) :
        """Clears all US parameters. Useful to release memory."""
        self.__Vs = None
        self.__Vkx = None
        self.__Vky = None
        self.__M = None
        for usm in self.__Lusm : usm.clearSK()
    #--------------------------------------------------------   
    def rebuildM(self,forced=False) :
        """rebuild the M array."""
        if self.__M is not None and not forced :
            print("M has already been updated.")
            return
        if (self.__Vs is None) or (self.__Vkx is None) :
            print("M cannot be updated because S or Kx are not defined.")
            return
        twoD = self.__Vky is None # 2D/3D
        self.__M = np.zeros( self.__shp, dtype=complex )
##        # For checking parameters
##        C,R,d = list(self.C),list(self.R),self.dim
##        print("update -> dim:",d,"; C:",C,"; R:",R)
        # ++++++ Interface at z = 0 ++++++
        up_layer = self.layers[0]
        usm = up_layer.usm
        FC = usm.fieldComponents(-up_layer.h,0.0)
        if self.__ths is None : # No Top Half-Space 
            if up_layer.npw == 6 : # Vacuum/Solid  (3 equations)                
                self.__M[...,:3,:6] = FC[...,3:,:]
            else : # Vacuum/Fluid, up_layer.npw == 2 (1 equation : Szz=0)                            
                self.__M[...,0,:2] = FC[...,5,:]
        elif self.__ths == "wall" :
            # Wall/Fluid, up_layer.npw == 2 (1 equation : Uz=0)                            
            self.__M[...,0,:2] = FC[...,2,:]
        elif self.__ths.nb_pw == 3 : # Solid
            FCths = self.__ths.fieldComponents(0.0,0.0)
            if up_layer.npw == 6 : # Solid/Solid  (6 equations)              
                self.__M[...,:6,:3] = -FCths[...,:3]
                self.__M[...,:6,3:9] = FC 
            else : # Solid/Fluid, up_layer.npw == 2 (4 equations)             
                self.__M[...,:4,:3] = -FCths[...,-4:,:3]
                self.__M[...,:4,3:5] = FC[...,-4:,:]
        else : # Fluid, self.__ths.nb_pw == 1 
            FCths = self.__ths.fieldComponents(0.0,0.0)
            if up_layer.npw == 6 : # Fluid/Solid  (4 equations)              
                self.__M[...,:4,0] = -FCths[...,-4:,0]
                self.__M[...,:4,1:7] = FC[...,-4:,:] 
            else : # Fluid/Fluid, up_layer.npw == 2 (2 equations)             
                self.__M[...,:2,0] = -FCths[...,2::3,0]
                self.__M[...,:2,1:3] = FC[...,2::3,:]
        # ++++++ Interface at z = zmax ++++++
        down_layer = self.layers[-1]
        usm = down_layer.usm
        FC = usm.fieldComponents(0.0,down_layer.h)
        if self.__bhs is None : # No Bottom Half-Space
            if down_layer.npw == 6 : # Solid/Vacuum            
                self.__M[...,-3:,-6:] = -FC[...,3:,:]
            else : # Fluid/Vacuum, down_layer.npw == 2  (Sz=0)                            
                self.__M[...,-1,-2:] = -FC[...,5,:]
        elif self.__bhs == "wall" :
            # Fluid/Wall, down_layer.npw == 2  (Uz=0)                            
            self.__M[...,-1,-2:] = -FC[...,2,:]
        elif self.__bhs.nb_pw == 3 : # Solid
            FCbhs = self.__bhs.fieldComponents(0.0,0.0)
            if down_layer.npw == 6 : # Solid/Solid  (6 equations)
                self.__M[...,-6:,-9:-3] = -FC                
                self.__M[...,-6:,-3:] = FCbhs[...,3:]
            else : # Fluid/Solid, down_layer.npw == 2 (4 equations) 
                self.__M[...,-4:,-5:-3] = -FC[...,-4:,:]            
                self.__M[...,-4:,-3:] = FCbhs[...,-4:,3:]
        else : # Fluid, self.__bhs.nb_pw == 1 
            FCbhs = self.__bhs.fieldComponents(0.0,0.0)
            if down_layer.npw == 6 : # Solid/Fluid  (4 equations)
                self.__M[...,-4:,-7:-1] = -FC[...,-4:,:]               
                self.__M[...,-4:,-1] = FCbhs[...,-4:,-1]
            else : # Fluid/Fluid, down_layer.npw == 2 (2 equations)
                self.__M[...,-2:,-3:-1] = -FC[...,2::3,:]
                self.__M[...,-2:,-1] = FCbhs[...,2::3,-1]
        # ++++++ Inner Interfaces ++++++  
        up_lay = self.layers[0]
        rmin = self.R[0]
        cmin,cmil = self.C[:2]
        for down_lay,cmax,rmax in \
            zip(self.layers[1:],self.C[2:],self.R[1:]) :
            FCup = up_lay.usm.fieldComponents(0.0,up_lay.h)
            FCdown = down_lay.usm.fieldComponents(-down_lay.h,0.0)
            if up_lay.npw == 6 :
                if down_lay.npw == 6 : # Solid/Solid                                
                    self.__M[...,rmin:rmax,cmin:cmil] = -FCup
                    self.__M[...,rmin:rmax,cmil:cmax] = FCdown
                else : # down_lay.npw == 2, Solid/Fluid                               
                    self.__M[...,rmin:rmax,cmin:cmil] = -FCup[...,2:,:]
                    self.__M[...,rmin:rmax,cmil:cmax] = FCdown[...,2:,:]
            else : # up_lay.npw == 2
                if down_lay.npw == 6 : # Fluid/Solid                             
                    self.__M[...,rmin:rmax,cmin:cmil] = -FCup[...,2:,:]
                    self.__M[...,rmin:rmax,cmil:cmax] = FCdown[...,2:,:]
                else : # down_lay.npw == 2, Fluid/Fluid                            
                    self.__M[...,rmin:rmax,cmin:cmil] = -FCup[...,2:,:]
                    self.__M[...,rmin:rmax,cmil:cmax] = FCdown[...,2:,:]
            up_lay = down_lay
            cmin,cmil = cmil,cmax
            rmin = rmax
        return
    #--------------------------------------------------------   
    def clearM(self) :
        """Clears only the M array. Useful to release memory."""
        self.__M = None
        print("Warning: be careful to use the 'rebuildM' method "+\
              "if necessary.")
    #--------------------------------------------------------
    def field(self,z,Vc) :
        """Deprecated: use ``fields(z,Vc,"Stroh vector",output="array")''.
        """
        # Compatibility with previous versions
        return fields(self,z,Vc,"Stroh vector",output="array")
    #--------------------------------------------------------
    def fields(self,z,Vc,wanted="all",output = "list", tol_pos=1e-9) :
        """Returns arrays (ns,nx[,ny]) of displacements and stresses
           at the vertical position z. Vc is the array of the
           partial waves coefficients. 'wanted' is a string in
           ('all','displacements','stresses','Sx','Sy','Sz','Stroh vector'),
           or a sublist of ('Ux','Uy','Uz','Sxx','Syy','Szz','Syz','Sxz',
           'Sxy').
           The output can be a list of arrays (default) or a big array."""
        Z,C = self.Z,self.C
        if -tol_pos <= z < 0.0 : z = 0.0
        elif Z[-1] < z <= Z[-1]+tol_pos : z = Z[-1]
        # Possible choices
        choices = ("Ux","Uy","Uz","Sxx","Syy","Szz","Syz","Sxz","Sxy")
        if wanted == "all" :
            wanted = choices
        elif wanted in ("displacements","U") :
            wanted = ("Ux","Uy","Uz")
        elif wanted in ("stresses","sigma") :
            wanted = ("Sxx","Syy","Szz","Syz","Sxz","Sxy")
        elif wanted in ("vertical stresses","sigma_z","Sz") :
            wanted = ("Sxz","Syz","Szz")
        elif wanted in ("sigma_x","Sx") :
            wanted = ("Sxx","Sxy","Sxz")
        elif wanted in ("sigma_y","Sy") :
            wanted = ("Sxy","Syy","Syz")
        elif wanted == "Stroh vector" :
            wanted = ("Ux","Uy","Uz","Sxz","Syz","Szz")
        elif isinstance(wanted,str) :
            msg = "USMultilayeredPlate.fields :: unrecognized\n" + \
                  "value '{}' for wanted".format(wanted)
            raise ValueError(msg)
        else :
            for F in wanted :
                if F not in choices :
                    msg = "USMultilayeredPlate.fields :: unrecognized\n"+\
                           "value '{}' in {} for wanted".format(F,wanted)
                    raise ValueError(msg)       
        if (self.__Vs is None) or (self.__Vkx is None) :
            print("Fields cannot be calculated because S or Kx are not "+\
                  "defined.")
            return None
        #++++++++ Computation of the 6-dimensional state vector +++++++++++
        # Top half-space : z < 0
        if z < 0 : # top half-space
            usm = self.__ths
            if usm is None or usm == "wall": # Vacuum
                print("Error : z = {} mm is in vacuum".format(1e3*z))
                return None
            npw = usm.nb_pw
            if npw != C[0] :
                print("Error in top half-space : {} != {}".format(\
                    npw,C[0]))
                return None
            CC = np.zeros( self.shape+(2*npw,), dtype = np.complex128)
            CC[...,:npw] = Vc[...,:npw] # Upgoing waves only
            P = usm.fieldComponents(z,0)
            V6 = np.einsum(usm.MVprod,P,CC)
        # Layer in the plate
        else :
            search_layer,cg = True,C[0]
            for i,(zip1,cd) in enumerate(zip(Z[1:],C[1:])) :
                if z <= zip1 : # z in layer no.i
                    usm = self.layers[i].usm
                    P = usm.fieldComponents(z-zip1,z-Z[i])
                    V6 = np.einsum(usm.MVprod,P,Vc[...,cg:cd])
                    search_layer = False
                    break
                cg = cd
            # Bottom half-space : z > Z[-1]
            if search_layer :
                usm = self.__bhs
                if usm is None or usm == "wall" : # Vacuum/Wall
                    print("Error : z = {} mm is in vacuum".format(1e3*z))
                    return None
                npw = usm.nb_pw
                if npw != self.dim-C[-1] :
                    print("Error in bottom half-space : {} != {}".format(\
                        npw,self.dim-C[-1]))
                    return None
                CC = np.zeros( self.shape+(2*npw,), dtype = np.complex128)
                CC[...,-npw:] = Vc[...,-npw:] # Downgoing waves only
                P = usm.fieldComponents(0,z-Z[-1])
                V6 = np.einsum(usm.MVprod,P,CC)
        #++++++++ Selection of the wanted fields +++++++++++++++++++++++++
        deja_calcules = ("Ux","Uy","Uz","Sxz","Syz","Szz")
        flds = []
        for w in wanted :
            if w in deja_calcules :
                flds.append(V6[...,deja_calcules.index(w)])
            elif w == "Sxx" :
                flds.append(usm.sigmaXX(V6,self.__Vkx,self.__Vky))
            elif w == "Syy" :
                flds.append(usm.sigmaYY(V6,self.__Vkx,self.__Vky))
            elif w == "Sxy" :
                flds.append(usm.sigmaXY(V6,self.__Vkx,self.__Vky))
            else :
                print("???")
                raise           
        if output != "array" : return flds
        big_array = np.empty( flds[0].shape + (len(flds),), \
                              dtype = np.complex128)
        for i,F in enumerate(flds) :
            big_array[...,i] = F
        return big_array
    #--------------------------------------------------------
    def copy(self) :
        first_layer = self.layers[0]
        mat0 = first_layer.usm.mat
        new_plate = USMultilayeredPlate(first_layer.h,mat0)
        for layer in self.layers[1:] :
            mat = layer.usm.mat
            new_plate.appendLayer(layer.h,mat)
        if self.topUSmat is not None : # Top half-space is not vacuum
            mat = self.topUSmat.mat
            new_plate.setTopHalfSpace(mat)
        if self.bottomUSmat is not None : # Top bottom-space is not vacuum
            mat = self.bottomUSmat.mat
            new_plate.setBottomHalfSpace(mat)
        return new_plate
    #-------------------------------------------------------- 
    def __str__(self) :
        msg = 70*"*"+"\n"
        msg += "{}-layered plate of thickness {:.2f} mm and with " + \
              "{} partial waves :"
        msg = msg.format(len(self.layers),1e3*self.Z[-1],self.dim)
        msg += "\n----------"
        if self.__ths is None :
            ths = "Vacuum"
        elif self.__ths == "wall" :
            ths = "Rigid Wall"
        else :
            npw = self.__ths.nb_pw
            ths = self.__ths.mat.name + \
                     " ({} partial upgoing wave{})".format(npw,\
                     "s" if npw > 1 else "")
        msg += "\n Top Half-Space: {}".format(ths)
        noeq0 = 0
        for n,(noeq1,lay,z) in enumerate(zip(self.R,self.layers,self.Z)) :
            msg += "\n========== Interface #{} ({} equations) " + \
                   "at z = {:.2f} mm =========="
            msg = msg.format(n,noeq1-noeq0,1e3*z)
            noeq0 = noeq1
            msg += "\n "+lay.__str__(n)
        msg += "\n========== Interface #{} ({} equations) " + \
                   "at z = {:.2f} mm =========="
        msg = msg.format(n+1,self.dim-noeq0,1e3*self.Z[-1])
        if self.__bhs is None :
            bhs = "Vacuum"
        elif self.__bhs == "wall" :
            bhs = "Rigid Wall"
        else :
            npw = self.__bhs.nb_pw
            bhs = self.__bhs.mat.name + \
                     " ({} partial downgoing wave{})".format(npw,\
                     "s" if npw > 1 else "")
        msg += "\n Bottom Half-Space: {}".format(bhs)
        msg += "\n"+70*"*"
        return msg
#========= Tests ===========================================================
if __name__ == "__main__" :
    np.set_printoptions(precision=2)
    water = Fluid({"rho":1000.0,"c":1500.0},"Water")
    alu = IsotropicElasticSolid({"rho":2700.0,"Young modulus":6.9e10,\
                                 "Poisson ratio":0.3},"Aluminum")
    crbepx = TransverselyIsotropicElasticSolid({"rho":1560.0,\
                   "c11":1.4e10, "c12": 8.0e9, "c33": 8.7e10,\
                   "c13": 9.0e9, "c44": 4.7e9}, "Carbon-Epoxy")
    plate1 = USMultilayeredPlate(0.0004,alu)
# 3D Test for one set of values s,kx,ky
    # s,kx,ky = 100.0+20.0j, 80.0, 60.0
    # plate1.update([s],[kx],[ky])
    # print(plate1.M[0,0,0])
    plate1.appendLayer(0.0011,alu)
    plate1.appendLayer(0.0005,alu)
    #plate1.appendLayer(0.002,crbepx,(90.0,45.0))
    plate1.appendLayer(0.0005,water)
    plate1.setTopHalfSpace(water)
    plate1.setBottomHalfSpace(water)
    print(plate1)
##    C,R,d = list(plate1.C),list(plate1.R),plate1.dim
##    cmin,cmil,rmin = 0,C[0],0
##    msg = "zeros in [ {}:{} , {}:{} ]: {}"
##    for cmax,rmax in zip(C[1:]+[d],R+[d]) :
##        if cmin > 0 :
##            test = abs(plate1.M[0,0,0,rmin:rmax,:cmin]).max()==0.0
##            print(msg.format(rmin,rmax,0,cmin,test))
##        print("[ {}:{} , {}:{} ]".format(rmin,rmax,cmin,cmax))
##        print(plate1.M[0,0,0,rmin:rmax,cmin:cmax])
##        if cmax < d :
##            test = abs(plate1.M[0,0,0,rmin:rmax,cmax:]).max()==0.0
##            print(msg.format(rmin,rmax,cmax,d,test))
##        cmin,cmil,rmin = cmil,cmax,rmax
    # Comparison with modes
    from Modes_Monolayer_Plate import *
    from numpy.linalg import eigvals
    import matplotlib.pyplot as plt
    Kmax,Fmax = 3000.0,2e6
    vecK = np.linspace(0.0,Kmax,151)
    vecS = 2j*pi*np.linspace(0.0,Fmax,201) + 1.0e-3
    plate1.update(vecS,vecK)
    xmin,xmax = 5.0e-7*vecS[0].imag/pi,5.0e-7*vecS[-1].imag/pi
    ymin,ymax = 1.0e-3*vecK[0],1.0e-3*vecK[-1]
##    VPmin = abs(eigvals(plate1.M)).min(axis=-1).transpose()
##    VP = -10*np.log(VPmin)/np.log(10)
##    plt.figure("Smallest eigenvalue",figsize=(18,12))
##    plt.imshow(VP,extent=(xmin,xmax,ymin,ymax),aspect="auto",
##               origin="lower",interpolation="none",
##               vmin=VP[:,20:].min(),vmax=VP[:,20:].max())
##    plt.colorbar()
    vecKmod = np.linspace(0.0,Kmax,61)
##    modes = ModesMonolayerPlate(alu,0.002,40,vecKmod)
##    for no in range(10) :
##        plt.plot(1e-6*modes.F[:,no],1e-3*modes.K,"+",color=(1,0.6,1),
##                 markeredgewidth=1.2,markersize=8)
##    plt.xlabel(r"Frequency $f$ [$MHz$]",fontsize=14)
##    plt.ylabel(r"Horizontal wavenumber $k$ [$mm^{-1}$]",fontsize=14)
##    plt.grid()
##    for c in [alu.cL, alu.cT, water.c ] :
##        pente = 2e3*pi/c
##        plt.plot([0,max(xmax,ymax/pente)],[0,max(ymax,xmax*pente)],"w--")
##    plt.xlim(xmin,xmax)
##    plt.ylim(ymin,ymax)        
##    plt.show()
    angcrbepx = (90.0,30.0)
    plate2 = USMultilayeredPlate(0.001,crbepx,angcrbepx)
    #plate2 = USMultilayeredPlate(0.0007,crbepx,angcrbepx)
    #plate2.appendLayer(0.0003,crbepx,angcrbepx)
    print(plate2)
    plate2.update(vecS,vecK)
    VPmin2 = abs(eigvals(plate2.M)).min(axis=-1).transpose()
    VP2 = -10*np.log(VPmin2)/np.log(10)
    plt.figure("Smallest eigenvalue 2",figsize=(18,12))
    plt.imshow(VP2,extent=(xmin,xmax,ymin,ymax),aspect="auto",
               origin="lower",interpolation="none",
               vmin=VP2[:,20:].min(),vmax=VP2[:,20:].max())
    plt.colorbar()
    modes2 = ModesMonolayerPlate(crbepx.rotate(*angcrbepx),0.001,40,vecKmod)
    for no in range(10) :
        plt.plot(1e-6*modes2.F[:,no],1e-3*modes2.K,"+",color=(1,0.6,1),
                 markeredgewidth=1.2,markersize=8)
    plt.xlabel(r"Frequency $f$ [$MHz$]",fontsize=14)
    plt.ylabel(r"Horizontal wavenumber $k$ [$mm^{-1}$]",fontsize=14)
    plt.grid()
    plt.xlim(xmin,xmax)
    plt.ylim(ymin,ymax)        
    plt.show()
