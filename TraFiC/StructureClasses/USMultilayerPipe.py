# Version 1.21 - 2023, July, 11
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
# ====== Initialization ===================================================
import numpy as np
import matplotlib.pyplot as plt
import os, sys
# ====== Initialization ===================================================
if __name__ == "__main__" : from TraFiC_init import *
# ====== Material Classes =================================================
from MaterialClasses import  *
from USTubularMaterialClasses import  *
from USMultilayerPlate import USMultilayerPlate
#==========================================================================
def USMultilayerStructure( file_path ) :
    """Creates a USMultilayerPlate or USMultilayerPipe from a text file
       located at file_path."""
    msg = "USMultilayerStructure :: Error:\n\t"
    try :
        with open( file_path , "r", encoding="utf8") as strm :
            text = strm.read().lower()
    except Exception as err :
        msg += f"Cannot open '{file_path}'\n\t{err}"
        raise ValueError(msg)
    if "plate" in text :
        return USMultilayerPlate.import_from_file( file_path )
    elif "pipe" in text or "cylinder" in text :
        return USMultilayerPipe.import_from_file( file_path )
    else :
        msg += f"Cannot identify the structure in\n\t'{file_path}'"
        raise ValueError(msg) 
#==========================================================================
class USTubularLayer :
    """Basic class with 5 attributes."""
    def __init__(self, r_min, r_max, us_material) :
        self.__m = us_material
        self.__n = 2*self.usm.nb_pw # number of partial waves
        self.__rmin = r_min
        self.__rmax = r_max
        self.__h = r_max - r_min
    @property
    def us_material(self) : return self.__m
    @property
    def usm(self) : return self.__m
    @property
    def thickness(self) : return self.__h
    @property
    def h(self) : return self.__h
    @property
    def r_min(self) : return self.__rmin
    @property
    def r_max(self) : return self.__rmax
    @property
    def npw(self) : return self.__n
    def __str__(self, number=None) :
        piece = "" if number is None else " #{}".format(number)
        msg = "Tubular layer" + piece + \
             f" of {self.usm.mat.name} and of radii from " + \
             f"{self.r_min*1e3:.2f} mm to {self.r_max*1e3:.2f} mm" + \
             f"\n\t(thickness {self.h*1e3:.2f} mm,"
        msg += " {} partial waves)".format(self.npw)
        return msg
#==========================================================================
class USMultilayerPipe :
    """ US multilayer pipe containing layers, each layer characterized
        by a radius range and a material."""
    INCYL = "Inner Cylinder"
    EXTSP = "External Space"
    VACUUM = USMultilayerPlate.VACUUM
    WALL = USMultilayerPlate.WALL
    HS_CONDITIONS = [VACUUM, WALL]
    UNDEFINED = USMultilayerPlate.UNDEFINED
    # Horizontal lines for text files
    LINESIZE = USMultilayerPlate.LINESIZE
    DLINE = USMultilayerPlate.DLINE # Double line
    SLINE = USMultilayerPlate.SLINE # Simple line
    STARS = USMultilayerPlate.STARS # Line of stars
    TWOPOWM20 = 2**-20
    #----------------------------------------------------------------------
    def __init__(self, r_min, r_max, material, angles=None, \
                 verbose=False ) :
        """Creates a monolayer pipe (or cylinder) in Vacuum."""
        if r_min < 0.0 or r_max < 0.0 or r_max <= r_min :
            msg =  "USMultilayerPipe builder :: Error:\n\t" + \
                  f"Condition on radii 0 <= {r_min:.3e} < " + \
                  f"{r_max:.3e} not satisfied"
            raise ValueError(msg)
        self.__Vs = None  # Vector of s values
        self.__Vk = None  # Vector of k values (wavenumbers in the 
                          #                     axial direction)
        self.__Vn = None  # Vector of n values (wavenumbers in the
                          #                     azimuthal direction)
        self.__shp = ()   # Shape of the array of dim-by-dim matrices
        self.__M = np.array([]) # Array of dim-by-dim matrices

        self.__lhs = None # Internal cylinder
        self.__rhs = None # Infinite space (r>r_max=self.R[-1])
        if r_min > 0.0 : # Pipe
            usmat = USTubularMaterial( material, angles, USset=self )
            self.__Lusm = [usmat] # List of US tubular materials
            new_layer = USTubularLayer( r_min, r_max, usmat )
            self.__lyrs = [ new_layer ] # List of layers
            self.__R = [ r_min,r_max ] # List of radii
            self.__nbi = 2 # Number of interfaces
            self.__dim = new_layer.npw # Number of unknown coefficients
            self.__rows = [new_layer.npw//2] # Row indexes
            self.__cols = [0,new_layer.npw ] # Column indexes
        else : # Cylinder
            self.__Lusm = []
            self.__lyrs = [] # List of layers
            self.__R = [ r_max ] # List of radii
            self.__nbi = 1 # Number of interfaces
            self.__dim = 0 # Number of unknown coefficients
            self.__rows = [] # Row indexes
            self.__cols = [0] # Column indexes
            self.setInnerCylinder(material)
        self.__verb = verbose
    #----------------------------------------------------------------------
    def __prt(self, *args) :
        self.__verb : print(*args)
    #----------------------------------------------------------------------
    @property
    def dim(self) : return self.__dim
    @property
    def shape(self) : return self.__shp
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
    def internalUSmat(self) : return self.__lhs
    @property
    def externalUSmat(self) : return self.__rhs
    @property
    def R(self) : return tuple(self.__R)
    @property
    def n_cols(self) : return tuple(self.__cols)
    @property
    def n_rows(self) : return tuple(self.__rows)
    @property
    def Vs(self) :
        if self.__Vs is None : return tuple()
        return tuple(self.__Vs.tolist())
    @property
    def Vk(self) : 
        if self.__Vk is None : return tuple()
        return tuple(self.__Vk.tolist())
    @property
    def Vn(self) :
        if self.__Vn is None : return tuple()
        return tuple(self.__Vn.tolist()) 
    @property
    def nMBytes(self) :
        mbs = 0
        for usm in self.__Lusm :
            mbs += usm.nMBytes
        if self.M is not None :
            mbs += 2**-20*self.M.nbytes
        return mbs    
    @property
    def nbPartialWaves(self) :
        npw = 0
        for usm in self.__Lusm :
            npw += 2*usm.nb_pw
        return npw
    #----------------------------------------------------------------------
    def appendLayer(self, r_max, material, angles=None) :
        new_usm = USTubularMaterial(material, angles, USset=self)
        r_min = self.R[-1] # Radius of the existing pipe
        new_layer = USTubularLayer(r_min,r_max,new_usm)
        prev_layer = self.__lyrs[-1]
        self.__lyrs.append(new_layer)
        self.__R.append(r_max) 
        self.__dim += new_layer.npw # Number of unknown coefficients
        self.__nbi += 1 # Number of interfaces
        new_nb_eq = prev_layer.usm.nb_pw + new_layer.usm.nb_pw
        self.__rows.append(self.__rows[-1]+new_nb_eq)            
        self.__cols.append(self.__cols[-1]+new_layer.npw)
        if new_usm not in self.__Lusm :
            self.__Lusm.append(new_usm)
            mat_to_update = new_usm
        else : mat_to_update = None
        if self.__Vs is not None : # Started calculation
            Vs,Vk,Vn = self.__Vs,self.__Vk,self.__Vn
            self.update( Vs, Vk, Vn, usmat=mat_to_update )
    #----------------------------------------------------------------------
    def prependLayer(self, r_min, material, angles=None) :
        if r_min < 1e-5 and ( isinstance(material, Fluid) or \
                isinstance(material, IsotropicElasticSolid) or \
                isinstance(material, TransverselyIsotropicElasticSolid)) :
            # Internal cylinder
            self.setInnerCylinder( material )
            return
        new_usm = USTubularMaterial( material, angles, USset=self )
        r_max = self.R[0] # Radius of the existing pipe
        new_layer = USTubularLayer(r_min,r_max,new_usm)
        next_layer = self.__lyrs[0]
        self.__lyrs = [new_layer] + self.__lyrs
        self.__R = [r_min] + self.__R
        self.__dim += new_layer.npw # Number of unknown coefficients
        self.__nbi += 1 # Number of interfaces
        new_nb_eq = new_layer.usm.nb_pw + next_layer.usm.nb_pw
        self.__rows = [new_nb_eq] + \
                       [ r+new_nb_eq for r in self.__rows ]            
        self.__cols = [new_layer.usm.nb_pw] + \
                       [ c+new_nb_eq for c in self.__cols ] 
        if new_usm not in self.__Lusm :
            self.__Lusm.append(new_usm)
            mat_to_update = new_usm
        else : mat_to_update = None
        if self.__Vs is not None : # Started calculation
            Vs,Vk,Vn = self.__Vs,self.__Vk,self.__Vn
            self.update(Vs,Vk,Vn,usmat=mat_to_update)
    #----------------------------------------------------------------------
    def setInnerCylinder(self, material) :
        """material as to be either a fluid or a transversely
           isotropic elastic medium the symmetry axis of which
           being the axis of the pipe."""
        if material is None : # dropping Internal Cylinder
            if self.__lhs is None : return # no change
            prev_nb_eq = self.__lhs.nb_pw
            self.__dim -= prev_nb_eq
            self.__cols = [ c - prev_nb_eq for c in self.__cols ]
            self.__rows = [ r - prev_nb_eq for r in self.__rows ]
            self.__lhs = None
            if self.__Vs is not None : # Started calculation
                Vs,Vk,Vn = self.__Vs,self.__Vk,self.__Vn
                self.update(Vs,Vk,Vn,usmat=None)
            return
        elif not (isinstance(material, Fluid) or \
                  isinstance(material, IsotropicElasticSolid) or \
                  isinstance(material, \
                             TransverselyIsotropicElasticSolid) \
                  ) :
            msg = "USMultilayerPipe.setInnerCylinder :: Error:" + \
                  "\n\tThe material has to be either a fluid or a" + \
                  "\n\ttransversely isotropic elastic medium the " + \
                  "\n\tsymmetry axis of which being the axis of the " + \
                  "pipe."
            return
        new_usmat = USTubularMaterial(material, USset=self)
        if new_usmat is self.__lhs : return # no change
        if self.__lhs is None : # adding Internal Cylinder
            prev_nb = 0
        else : # changing Internal Cylinder
            prev_nb = self.__lhs.nb_pw
        new_nb = new_usmat.nb_pw
        dnb = new_nb - prev_nb
        self.__dim += dnb
        self.__cols = [ c + dnb for c in self.__cols ]
        self.__rows = [ r + dnb for r in self.__rows ]
        self.__lhs = new_usmat
        if new_usmat not in self.__Lusm :
            self.__Lusm.append(new_usmat)
            mat_to_update = new_usmat
        else : mat_to_update = None
        if self.__Vs is not None : # Started calculation
            Vs,Vk,Vn = self.__Vs, self.__Vk, self.__Vn
            self.update(Vs, Vk, Vn, usmat=mat_to_update )
    #----------------------------------------------------------------------
    def setExternalSpace(self, material) :
        """material as to be either a fluid or a transversely
           isotropic elastic medium the symmetry axis of which
           being the axis of the pipe."""
        if material is None : # dropping External Space
            if self.__rhs is None : return # no change
            prev_nb_eq = self.__rhs.nb_pw
            self.__dim -= prev_nb_eq
            # self.__cols and self.__rows unchanged
            self.__rhs = None
            if self.__Vs is not None : # Started calculation
                Vs,Vk,Vn = self.__Vs,self.__Vk,self.__Vn
                self.update(Vs,Vk,Vn,usmat=None)
            return
        elif not ( isinstance(material, Fluid) or \
                   isinstance(material, \
                              TransverselyIsotropicElasticSolid) ) :
            msg = "USMultilayerPipe.setExternalSpace :: Error:" + \
                  "\n\tThe material as to be either a fluid or a" + \
                  "\n\ttransversely isotropic elastic medium the" + \
                  "\n\tsymmetry axis of which being the axis of the " + \
                  "pipe."
            return
        new_usmat = USTubularMaterial(material, USset=self)
        if new_usmat is self.__rhs : return # no change
        if self.__rhs is None : # adding External Space
            prev_nb = 0
        else : # changing External Space
            prev_nb = self.__rhs.nb_pw
        new_nb = new_usmat.nb_pw
        dnb = new_nb - prev_nb
        self.__dim += dnb
        # self.__C and self.__R unchanged
        self.__rhs = new_usmat
        if new_usmat not in self.__Lusm :
            self.__Lusm.append(new_usmat)
            mat_to_update = new_usmat
        else : mat_to_update = None
        if self.__Vs is not None : # Started calculation
            Vs,Vk,Vn = self.__Vs, self.__Vk, self.__Vn
            self.update(Vs, Vk, Vn, usmat=mat_to_update )
    #----------------------------------------------------------------------
    def update(self, Vs, Vk, Vn=None, usmat="All", buildM=True) :
        if (self.__Vs is not Vs) or (self.__Vk is not Vk) or \
           (self.__Vn is not Vn) : # detected changes
            usmat="All"
        axisym = (Vn is None)
        if usmat is not None : # one or more US material to update
            self.__Vs = np.array(Vs)
            self.__Vk = np.array(Vk)
            if axisym : # axisymmetrical
                self.__Vn = None
                S2,K = np.meshgrid(self.__Vs**2,self.__Vk,indexing="ij")
                K2 = K**2
                if usmat == "All" :
                    for usm in self.__Lusm : usm.update(S2,K,K2)
                else : usmat.update(S2,K,K2)
            else : # 3D
                self.__Vn = np.array(Vn)
                S2,K,N = np.meshgrid(self.__Vs**2,self.__Vk,self.__Vn,\
                                    indexing="ij")
                K2 = K**2
                if usmat == "All" :
                    for usm in self.__Lusm :
                        usm.update(S2,K,K2,N)
                else : usmat.update(S2,K,K2,N)
            self.__shp = S2.shape + (self.dim,self.dim)
        else : # usmat == None and Vs,Vkx,Vky did not change
            self.__shp = self.__shp[:-2] + (self.dim,self.dim)
        # Global matrix
        if buildM : self.rebuildM(forced=True)
        else : self.clearM()
    #----------------------------------------------------------------------
    def clearSKN(self) :
        """Clears all US parameters. Useful to release memory."""
        self.__Vs = None
        self.__Vk = None
        self.__N = None
        self.__M = None
        for usm in self.__Lusm : usm.clearEta()
    #----------------------------------------------------------------------
    def rebuildM(self,forced=False) :
        """rebuild the M array."""
        prt = self.__prt
        if self.__M is not None and not forced :
            prt("M has already been updated.")
            return
        if (self.__Vs is None) or (self.__Vk is None) :
            prt("M cannot be updated because S or K are not defined.")
            return
        axisym = self.__Vn is None # Axisymmetrical/3D
        self.__M = np.zeros( self.shape, dtype=complex )
        if axisym : # axisymmetrical
            S2,K = np.meshgrid(self.__Vs**2,self.__Vk,indexing="ij")
            N = None
        else : # 3D
            S2,K,N = np.meshgrid(self.__Vs**2,self.__Vk,self.__Vn,\
                                indexing="ij")
##        # For checking parameters
##        cols,rows,d = list(self.n_cols),list(self.n_rows),self.dim
##        print("update -> dim:",d,"; cols:",cols,"; rows:",rows)
        # ++++++ Interface at r = r_min ++++++
        if len(self.layers) == 0 : # Cylinder
            r = self.R[0] 
            FClhs = self.__lhs.field_components(r,r,S2,K,N)
            if self.__rhs is not None :
                FCrhs = self.__rhs.field_components(r,r,S2,K,N)
            if self.__lhs.nb_pw == 3 : # Solid Internal Cylinder
                if self.__rhs is None : # Solid/Vacuum
                    self.__M[...,:,:] = FClhs[...,-3:,-3:]# Sigma_r = 0
                elif self.__rhs.nb_pw == 3 : # Solid/Solid  (6 equations)              
                    self.__M[...,:6,:3] = FClhs[...,-3:]   # Ingoing only
                    self.__M[...,:6,3:] =  -FCrhs[...,:3]  # Outgoing only
                else : # Solid/Fluid (4 equations)             
                    self.__M[...,0,:3] = FClhs[...,0,-3:]     # Ur
                    self.__M[...,1:4,:3] = FClhs[...,-3:,-3:] # Sigma_r
                    self.__M[...,0,-1] = -FCrhs[...,0,0]      # Ur
                    self.__M[...,1,-1] = -FCrhs[...,3,0]      # Sigma_rr
            else : # Fluid, self.__lhs.nb_pw == 1
                if self.__rhs is None : # Fluid/Vacuum
                    self.__M[...,0,0] = FClhs[...,0,1]       # U_r = 0
                elif self.__rhs.nb_pw == 3 : # Fluid/Solid  (4 equations)              
                    self.__M[...,0,0] = FClhs[...,0,1]       # Ur              
                    self.__M[...,1,0] = FClhs[...,3,1]       # Sigma_rr
                    self.__M[...,0,1:] = -FCrhs[...,0,:3]    # Ur
                    self.__M[...,1:4,1:] = -FCrhs[...,-3:,:3]# Sigma_r   
                else : # Fluid/Fluid, first_layer.npw == 2 (2 equations)             
                    self.__M[...,0,0] = FClhs[...,0,1]    # Ur
                    self.__M[...,1,0] = FClhs[...,3,1]    # -P
                    self.__M[...,0,1] = -FCrhs[...,0,0]   # Ur
                    self.__M[...,1,1] = -FCrhs[...,3,0]   # -P
            return
        first_layer = self.layers[0]
        usm = first_layer.usm
        r = first_layer.r_min
        r_ref = 0.5*(r+first_layer.r_max)
        FC = usm.field_components(r,r_ref,S2,K,N)
        if self.__lhs is None : # No Internal Cylinder
            if first_layer.npw == 6 : # Vacuum/Solid  (3 equations)              
                self.__M[...,:3,:6] = -FC[...,3:,:]
            else : # Wall/Fluid, first_layer.npw == 2 (1 equation : Ur=0)                            
                self.__M[...,0,:2] = -FC[...,0,:]
        elif self.__lhs.nb_pw == 3 : # Solid Internal Cylinder
            FClhs = self.__lhs.field_components(r,r,S2,K,N)
            if first_layer.npw == 6 : # Solid/Solid  (6 equations)              
                self.__M[...,:6,:3] =  FClhs[...,-3:]     # Ingoing only
                self.__M[...,:6,3:9] = -FC 
            else : # Solid/Fluid, first_layer.npw == 2 (4 equations)             
                self.__M[...,0,:3] = FClhs[...,0,-3:]     # Ur
                self.__M[...,1:4,:3] = FClhs[...,-3:,-3:] # Sigma_r
                self.__M[...,0,3:5] = -FC[...,0,:]        # Ur
                self.__M[...,1,3:5] = -FC[...,3,:]        # Sigma_rr
        else : # Fluid, self.__lhs.nb_pw == 1 
            FClhs = self.__lhs.field_components(r,r,S2,K,N)
            if first_layer.npw == 6 : # Fluid/Solid  (4 equations)
                self.__M[...,0,0] = FClhs[...,0,1]      # Ur              
                self.__M[...,1,0] = FClhs[...,3,1]      # Sigma_rr
                self.__M[...,0,1:7] = -FC[...,0,:]      # Ur
                self.__M[...,1:4,1:7] = -FC[...,-3:,:]  # Sigma_r   
            else : # Fluid/Fluid, first_layer.npw == 2 (2 equations)
                self.__M[...,0,0] = FClhs[...,0,1]      # Ur
                self.__M[...,1,0] = FClhs[...,3,1]      # -P
                self.__M[...,0,1:3] = -FC[...,0,:]      # Ur
                self.__M[...,1,1:3] = -FC[...,3,:]      # -P
        # ++++++ Interface at r = r_max ++++++
        last_layer = self.layers[-1]
        usm = last_layer.usm
        r = last_layer.r_max
        r_ref = 0.5*(last_layer.r_min+r)
        FC = usm.field_components(r,r_ref,S2,K,N)
        if self.__rhs is None : # No External Space
            if last_layer.npw == 6 : # Solid/Vacuum            
                self.__M[...,-3:,-6:] = FC[...,3:,:]
            else : # Fluid/Wall, last_layer.npw == 2  (Ur=0)
                self.__M[...,-1,-2:] = FC[...,0,:]
        elif self.__rhs.nb_pw == 3 : # Solid
            FCrhs = self.__rhs.field_components(r,r,S2,K,N)
            if last_layer.npw == 6 : # Solid/Solid  (6 equations)
                self.__M[...,-6:,-3:] = -FCrhs[...,:3]
                self.__M[...,-6:,-9:-3] = FC  
            else : # Fluid/Solid, last_layer.npw == 2 (4 equations)
                self.__M[...,-4,-3:] = -FCrhs[...,0,:3]    # Ur           
                self.__M[...,-3:,-3:] = -FCrhs[...,-3:,:3] # Sigma_r 
                self.__M[...,-4,-5:-3] = FC[...,0,:]       # Ur 
                self.__M[...,-3,-5:-3] = FC[...,-1,:]      # Sigma_rr=-P
        else : # Fluid, self.__rhs.nb_pw == 1
            FCrhs = self.__rhs.field_components(r,r,S2,K,N)
            if last_layer.npw == 6 : # Solid/Fluid  (4 equations)
                self.__M[...,-4,-1] = -FCrhs[...,0,0]   # Ur
                self.__M[...,-3,-1] = -FCrhs[...,-1,0]  # Sigma_rr=-P
                self.__M[...,-4,-7:-1] = FC[...,0,:]    # Ur
                self.__M[...,-3:,-7:-1] = FC[...,-3:,:] # Sigma_r  
            else : # Fluid/Fluid, last_layer.npw == 2 (2 equations)
                self.__M[...,-2,-1] = -FCrhs[...,0,0]   # Ur 
                self.__M[...,-1,-1] = -FCrhs[...,-1,0]  # -P 
                self.__M[...,-2,-3:-1] = FC[...,0,:]    # Ur
                self.__M[...,-1,-3:-1] = FC[...,-1,:]   # -P 
        # ++++++ Inner Interfaces ++++++
        in_lay = self.layers[0]
        r = in_lay.r_max
        r0in = 0.5*(in_lay.r_min+r)
        rmin = self.n_rows[0]
        cmin,cmil = self.n_cols[:2]
        for out_lay,cmax,rmax in \
            zip(self.layers[1:],self.n_cols[2:],self.n_rows[1:]) :
            r0out = 0.5*(r+out_lay.r_max)
            FCin = in_lay.usm.field_components(r,r0in,S2,K,N)
            FCout = out_lay.usm.field_components(r,r0out,S2,K,N)
            if in_lay.npw == 6 :
                if out_lay.npw == 6 : # Solid/Solid
                    self.__M[...,rmin:rmax,cmin:cmil] = FCin
                    self.__M[...,rmin:rmax,cmil:cmax] = -FCout
                else : # out_lay.npw == 2, Solid/Fluid
                    self.__M[...,rmin,cmin:cmil] = FCin[...,0,:]    # Ur
                    self.__M[...,rmin+1:rmax,cmin:cmil] = FCin[...,-3:,:]
                    self.__M[...,rmin,cmil:cmax] = -FCout[...,0,:]  # Ur
                    # Fixed 2019 Dec., 08:
                    self.__M[...,rmin+1,cmil:cmax] = \
                                                    -FCout[...,-1,:]# Srr
            else : # in_lay.npw == 2
                if out_lay.npw == 6 : # Fluid/Solid
                    self.__M[...,rmin,cmin:cmil] = FCin[...,0,:]    # Ur
                    # Fixed 2019 Dec., 08:
                    self.__M[...,rmin+1,cmin:cmil] = FCin[...,-1,:] # Srr
                    self.__M[...,rmin,cmil:cmax] = -FCout[...,0,:]  # Ur
                    self.__M[...,rmin+1:rmax,cmil:cmax] = \
                                                    -FCout[...,-3:,:]
                else : # out_lay.npw == 2, Fluid/Fluid
                    self.__M[...,rmin,cmin:cmil] = FCin[...,0,:]    # Ur
                    self.__M[...,rmax-1,cmin:cmil] = FCin[...,-1,:] # -P
                    self.__M[...,rmin,cmil:cmax] = -FCout[...,0,:]  # Ur
                    self.__M[...,rmax-1,cmil:cmax] = -FCout[...,-1,:]#-P
            in_lay = out_lay
            r = in_lay.r_max
            r0in = r0out
            cmin,cmil = cmil,cmax
            rmin = rmax
        return
    #----------------------------------------------------------------------
    def clearM(self) :
        """Clears only the M array. Useful to release memory."""
        self.__M = None
        self.__prt("Warning: be careful to use the 'rebuildM' method" + \
                   " if necessary.")
    #----------------------------------------------------------------------
    def field(self, r, Vc) :
        """Deprecated:
                   use ``fields(r, Vc, "Stroh vector", output="array")''.
        """
        # Compatibility with previous versions
        return fields(self, z, Vc, "Stroh vector", output="array")
    #----------------------------------------------------------------------
    def fields(self, r, Vc, wanted="all", output="list", tol_pos=1e9) :
        """Returns arrays (ns,nk[,na]) of displacements and stresses
           at the vertical position z. Vc is the array of the
           partial waves coefficients. 'wanted' is a string in
           ('all', 'displacements', 'stresses', 'Sr', 'Sa', 'Sz',
            'Stroh vector', 'radial stresses', 'axial stresses',
            'azimuthal stresses'), or a sublist of ('Ur', 'Ua', 'Uz',
            'Srr', 'Saa', 'Szz', 'Saz', 'Srz', 'Sra').
           The output can be a list of arrays (default) or a big array.
        """
        # Possible choices
        choices = ("Ur","Ua","Uz","Srr","Saa","Szz","Saz","Srz","Sra")
        if wanted == "all" :
            wanted = choices
        elif wanted in ("displacements","U") :
            wanted = ("Ur","Ua","Uz")
        elif wanted in ("stresses","sigma") :
            wanted = ("Srr","Saa","Szz","Saz","Srz","Sra")
        elif wanted in ("radial stresses","sigma_r","Sr") :
            wanted = ("Srr","Sra","Srz")
        elif wanted in ("azimuthal stresses","sigma_a","Sa") :
            wanted = ("Sra","Saa","Saz")
        elif wanted in ("axial stresses","sigma_z","Sz") :
            wanted = ("Srz","Saz","Szz")
        elif wanted == "Stroh vector" :
            wanted = ("Ur","Ua","Uz","Srr","Sra","Srz")
        elif isinstance(wanted,str) :
            msg = "USMultilayerPipe :: fields - Error:"
            msg += f"\n\tUnrecognized value '{wanted}' for wanted."
            raise ValueError(msg)
        else :
            for F in wanted :
                if F not in choices :
                    msg = "USMultilayerPipe :: fields - Error:"
                    msg += "\n\tUnrecognized value "+\
                          f"'{F}' in {wanted} for wanted"
                    raise ValueError(msg)
        if (self.__Vs is None) or (self.__Vk is None) :
            msg = "USMultilayerPipe :: fields - Error:"
            msg += "\n\tFields cannot be calculated because S or K"
            msg += "\n\tare not defined."
            raise ValueError(msg)
        axisym = self.__Vn is None # Axisymmetrical/3D
        if axisym : # axisymmetrical
            S2,K = np.meshgrid(self.__Vs**2,self.__Vk,indexing="ij")
            N = None
        else : # 3D
            S2,K,N = np.meshgrid(self.__Vs**2,self.__Vk,self.__Vn,\
                                indexing="ij")
        R,C = self.R,self.__cols # radii and column indexes
        # Tolerancy
        if R[0]-tol_pos <= r < R[0] : r = R[0]
        elif R[-1] < r <= R[-1]+tol_pos : r = R[-1]
        # internal cylinder
        if r < R[0] : 
            usm = self.__lhs
            if usm is None : # Vacuum
                msg =  "USMultilayerPipe :: fields - Error:"
                msg += "\n\tInternal cylinder: "
                msg += "r = {} mm is in vacuum".format(1e3*r)
                raise ValueError(msg)
            npw = usm.nb_pw
            if npw != C[0] :
                msg = "USMultilayerPipe :: fields - Error:"
                msg += "\n\tInternal cylinder: "
                msg += f"indexing -> {npw} != {C[0]}"
                raise ValueError(msg)
            CC = Vc[...,:npw]
            P = usm.field_components(r,R[0],S2,K,N)
            F = np.einsum(usm.MVprod,P[...,npw:],CC) # Ingoing waves only
            # usm is the US material and F the Stroh vector (4 or 6)
        else :
            if len(self.layers) == 0 : # Cylinder
                if r == R[0] :
                    usm = self.__lhs
                    if usm is None : # Vacuum cylinder
                        cherche = True
                    else :
                        npw = usm.nb_pw
                        if npw != C[0] :
                            msg = "USMultilayerPipe :: fields - Error:"
                            msg += "\n\tInternal cylinder: "
                            msg += f"indexing -> {npw} != {C[0]}"
                            raise ValueError(msg)
                        CC = Vc[...,:npw]
                        P = usm.field_components(r,R[0],S2,K,N)
                        F = np.einsum(usm.MVprod,P[...,npw:],CC)
                        cherche = False                    
            else :
                cg,cherche = C[0],True
                for i,(rip1,cd) in enumerate(zip(R[1:],C[1:])) :
                    if r <= rip1 : # r in layer no.i
                        cherche = False
                        lay_i = self.layers[i]
                        usm = lay_i.usm
                        r0 = 0.5*(lay_i.r_min+lay_i.r_max)
                        P = usm.field_components(r,r0,S2,K,N)
                        F = np.einsum(usm.MVprod,P,Vc[...,cg:cd])
                        # usm is the US material and F the Stroh vector
                        # (4 or 6)
                        break
                    cg = cd
            if cherche : # External Space : r > R[-1]
                usm = self.__rhs
                if usm is None : # Vacuum
                    msg =  "USMultilayerPipe :: fields - Error:"
                    msg += "\n\tExternal space: "
                    msg += "r = {} mm is in vacuum".format(1e3*r)
                    raise ValueError(msg)
                npw = usm.nb_pw
                npw_verif = self.dim-C[-1]
                if npw != npw_verif :
                    msg = "USMultilayerPipe :: fields - Error:"
                    msg += "\n\tExternal space: "
                    msg += "indexing -> {} != {}".format(npw,npw_verif)
                    raise ValueError(msg)
                CC = Vc[...,-npw:]
                P = usm.field_components(r,R[-1],S2,K,N)
                F = np.einsum(usm.MVprod,P[...,:npw],CC)
                                # Outgoing waves only
                # usm is the US material and F the Stroh vector (4 or 6)
        vectors = []
        for choice in wanted :
            if choice == "Ur" :
                vectors.append(F[...,0])
            elif choice == "Ua" :
                vectors.append(F[...,1])
            elif choice == "Uz" :
                vectors.append(F[...,2])
            elif choice == "Srr" :
                vectors.append(F[...,3])
            elif choice == "Sra" :
                if isinstance(usm,USTubularFluid) :
                    vectors.append(np.zeros_like(F[...,3]))
                else : vectors.append(F[...,4])
            elif choice == "Srz" :
                if isinstance(usm,USTubularFluid) :
                    vectors.append(np.zeros_like(F[...,3]))
                else : vectors.append(F[...,5])
            elif choice == "Saa" :
                vectors.append( \
                    usm.sigmaThetaTheta(F,r,self.__Vk,self.__Vn))
            elif choice == "Szz" :
                vectors.append( \
                    usm.sigmaZZ(F,r,self.__Vk,self.__Vn))
            elif choice == "Saz" :
                vectors.append( \
                    usm.sigmaThetaZ(F,r,self.__Vk,self.__Vn))         
        if output != "array" : return vectors
        big_array = np.empty( vectors[0].shape + (len(vectors),), \
                              dtype = np.complex128)
        for i,V in enumerate(vectors) :
            big_array[...,i] = V
        return big_array
                    
    #----------------------------------------------------------------------
    def copy(self) :
        """Copy without s,k,n parameters."""
        if len(self.layers) == 0 : # Cylinder
            mat0 = self.internalUSmat.mat_diff_IES
            new_pipe = USMultilayerPipe(0.0,self.R[0],mat0)
        else : # Actual pipe
            first_layer = self.layers[0]
            mat0 = first_layer.usm.mat_diff_IES
            new_pipe = USMultilayerPipe(self.R[0],self.R[1],mat0)
            for layer,r in zip(self.layers[1:],self.R[2:]) :
                mat = layer.usm.mat_diff_IES
                new_pipe.appendLayer(r,mat)
            if self.internalUSmat is not None : # Internal Cylinder
                                                # not vacuum
                mat = self.internalUS.mat_diff_IES
                new_pipe.setInnerCylinder(mat)
        if self.externalUSmat is not None : # External Space not vacuum
            mat = self.externalUSmat.mat_diff_IES
            new_pipe.setExternalSpace(mat)
        return new_pipe
    #----------------------------------------------------------------------
    def __str__(self) :
        msg = self.STARS+"\n"
        dbb = 18*"="
        if len(self.layers) == 0 :
            msg += f"Cylinder of radius {1e3*self.R[0]:.2f} mm, "+\
                   f"with {self.dim} partial waves :"+\
                    "\n----------"
            npw = self.__lhs.nb_pw
            lhs = self.__lhs.mat_diff_IES.name + \
                    " ({} partial ingoing wave{})".format(npw,\
                    "s" if npw > 1 else "")
            msg += "\n Cylinder: {}".format(lhs)
        else :
            if isinstance(self.__lhs, USTubularMat) :
                msg += f"{len(self.layers)+1}-layered cylinder of " + \
                       f"radius {1e3*self.R[-1]:.2f} mm, " + \
                       f"with {self.dim} partial waves :"+\
                        "\n----------"
            else :
                msg += f"{len(self.layers)}-layered pipe of internal " + \
                       f"radius {1e3*self.R[0]:.2f} mm and external " + \
                       f"radius {1e3*self.R[-1]:.2f} mm,\n\twith " + \
                       f"{self.dim} partial waves :"+\
                        "\n----------"
            if self.__lhs is None :
                lhs = self.VACUUM
            elif self.__lhs == "wall" :
                lhs = self.WALL                
            else :
                npw = self.__lhs.nb_pw
                lhs = self.__lhs.mat_diff_IES.name + \
                         f" ({npw} partial ingoing " + \
                          "wave{})".format("s" if npw > 1 else "")
            msg += "\n Internal Cylinder: {}".format(lhs)
            noeq0 = 0
            for n,(noeq1,lay,r) in enumerate(\
                                     zip(self.n_rows,self.layers,self.R)) :
                msg += "\n" + dbb + f" Interface #{n} ({noeq1-noeq0} " + \
                       f"equations) at r = {1e3*r:.2f} mm " + dbb
                noeq0 = noeq1
                msg += "\n "+lay.__str__(n)
            msg += "\n" + dbb + f" Interface #{n+1} ({self.dim-noeq0} " + \
                   f"equations) at r = {1e3*self.R[-1]:.2f} mm " + dbb
        # External space
        if self.__rhs is None :
            rhs = self.VACUUM
        elif self.__rhs == "wall" :
            rhs = self.WALL
        else :
            npw = self.__rhs.nb_pw
            rhs = self.__rhs.mat_diff_IES.name + \
                    f" ({npw} partial outgoing " + \
                     "wave{})".format("s" if npw > 1 else "")
        msg += "\n External Space: {}".format(rhs)
        msg += "\n" + self.STARS + "\n"
        return msg
    #----------------------------------------------------------------------
    def write_in_file(self, file_path=None) :
        """Pipe parameters saved in a text file.
           If file_path is None, the text is simply returned."""
        # Material list 
        materials = []
        # Layers
        layer_data = []
        for lay in self.layers :
            w_mm = 1e3*lay.h
            material = lay.usm.mat_diff_IES
            if material not in materials :
                materials.append(material)
            w = f"Width: {w_mm:.3f} mm"
            mat = f"Material: {material.name}"
            layer_data.append( [w,mat] )
        # Inner Cylinder
        r_str = f"radius {1e3*self.R[0]:.3f} mm"
        if self.__lhs is None :
            inner_cylinder = (r_str, self.VACUUM)
        elif self.__lhs == "wall" :
            inner_cylinder = (r_str, self.WALL)
        else :
            material = self.__lhs.mat_diff_IES
            if material not in materials :
                materials.append(material)
            inner_cylinder = (r_str, material.name)
        # External space
        r_str = f"radius {1e3*self.R[-1]:.3f} mm"
        if self.__rhs is None :
            external_space = (r_str, self.VACUUM)
        elif self.__rhs == "wall" :
            external_space = (r_str, self.WALL)
        else :
            material = self.__rhs.mat_diff_IES
            if material not in materials :
                materials.append(material)
            external_space = (r_str, material.name)
        # Exported text
        text = self.export_to_text(layer_data, materials, \
                                   inner_cylinder, external_space)
        if file_path is None :
            return text        
        try :
            with open(file_path, "w", encoding="utf8") as strm :
                strm.write(text)
        except Exception as err :
            msg = f"USMultilayerPipe.write_in_file\n'{err}'"
            raise ValueError(msg)
    #----------------------------------------------------------------------
    @staticmethod
    def export_to_text(layer_data, materials, InCyl, ExtSp) :  
        """ Export data of a multilayer pipe to text.
            layer_data: pairs of string
                       ["Width: {value} mm", "Material: {name}"]
            materials: list of materials
            InCyl (inner cylinder):
                  pair [ "radius {value} mm",
                          name, "[ Vacuum ]" or "[ Wall ]" ]
            ExtSp (external space): idem
        """
        sline = USMultilayerPipe.SLINE
        dline = USMultilayerPipe.DLINE
        text = dline
        nb_lay, nb_mat = len(layer_data), len(materials)
        r_str, incylmat = InCyl
        # ``Pipe'' : empty inner cylinder or rigid wall
        # ``Cylinder'' : inner cylinder with an actual material
        if incylmat in USMultilayerPipe.HS_CONDITIONS :
            text += f"\nPipe with {nb_lay} layer"
            if nb_lay > 1 : text += "s"
        else :
            text += f"\nCylinder with {nb_lay+1} layer"
            if nb_lay > 0 : text += "s"
        text += f" and {nb_mat} material"
        if nb_mat>1 : text += "s"
        text += "\n" + dline + "\n"
        text += f"Material of inner cylinder of {r_str}: " + incylmat
        text += "\n" + sline + "\n"
        for i,(w,m) in enumerate(layer_data,1) :
            text += f"Layer {i}:\n\t{w}\n\t{m}"
            text += "\n" + sline + "\n"
        r_str, extmat = ExtSp
        text += f"External {r_str}\n"
        text += "Material of external space: " + extmat
        text += "\n" + dline + "\n"
        if len(materials) > 0 :
            for mat in materials[:-1] :
                text += mat.tosave() + sline + "\n"
            text += materials[-1].tosave() + dline + "\n"
        return text
    #----------------------------------------------------------------------
    @staticmethod
    def import_elements_from_text(text, raised_errors=True) :  
        """ Import data from a multilayer pipe to text.
            returns (layer_data, materials, InCyl, ExtSp)   
            layer_data: pairs [ w_m (float), material ]
            materials: list of materials
            InCyl (inner cylinder):
                  pair [ "radius {value} mm",
                          name, VACUUM, WALL or UNDEFINED ]
            ExtSp (external space): idem
        """
        error_msg = "USMultilayerPipe.import_elements_from_text :: error:"
        VACUUM = USMultilayerPipe.VACUUM
        WALL = USMultilayerPipe.WALL
        HS_CONDITIONS = [VACUUM, WALL]
        UNDEFINED = USMultilayerPipe.UNDEFINED
        INCYL = USMultilayerPipe.INCYL
        EXTSP = USMultilayerPipe.EXTSP
        # Warning: note the .lower()
        rows = [ r.strip().lower() for r in text.split("\n") ]
        layer_data, materials = [],[]
        # Number of layers and materials
        searching = True
        nb_layer, nb_material = None, None
        for r,row in enumerate(rows) :
            if "pipe with" in row or "cylinder with" in row :
                searching = False
                words = row.split()
                for i,w in enumerate(words,-1) :
                    if w.startswith("layer") :
                        nb_layer = int(words[i])
                        if "cylinder with" in row :
                            nb_layer -= 1 # Inner cylinder not counted
                    if w.startswith("material") :
                        nb_material = int(words[i])
                break
        if searching or nb_layer is None or nb_material is None :
            msg = error_msg + "\n\t'pipe/cylinder with x layer(s) " + \
                              "and y material(s)' not found."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        # Inner Cylinder, external space and material names
        r_next = r+1
        idx_beg, idx_end, idx_mat = None, None, []
        for r,row in enumerate(rows[r_next:],r_next) :
            if "inner cylinder" in row:
                idx_beg = r
            elif "external space" in row:
                idx_end = r
            elif "name" in row :
                idx_mat.append(r)
        idx_mat.append(r+1)
        if idx_beg is None or idx_end is None or idx_beg >= idx_end :
            msg = error_msg + \
                  "\n\t'unable to identify the geometry area."            
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        if len(idx_mat) != nb_material + 1 :
            msg = error_msg + "\n\t'number of material and material " \
                              "areas don't match."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        # Material Dictionary
        materials = dict()
        for beg,end in zip( idx_mat[:-1], idx_mat[1:] ) :
            txt = "\n".join(rows[beg:end])
            try :
                mat = ImportMaterialFromText(txt)
                materials[mat.name] = mat
            except Exception as err :
                msg = error_msg + f"\n\tMaterial definition\n\t{err}."
                if raised_errors : raise ValueError(msg)
                return False, msg, None, None
        # Inner cylinder
        row = rows[idx_beg]
        try :
            b,e = row.index("radius"), row.index(":")
            r_str_in = row[b:e]
        except Exception as err :
            msg = error_msg + \
                  f"\n\t'radius' or ':' not found in '{row}'\n\t{err}."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None 
        try :
            val,unit = r_str_in.split()[-2:]
            val = float(val)
        except Exception as err :
            msg = error_msg + \
                  f"\n\t[{INCYL}] {r_str_in}\n\t{err}."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        if unit == "mm" :
            r_in_m = 1e-3*val
        elif unit == "m" :
            r_in_m = val
        else :
            msg = error_msg + f"\n\t[{INCYL}] {r_str_in}\n\t" + \
                  f"unknown unit '{unit}'."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None           
        if "undefined" in row or "unknown" in row :
            InCyl = UNDEFINED
        else :
            try :
                InCyl = row.split(":")[1].strip().title()
            except Exception as err :
                msg = error_msg + f"\n\t{INCYL}\n\t{err}."
                if raised_errors : raise ValueError(msg)
                return False, msg, None, None
        if InCyl in HS_CONDITIONS :
            pass
        elif InCyl in materials.keys() :
            InCyl = materials[InCyl]
        else :
            msg = error_msg + f"\n\t{INCYL}: unknown " + \
                              f"'{InCyl}' material."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Layers
        r_next, num_lay = idx_beg+1, 0
        for r,row in enumerate(rows[r_next:idx_end],r_next) :
            if row.startswith("layer") :
                num_lay += 1
                try :
                    number = int(row.split(":")[0].split()[1])
                except Exception as err :
                    msg = error_msg + f"\n\t[{r}] {row}\n\t{err}."
                    if raised_errors : raise ValueError(msg)
                    return False, msg, None, None
                if number != num_lay :
                    msg = error_msg + f"\n\t[{r}] {row}\n\t" + \
                          f"Layer Number should be {num_lay}."
                    if raised_errors : raise ValueError(msg)
                    return False, msg, None, None
                w_row,m_row = rows[r+1:r+3]
                # Width
                if "undefined" in w_row or "unknown" in w_row :
                    w_m = None
                else :
                    try :
                        val,unit = w_row.split()[-2:]
                        val = float(val)
                    except Exception as err :
                        msg = error_msg + \
                              f"\n\t[{r+1}]{w_row}\n\t{err}."
                        if raised_errors : raise ValueError(msg)
                        return False, msg, None, None
                    if unit == "mm" :
                        w_m = 1e-3*val
                    elif unit == "m" :
                        w_m = val
                    else :
                        msg = error_msg + f"\n\t[{r+1}]{w_row}\n\t" + \
                              f"unknown unit '{unit}'."
                        if raised_errors : raise ValueError(msg)
                        return False, msg, None, None
                # Material
                if "undefined" in m_row or "unknown" in m_row :
                    mat = None
                else :
                    try :
                        mat_name = m_row.split(":")[-1].strip().title()
                    except Exception as err :
                        msg = error_msg + \
                              f"\n\t[{r+2}]{m_row}\n\t{err}."  
                        if raised_errors : raise ValueError(msg)
                        return False, msg, None, None
                    if mat_name in materials.keys() :
                        mat = materials[mat_name]
                    else :
                        msg = error_msg + f"\n\t[{r+1}]{m_row}\n\t" + \
                              f"unknown material '{mat_name}'."
                        if raised_errors : raise ValueError(msg)
                        return False, msg, None, None
                # Updating of layer_data:
                layer_data.append( [w_m, mat] )
        if num_lay != nb_layer : 
            msg = f"\n\t{num_lay} layers found instead of {nb_layer}."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        # External space
        row = rows[idx_end-1]
        try :
            b = row.index("radius")
            r_str_out = row[b:]
        except Exception as err :
            msg = error_msg + \
                  f"\n\t'radius' not found in '{row}'\n\t{err}."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None 
        try :
            val,unit = r_str_out.split()[-2:]
            val = float(val)
        except Exception as err :
            msg = error_msg + \
                  f"\n\t[{INCYL}] {r_str_out}\n\t{err}."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        if unit == "mm" :
            r_out_m = 1e-3*val
        elif unit == "m" :
            r_out_m = val
        else :
            msg = error_msg + f"\n\t[{INCYL}] {r_str_out}\n\t" + \
                  f"unknown unit '{unit}'."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None               
        row = rows[idx_end]
        if "undefined" in row or "unknown" in row :
            ExtSp = None
        else :
            try :
                ExtSp = row.split(":")[1].strip().title()
            except Exception as err :
                msg = error_msg + f"\n\t{EXTSP}\n\t{err}."
                if raised_errors : raise ValueError(msg)
                return False, msg, None, None
        if ExtSp in HS_CONDITIONS :
            pass
        elif ExtSp in materials.keys() :
            ExtSp = materials[ExtSp]
        else :
            msg = error_msg + f"\n\t{EXTSP}: unknown '{ExtSp}' material."
            if raised_errors : raise ValueError(msg)
            return False, msg, None, None
        return layer_data, materials, (r_in_m,InCyl), (r_out_m,ExtSp)
    #----------------------------------------------------------------------
    @staticmethod
    def import_from_file(file_path) :  
        """ Returns a USMultilayerPipe instance from a text file.
        """    
        error_msg = f"File '{file_path}'\ndoes not seem to " + \
                     "correspond to a multilayer pipe:"
        pos_enc = ("utf8","cp1252")
        for enc in pos_enc :
            try :
                with open(file_path, "r", encoding=enc) as strm :
                    text = strm.read()
                ok = True
                break
            except :
                ok = False
        if not ok :
            msg = error_msg + f"\n\tencoding error: not in {pos_enc}."
            raise ValueError(msg)
        return USMultilayerPipe.import_from_text( text )
    #----------------------------------------------------------------------
    @staticmethod
    def import_from_text(text) :  
        """ Returns a USMultilayerPipe instance from text.
        """       
        VACUUM = USMultilayerPipe.VACUUM
        WALL   = USMultilayerPipe.WALL
        # Read data from text
        layer_data, material, (r_in,InCyl), (r_out,ExtSp) = \
                    USMultilayerPipe.import_elements_from_text(text)
        if len(layer_data) == 0 : # A simple cylinder
            new_pipe = USMultilayerPipe(0.0, r_in, InCyl)
            if ExtSp == VACUUM :
                pass
            elif ExtSp == WALL :
                new_pipe.setExternalSpace("wall")
            else :
                new_pipe.setExternalSpace(ExtSp)
            return new_pipe
        # Actual pipe
        w_m, mat = layer_data[0]
        cr = r_in + w_m
        new_pipe = USMultilayerPipe(r_in, cr, mat)
        # additional layer(s)
        for w_m, mat in layer_data[1:] :
            cr += w_m
            new_pipe.appendLayer(cr, mat)
        # Inner Cylinder
        if InCyl == VACUUM :
            pass
        elif InCyl == WALL :
            new_pipe.setInnerCylinder("wall")
        else :
            new_pipe.setInnerCylinder(InCyl)
        # External space
        if ExtSp == VACUUM :
            pass
        elif ExtSp == WALL :
            new_pipe.setExternalSpace("wall")
        else :
            new_pipe.setExternalSpace(ExtSp)
        return new_pipe
#========= Tests ==========================================================
if __name__ == "__main__" :
    # Test of the function USMultilayerStructure
    test1 = USMultilayerStructure( "Data/Plates/test_23-06-09.txt" )
    print( test1.write_in_file() )
    test2 = USMultilayerStructure( "Data/Pipes/test_23-07-05.txt" )
    print( test2.write_in_file() )
    # Example
    np.set_printoptions( precision=2 )
    water = Fluid( {"rho":1000.0, "c":1500.0}, "Water")
    crbepx = TransverselyIsotropicElasticSolid({"rho":1560.0,\
                   "c11":1.4e10, "c12": 8.0e9, "c33": 8.7e10,\
                   "c13": 9.0e9, "c44": 4.7e9}, "Carbon-Epoxy")
    r_min,r_max = 4e-3,5e-3
    single_layer = True
    if single_layer :
        pipe1 = USMultilayerPipe( r_min, r_max, crbepx)
    else : # two layers for testing
        r_int = 0.3*r_min + 0.7*r_max
        pipe1 = USMultilayerPipe( r_min, r_max, crbepx)
        pipe1.appendLayer(r_max, crbepx)
    pipe1.setInnerCylinder(water)
    print(pipe1)
    file_pth = "Data/Pipes/test_23-07-06.txt"
    pipe1.write_in_file(file_pth)
    pipe2 = USMultilayerPipe.import_from_file( file_pth )
    print(pipe1.write_in_file()==pipe2.write_in_file())
    #-----------------------------------------
    # k and s parameters
    test_KS = True
    if test_KS :
        from TimeGridClass import TimeGrid
        tg = TimeGrid(8.0e-5, 0.2e-6)
        from SpaceGridClasses import Space1DGrid
        sg = Space1DGrid(200, 1e-3)
        n = 1 # Azimuthal wavenumber
        pipe1.update(tg.S-0.95*tg.gamma,sg.K,[n])
    # Comparison with modes
    Comparison = True
    if test_KS and Comparison :
        from Modes_Monolayer_Pipe import *
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        K_ordon = sg.sort2cent(sg.K)
        modes = ModesMonolayerPipe(crbepx, r_min, r_max, K_ordon, 50, n=n)
        fig = plt.figure("Dispersion curves",figsize=(15,7.5))
        ax1, ax2 = fig.subplots(1,2)
        fig.subplots_adjust(0.05, 0.08, 0.95, 0.95, 0.2)
        Tab = sg.sort2cent(\
                    -np.log10(abs(np.linalg.det(pipe1.M[:,:,0])) \
                              ).transpose())
        Tab /= pipe1.dim
        maxT = Tab.max()
        ims = []
        ranges = (-0.5*1e-6*tg.df, 1e-6*(tg.F[-1]+0.5*tg.df), \
                  1e-3*(sg.k_min-0.5*sg.dk), 1e-3*(sg.k_max+0.5*sg.dk))
        for ax in [ax1, ax2] :
            ims.append( ax.imshow( Tab-maxT, vmax=0, vmin=-3.0, \
                                   aspect="auto", cmap="gray", \
                                   interpolation="none", origin="lower", \
                                   extent=ranges ) )
            ax.set_xlabel(r"Frequency $f$ [MHz]", fontsize=14)
            ax.set_ylabel(r"Axial wavenumber $k_z$ [$\mathrm{mm}^{-1}$]", \
                          fontsize=14)
            ax.set_xlim(0, 1e-6*tg.F[-1])
        fig.suptitle( f"$n = {n}$", fontsize=14)
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', \
                  '#8c564b', '#e377c2', '#bcbd22', '#17becf', '#888800']
        for no,clr in enumerate( colors ):
            ax2.plot( 1e-6*modes.F[:,no], 1e-3*modes.K, "-", color=clr)
        ax2.grid()
        dvd = make_axes_locatable(ax2)
        cax = dvd.append_axes('right', size='2%', pad=0.06)
        plt.colorbar(ims[1], cax=cax, extend="min")
        plt.show()
