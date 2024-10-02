# Version 1.12 - 2024, October, 2nd
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import os, sys, psutil, pickle
import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
#++++ TraFiC location ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Relative path to TraFiC code:
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import TraFiC_init
from USMultilayerPlate import USMultilayerPlate
from USMultilayerPipe import USMultilayerPipe
from ComputationParameters import ComputationParameters
from TraFiC_utilities import now
from time import time
#=========================================================================
class FieldComputationInProgress :
    """Management of field computation, with saved data files.
       comp_params_or_path: ComputationParameters instance or file path
       structure: USMultilayerPlate or USMultilayerPipe instance
                  (with ComputationParameters instance only).
    """
    #---------------------------------------------------------------------
    # root of TraFiC is one level upper than the current file
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Available RAM :
    MAXMEM = 16.0 # Fixed maximum
    GIGA = 1073741824
    AVAILABLERAM = round(0.5*psutil.virtual_memory().free/GIGA,2)
                       # 50% of the virtual available memory
    MAXMEM = min(MAXMEM, AVAILABLERAM)
    FNAME = "Computation_in_progress.txt"
    LINESIZE = USMultilayerPlate.LINESIZE
    STAGES = ["Creation",
              "Definition of the computation parameters",
              "Definition of the structure",
              "Computation of the Green tensors",
              "Ready for field computations for given sources" ]
    INAMES = {0: "Wall/Fluid", 1: "Vacuum/Fluid" , 2: "Fluid/Fluid",
              3: "Vacuum/Solid", 4: "Fluid/Solid", 6: "Solid/Solid"}
    PLATEDS = {0: (("Uz",),), 1: (("Fz","Szz"),),
               2: (("Uz",), ("Fz","Szz")),
               3: (("Fx","Sxz"), ("Fy","Syz"), ("Fz","Szz")),
               4: (("Uz",), ("Fx","Sxz"), ("Fy","Syz"), ("Fz","Szz")),
               6: (("Ux",), ("Uy",), ("Uz",), ("Fx","Sxz"),
                   ("Fy","Syz"), ("Fz","Szz")) }
    PIPEDS  = {0: (("Ur",),), 1: (("Fr","Srr"),),
               2: (("Ur",), ("Fr","Srr")),
               3: (("Fr","Srr"), ("Fa","Sra"), ("Fz","Srz")),
               4: (("Ur",), ("Fr","Srr"), ("Fa","Sra"), ("Fz","Srz")),
               6: (("Ur",), ("Ua",), ("Uz",), ("Fr","Srr"),
                   ("Fa","Sra"), ("Fz","Srz")) }
    #---------------------------------------------------------------------
    def __init__(self, comp_params_or_path, structure=None,
                 cylindrically_symmetric=False, verbose=False) :
        msg = "FieldComputationInProgress constructor :: "
        # Verbose or not
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        prt = self.__prt
        if isinstance(comp_params_or_path, str) : # File path
            self.update_from_file( comp_params_or_path )
        else : # New computation
            # Computation parameters
            comput_params = comp_params_or_path
            if not isinstance( comput_params, ComputationParameters ) :
                msg += ("Error:\n\tThe 1st parameter is not a "
                        + "'ComputationParameters'\n\tinstance.")
                raise ValueError(msg)
            self.__CP = comput_params
            # Multilayer structure
            if isinstance( structure, USMultilayerPlate ) :
                self.__case = "Plate"
                self.__DS = self.PLATEDS
            elif isinstance( structure, USMultilayerPipe ) :
                self.__case = "Pipe"
                self.__DS = self.PIPEDS
            else :
                msg += ("Error:\n\tThe 2nd parameter is neither a "
                        + "'USMultilayerPlate' nor a"
                        + "\n\t'USMultilayerPipe' instance.")
                raise ValueError(msg)
            self.__MS = structure # Format for US structure
            self.__fmt_struc_file_path = (
                os.path.join(self.Green_tensor_path,
                             self.__case + "_{}.pckl"))
            # Head file exists?
            if os.path.isfile( self.head_file_path ) :
                msg += (f"Warning:\n\tComputation '{self.label}' "
                        + "already started.")
                prt( msg )
                self.update_from_file( ) # Update with check
            else :
                self.__stage = 2 # Ready for Green tensor computation
                self.__creation = now(True)
                self.__update_text()
                self.__update_partition() # For computation partitioning
                self.save()
                self.__FC_dict = None # Existing field computations
    #---------------------------------------------------------------------
    def __update_text(self, additional_text1="", additional_text2="" ) :
        # Stage #0: creation 
        self.__text = self.stage_text(0)
        self.__text +=  self.__creation + "\n" 
        # Stage #1: computer parameters
        self.__text += self.stage_text(1)
        self.__text += self.__CP.head_text  
        # Stage #2: multilayer structure
        self.__text += self.stage_text(2)
        self.__text += self.__MS.write_in_file()
        if self.__stage == 2 : return
        # Stage #3: started Green tensor computation
        self.__text += self.stage_text(3)
        self.__text += additional_text1 + "\n"
        # Stage #4: finished Green tensor computation
        if self.__stage == 4 :
            self.__text += self.stage_text(4)
        if additional_text2 != "" :
            self.__text += additional_text2 + "\n"            
    #---------------------------------------------------------------------
    def __update_partition(self, stored_parts=None) :
        """For computation partitioning. Update attributes:
               unit_needed_size: size for one value of s and k
               parts: list of different parts
               part_case : (part. type, pc1, pc2)
                          "No. part." : None, None
                          "Part. on s": (number of cases, s step), None
                          "Part. on x": nb_frac, (nbx, x step)
                          "Part. on y": nb_frac, (nby, y step)
                          nb_frac is the number of cases for each s value.
                          number of cases = number_of_s * nb_frac.
        """
        msg = "FieldComputationInProgress.__update_partition :: "
        prt = self.__prt
        # Estimation of the needed memory for each s value and
        # each wavenumber value
        # Matrix M + coefficients + polarizations + wavenumbers
        nbc = self.__MS.dim # Dimension: number of coefficients,
                            # size of the square matrix M
        nbpw = self.__MS.nbPartialWaves # Number of partial waves
                                        # Less or equal to nbc
                                        # number of polarizations
                                        # and normal wavevectors
        self.__unit_needed_size = 16*( nbc + nbc*nbc + nbpw*7 )
        max_mem = self.MAXMEM
        nb_max = round( np.floor( max_mem * self.GIGA
                                  / self.__unit_needed_size ) )
        prt(f"{msg}\nnb_max: {nb_max}")
        # Maximum number of cases for each record
        ns = self.__CP.ns # Number of complex Laplace values s
        nbds = round( np.floor( np.log10(ns-1) ) ) + 1
        fmts = "{:0" + f"{nbds:d}" + "d}"
        nx = self.__CP.nx # Number of wavenumbers in the first
                          # space direction
        if self.__CP.is_3d :
            nx *= self.__CP.ny
        nbs_max = nb_max // nx
        prt("nbs_max:", nbs_max)
        if nbs_max == 0 : # nb_frac*ns computations
            if self.__CP.is_3d :
                nbx_max = nb_max // self.__CP.ny
                if nbx_max == 0 : # self.__CP.ny too high
                                  # should not happen!
                    # Partioning on y
                    nb_frac_ny = round(np.ceil(self.__CP.ny/nb_max))
                    nb_frac = self.__CP.nx*nb_frac_ny
                    self.__part_case = ("Part. on y", nb_frac,
                                        (nb_frac_ny, nb_max) )
                    slices = [ (slice(js,js+1), slice(jy,jy+nb_max),
                                slice(jx,jx+1))
                               for js in range(ns)
                               for jx in range(self.__CP.nx)
                               for jy in range(0, self.__CP.ny, nb_max)
                               ]
                else :
                    # Partioning on x
                    nb_frac = round(np.ceil(self.__CP.nx/nbx_max))
                    self.__part_case = ("Part. on x", nb_frac,
                                        (nb_frac, nbx_max) )
                    slices = [ (slice(js,js+1), slice(None),
                                slice(jx,jx+nbx_max))
                               for js in range(ns)
                               for jx in range(0, self.__CP.nx, nbx_max)
                               ]
            else : # 2d
                nb_frac = round(np.ceil(nx/nb_max))
                self.__part_case = ("Part. on x", nb_frac,
                                    (nb_frac, nb_max) )
                slices = [ (slice(js,js+1), slice(jx,jx+nb_max))
                           for js in range(ns)
                           for jx in range(0, nx, nb_max)
                           ]
            msg += (f"Warning:\n\tNot enough memory for a single s "
                    + f"value.\n\tMust be divided into {nb_frac} parts.")
            prt(msg)
            parts = []
            nbd = round( np.floor( np.log10(nb_frac) ) ) + 1
            fmt = "_{:0" + f"{nbd:d}" + "d}on" + f"{nb_frac}"
            subparts = [ fmt.format(i) for i in range(1,nb_frac+1) ]
            for n in range(ns) :
                str_s = fmts.format(n)
                parts.extend( [ str_s+s for s in subparts ] )
        elif nbs_max == 1 : # ns computations
            parts = [ fmts.format(n) for n in range(ns) ]
            self.__part_case = ("Part. on s", (ns, 1), None )
            if self.__CP.is_3d :
                slices = [ (slice(js,js+1), slice(None), slice(None))
                           for js in range(ns) ]
            else : # 2d
                slices = [ (slice(js,js+1), slice(None))
                           for js in range(ns) ]
        elif ns <= nbs_max : # A single computation
            parts = [ fmts.format(0) + "to" + fmts.format(ns-1) ]
            self.__part_case = ("No part.", None, None )
            if self.__CP.is_3d :
                slices = [ (slice(None), slice(None), slice(None)) ]
            else : # 2d
                slices = [ (slice(None), slice(None)) ]
        else :
            nb_parts = round( np.ceil(ns/nbs_max) )
            nbsm1 = nbs_max - 1
            parts = [ fmts.format(i) + "to" + fmts.format(i+nbsm1)
                      for i in range(0,ns-nbs_max,nbs_max) ]
            b = len(parts) * nbs_max
            parts.append(fmts.format(b) + "to" + fmts.format(ns-1))
            self.__part_case = ("Part. on s", (nb_parts, nbs_max), None )
            if self.__CP.is_3d :
                slices = [ (slice(js,js+nbs_max), slice(None),
                            slice(None)) for js in range(0, ns, nbs_max) ]
            else : # 2d
                slices = [ (slice(js,js+nbs_max), slice(None))
                           for js in range(0, ns, nbs_max) ]
        self.__parts = parts
        if stored_parts is  not None :
            if parts != stored_parts :
                msg += ("Error: the stored partition:\n\t\t"
                        + f"{stored_parts}\n\tdiffers from the computed"
                        + f" one:\n\t\t{parts}.\n\tPlease remove old "
                        + "files before computation.")
                raise ValueError(msg)
        self.__slices = slices
        if len(self.__parts) < 7 :
            prt(f"self.__parts:", self.__parts)
            prt(f"self.__slices:", self.__slices)
        else :
            prt(f"self.__parts: {len(self.__parts)} elements\n"
                + f"{self.__parts[:3]}\n   [...]\n{self.__parts[-3:]}")
            prt(f"self.__slices: {len(self.__slices)} elements\n"
                + f"{self.__slices[:3]}\n   [...]\n{self.__slices[-3:]}")
    #---------------------------------------------------------------------
    def save( self ) :        
        with open( self.head_file_path, "w", encoding="utf8" ) as strm :
            strm.write( self.__text )
    #---------------------------------------------------------------------
    def compute_Green_tensors( self, possible_excitations ) :
        """To start or continue the computation of Green tensors.
             possible_excitations is a list of pairs ( S, I ) where :
                > S is in {"Ux"/"Ur", "Uy"/"Ua", "Uz",
                           "Fx"="Sxz"/"Fr"="Srr",
                           "Fy"="Syz"/"Fa"="Sra",
                           "Fz"="Szz"/"Fz"="Srz" }
                > I is the number of the interface where a jump on S is
                  imposed.
                > (S, I) can be simply replaced by S for the interface
                  either between the first layer and the upper half-space
                  in a plate (I=0) or between the last layer and the
                  external space in a pipe (I=-1)
                > Wall/Fluid : "Uz"/"Ur" only
                  Vacuum/Fluid : "Fz"/"Fr" only                  
                  Vacuum/Solid : "Fx"/"Fr", "Fy"/"Fa", "Fz" only
                  Fluid/Fluid : "Fz"/"Fr", "Uz"/"Ur" only
                  Fluid/Solid : "Uz"/"Ur", "Fx"/"Fr", "Fy"/"Fa", "Fz" only
                  Solid/Solid : all are possible
        """
        msg = "FieldComputationInProgress.computeGreenTensors :: "
        prt = self.__prt
        prt(f"{msg}\nself.__stage: {self.__stage}")
        if self.__stage == 2 : # New Green tensor computation
            self.__stage = 3
            text_stage = self.STAGES[3]
            prt(f"+++++ Stage 3: {text_stage}")
            self.__text += self.stage_text(3)
            # Excitation positions and characteristics
            Imax = len(self.__MS.layers)
            if self.__case == "Plate" :
                default_I = 0 # Interface at z=0
                L = USMultilayerPlate.THS # Top half-space
                R = USMultilayerPlate.BHS # Top half-space
                matL = self.__MS.topUSmat # Top half-space
                matR = self.__MS.bottomUSmat # Bottom half-space
            else : # self.__case == "Pipe"
                default_I = Imax # External interface
                R = USMultilayerPipe.INCYL # Inner Cylinder
                L = USMultilayerPipe.EXTSP # External space
                matL = self.__MS.internalUSmat # External space
                matR = self.__MS.externalUSmat # Inner Cylinder
            nr0 = (0,) + self.__MS.n_rows
            nr1 = self.__MS.n_rows + (self.__MS.dim,)
            new_text = "Excitations:\n"
            for S_I in possible_excitations :
                try :
                    S,I = S_I
                    S,I = str(S), int(I)
                except : # Default interface
                    S = S_I
                    I = default_I
                r0,r1 = nr0[I], nr1[I]
                nbv = r1-r0
                if nbv == 1 : # Wall/Fluid or Vacuum/Wall
                    if I == 0 :
                        if matL is not None : nbv = 0 # Wall
                    else : # I == Imax
                        if matR is not None : nbv = 0 # Wall
                rank = None
                for r,LS in enumerate(self.__DS[nbv], r0) :
                    if S in LS :
                        rank = r
                        break
                if rank is None : # Error
                    msg += (f"Error:\n\t'{S}' is not in "
                            + f"{self.__DS[nbv]} ({self.INAMES[nbv]} "
                            + "interface).")
                    raise ValueError(msg)
                new_text += (f"\t'{S}' in interface #{I}: "
                             + f"rank {rank} in [{r0}, {r1-1}] "
                             + f"({self.INAMES[nbv]}).\n")               
            new_text += f"Maximum memory: {self.MAXMEM:.2f} GB\n"
            nbp = len(self.__parts)
            new_text += f"{nbp} case"
            if nbp == 1 :
                new_text += f" to compute:\n\t{self.__parts[0]} - to do\n"
            else : # nbp > 1
                new_text += "s to compute:\n\t"
                new_text += " - to do\n\t".join(self.__parts)
                new_text += " - to do\n"
            # Updating self.__text
            if nbp < 7 :
                prt(new_text)
            else :
                rows = new_text.split("\n")
                prt("\n".join(rows[:-nbp+4])
                    + "\n   [...]\n"
                    + "\n".join(rows[-3:]))
            self.__text += new_text
            self.save()
        self.pick_up_Green_tensor_computation()
    #---------------------------------------------------------------------
    def pick_up_Green_tensor_computation( self ) :
        msg = ("FieldComputationInProgress.pick_up_Green_tensor_"
               + "computation\n:: Error:\n\t")
        prt = self.__prt
        if self.__stage == 4 :
            prt("FieldComputationInProgress.pick_up_Green_tensor_"
                + "computation\n:: Already done!")
            return
        prt("Beginning of FieldComputationInProgress."
            + "pick_up_Green_tensor_computation\n"
            + f"self.__stage: {self.__stage}")
        # Updating parameters from self.__text
        rows = self.__text.split("\n")
        # Excitations
        # private attributes exc_numbers and fmt_coef_file_pathes
        nbp, rbp = self.__update_exc_from_text(rows)
        if nbp < 7 :
            prt("rows:\n" + "\n".join( [f"{no:3d} {r}" for no,r in
                                        enumerate(rows)]))
        else :
            prt("rows:\n" + "\n".join( [f"{no:3d} {r}" for no,r in
                                        enumerate(rows[:-nbp+4])])
                + "\n   [...]\n"
                + "\n".join( [f"{no:3d} {r}" for no,r in
                              enumerate(rows[-3:], len(rows)-3)]))
        prt(f"nbp {nbp}, rbp {rbp}\nself.__fmt_coef_file_pathes"
            +f" '{self.__fmt_coef_file_pathes}'")
        exc_numbers = self.__exc_numbers
        # Successive cases
        parts_info = []
        for part,row in zip(self.__parts,rows[rbp:rbp+nbp]) :
            rpart,state = row.strip().split(" - ")
            if rpart != part :
                msg += f"Error:\n\t'{rpart}' != '{part}'"
                raise ValueError(msg)
            parts_info.append( state )
        if nbp < 7 : prt(f"parts_info ({len(parts_info)} elements):"
                         + f"\n{parts_info}")
        else : prt(f"parts_info ({len(parts_info)} elements):\n"
                   + f"{parts_info[:-nbp+4]}\n"
                   + f"   [...]\n{parts_info[-3:]}")
        # Partitioning the vectors Vs and Vk
        # Complex Laplace values
        ns = self.__CP.ns
        time_gd = self.__CP.time_grid
        Vs = time_gd.S
        # Wavevectors
        space_gd = self.__CP.space_grid
        nx = self.__CP.nx # Number of wavenumbers in the first
                          # space direction
        pt, pt1, pt2 = self.__part_case
        if self.__CP.is_3d :
            ny = self.__CP.ny
            Vkx = space_gd.Kx
            Vky = space_gd.Ky
            prt(f"Vkx: {Vkx[:3]}\n        [...]\n     {Vkx[-3:]}\n"
                + f"Vky: {Vky[:3]}\n        [...]\n     {Vky[-3:]}")
            if pt == "No part." : # a single computation
                LSK = [[Vs, Vkx, Vky]]
            elif pt == "Part. on s" :
                nbs, ds = pt1
                LSK = [ [Vs[i:i+ds], Vkx, Vky] for i in range(0,ns,ds) ]                  
            elif pt == "Part. on x" :
                msg += f"Partitioning on x not yet implemented."
                raise ValueError(msg)
            elif pt == "Part. on y" :
                msg += f"Partitioning on y not yet implemented."
                raise ValueError(msg)
        else : # 2d
            Vkx = space_gd.K
            if pt == "No part." : # a single computation
                LSK = [[Vs, Vkx]]
            elif pt == "Part. on s" :
                nbs, ds = pt1
                LSK = [ [Vs[i:i+ds], Vkx] for i in range(0,ns,ds) ]       
            elif pt == "Part. on x" :                
                msg += f"Partitioning on x not yet implemented."
                raise ValueError(msg)
        if len(LSK) != len(self.__parts) :
            msg += f"{len(LSK)} [len(LSK)] != {nbs} [len(self.__parts)]"
            raise ValueError(msg)
        # for the last value of s
        last_part = [ False for _ in self.__parts]
        last_part[-1] = True
        # Computation 
        fmt_struc_file_path = self.__fmt_struc_file_path
        for r,(state, part, SK, lp) in enumerate(
                zip(parts_info, self.__parts, LSK, last_part), rbp):
            done = False
            struct_path = fmt_struc_file_path.format(part)
            struct_file_exists = os.path.isfile(struct_path)
            exc_paths = [ fmt.format(part) for fmt
                          in self.__fmt_coef_file_pathes ]
            exc_paths_exist = [ os.path.isfile(p) for p in exc_paths ]
            if "to do" not in state :
                if struct_file_exists and np.all(exc_paths_exist) :
                    prt(part, state)
                    done = True # already done
                else :
                    wrn = msg.replace("Error","Warning")
                    wrn += "The following files don't exist:"
                    if not struct_file_exists :
                        wrn += f"\n\t\t> {os.path.basename(struct_path)}"
                    for e,p in zip(exc_paths_exist,exc_paths) :
                        if not e : wrn += f"\n\t\t> {os.path.basename(p)}"
                    prt(wrn)
            if done : continue
            # Not finished
            prt(part, "starts at", now(),"...")
            beg = time()
            if struct_file_exists :
                with open(struct_path, "br") as strm :
                    cur_struct = pickle.load(strm)
                ok = (cur_struct.write_in_file()
                      == self.__MS.write_in_file())
                if ok :
                    dur = time()-beg
                    prt(f'\t[{dur:.1f}"]'
                        + f"'{os.path.basename(struct_path)}' loaded.")
            if not struct_file_exists or not ok :
                dur = time()-beg
                prt(f'\t[{dur:.1f}"] Loading the s and k values...')
                cur_struct = self.__MS
                cur_struct.update(*SK, buildM=False)
                with open(struct_path, "bw") as strm :
                    pickle.dump(cur_struct, strm)
                dur = time()-beg
                prt(f'\t[{dur:.1f}"]'
                    + f"'{os.path.basename(struct_path)}' saved.")
            dur = time()-beg
            prt(f'\t[{dur:.1f}"] Building the M matrix...')
            cur_struct.rebuildM( forced=True )
            dur = time()-beg
            prt(f'\t[{dur:.1f}"] ...done')
            # Unit source USRC
            if self.is_3d :
                Vs, Vkx, Vky = SK
                SRC_S = np.ones_like(Vs)
                if lp : SRC_S[-1] = 0.0 # s = s_max
                SRC_Kx = np.ones_like(Vkx)
                dif = np.abs(Vkx-self.__CP.space_grid.kx_max) 
                idx = np.argmin(dif)
                if dif[idx] <  0.1 * self.__CP.space_grid.dkx :
                    SRC_Kx[idx] = 0.0 # kx = kx_max
                SRC_Ky = np.ones_like(Vky)
                dif = np.abs(Vky-self.__CP.space_grid.ky_max) 
                idx = np.argmin(dif)
                if dif[idx] <  0.1 * self.__CP.space_grid.dky :
                    SRC_Ky[idx] = 0.0 # ky = ky_max
                SRC_K = np.einsum("i,j->ij", SRC_Kx, SRC_Ky)
                USRC = np.einsum("i,jk->ijk",SRC_S,SRC_K) 
            else :
                Vs, Vk = SK
                SRC_S = np.ones_like(Vs)
                if lp : SRC_S[-1] = 0.0 # s = s_max
                SRC_K = np.ones_like(Vk)
                dif = np.abs(Vk-self.__CP.space_grid.k_max) 
                idx = np.argmin(dif)
                if dif[idx] <  0.1 * self.__CP.space_grid.dk :
                    SRC_K[idx] = 0.0 # k = k_max
                USRC = np.einsum("i,j->ij",SRC_S,SRC_K)
            for done, path, (_,_,rank) in zip(
                    exc_paths_exist, exc_paths, exc_numbers) :
                if done :
                    dur = time()-beg
                    print(f'\t[{dur:.1f}"] '
                          + f"'{os.path.basename(path)}' already done.")
                    continue
                dur = time()-beg
                print(f'\t[{dur:.1f}"] '
                      + f"Computing '{os.path.basename(path)}'...")
                B = np.zeros( (cur_struct.M.shape)[:-1] ,
                              dtype = np.complex128 )
                B[..., rank] = USRC # rank characterizes the nature and
                                    # the position of the excitation
                C = solve( cur_struct.M, B ) # Coefficients to be saved.
                np.save(path, C)
                dur = time()-beg
                print(f'\t[{dur:.1f}"] ...done')
            # End of excitations
            cur_struct.clearM()
            old_row = rows[r]
            new_row = f"\t{part} - done [{now(True)}]"
            prt(old_row, "=>", new_row)
            rows[r] = new_row
            self.__text = "\n".join(rows)
            self.save()
        # All computations are done
        self.__stage = 4
        self.__text += "\n" + self.stage_text(4)
        self.save()
        prt("End of FieldComputationInProgress."
            + "pick_up_Green_tensor_computation")
    #---------------------------------------------------------------------
    def __update_exc_from_text( self, rows ) :
        """Updates the private attributes exc_numbers and
           fmt_coef_file_pathes.
           Returns the number of possible excitations and the index
           of the first row for fractionning the computation."""
        # row format : "'{S}' in interface #{I}: rank {rank} in ..." 
        # Cases to compute and path formats
        exc_numbers = []
        nbp, rbp = None, None
        for r,row in enumerate(rows) :
            if "in interface" in row and "rank" in row :
                S,_,_,I,_,rank = row.split()[:6]
                S = S.replace("'","")
                I = int( I.replace("#","").replace(":","") )
                rank = int(rank)
                exc_numbers.append( (S,I,rank) )
            if "case" in row and "to compute:" in row :
                nbp = int(row.split()[0])
                rbp = r+1
                break
        self.__exc_numbers = tuple( exc_numbers )
        self.prt("exc_numbers:", self.__exc_numbers)
        self.__fmt_coef_file_pathes = []
        nbdi = round( np.floor( np.log10(len(self.__MS.layers)) ) ) + 1
        fmti = "_interface{:0" + f"{nbdi}" + "d}.npy"
        for S,I,_ in exc_numbers :
            self.__fmt_coef_file_pathes.append(
                os.path.join(self.Green_tensor_path,
                             "Coef_{}_" + f"{S}" + fmti.format(I)) )
        self.__fmt_coef_file_pathes = tuple(self.__fmt_coef_file_pathes)
        return nbp, rbp
    #---------------------------------------------------------------------
    def update_from_file(self, file_path=None ) :
        """Updates the FieldComputationInProgress for file_path.
           The default value of file_path is the attribute head_file_path.
        """
        msg = "FieldComputationInProgress.update_from_file :: "
        prt = self.__prt
        prt("Beginning of FieldComputationInProgress.update_from_file")
        if  file_path is None :
            file_path = self.head_file_path
            old_CP = self.__CP
            old_MS = self.__MS
            check = True
        else :
            check = False
        # ++++++++++ Reading text file +++++++++++++++++++++++++++++++++++
        with open( file_path, "r", encoding="utf8" ) as strm :
            rows = [ r.strip() for r in strm ]
        # stages: list of stage numbers [0,1,2,...]
        # blocks: texts for different stages
        stages, blocks, bs, cs = [], [], False, False
        for n, r in enumerate(rows) :
            if "STAGE " in r :
                if bs or cs :
                    msg += "Error:\n\tUnexpected new stage"
                    raise ValueError(msg)
                wds = r.split()
                no = int( wds[ wds.index("STAGE")+1 ] )
                stages.append(no)
                bs = True
            if r.startswith("|||") :
                if bs :
                    nb = n+1 # First row of the new block
                    bs = False
                    cs = True
                elif cs :
                    blocks.append( "\n".join(rows[nb:n]) )
                    cs = False
        if cs : # Last block
            if len(blocks) < 4 :         
                blocks.append( "\n".join(rows[nb:]) )
            else : # tabulations for field computations
                blocks.append( "\n\t".join(rows[nb:]) )
            cs = False
        stage = stages[-1]
        if not np.allclose( stages, np.arange(stage+1) ) :
            msg += f"Error:\n\tstages is {stages}"
            raise ValueError(msg)
        # ++++++++++ Initializations +++++++++++++++++++++++++++++++++++++
        self.__stage = stage
        self.__creation = blocks[0].strip()
        self.__CP = ComputationParameters.from_head_text(blocks[1])
        if check :
            if old_CP.head_text != self.__CP.head_text :
                msg += ("Error:\n\tThe new and old computation "
                        + "parameters differ.\n\tPlease check the "
                        + "existing file and delete it for a new "
                        + "computation.")
                raise ValueError(msg)
        if "plate" in blocks[2].lower() :
            self.__case = "Plate"
            self.__MS = USMultilayerPlate.import_from_text(blocks[2])
            self.__DS = self.PLATEDS
        else : # "pipe" in blocks[2].lower() :
            self.__case = "Pipe"
            self.__MS = USMultilayerPipe.import_from_text(blocks[2])
            self.__DS = self.PIPEDS
        self.__fmt_struc_file_path = os.path.join(
                self.Green_tensor_path,
                self.__case + "_{}.pckl")        
        if check :
            if old_MS.write_in_file() != self.__MS.write_in_file() :
                msg += ("Error:\n\tThe new and old computation "
                        + "parameters differ.\n\tPlease check the "
                        + "existing file and delete it for a new "
                        + "computation.")
                raise ValueError(msg)
        # ++++ For computation partitioning ++++++++++++++++++++++++++++++
        stored_parts = None
        if len(blocks) > 3 : # New in version >= 1.12
                             # (fixed bug on memory)
            memory_GB = None
            stored_parts = []
            for row in blocks[3].split("\n") :
                if row.startswith("Maximum memory:") :
                    memory_GB = float(row.split()[2])
                    prt(f"Stored memory: {memory_GB:.2f} GB")
                    if not np.isclose(self.MAXMEM, memory_GB):
                        if memory_GB > self.AVAILABLERAM :
                            msg += ("Warning:\n\tThe saved memory "
                                    + f"{memory_GB:.2f} GB is greater than"
                                    + " the available memory "
                                    + f"{self.AVAILABLERAM} GB!")
                            print(msg)
                        prt("The maximum memory has been replaced by"
                            + " the saved value:\n\t"
                            + f"{self.MAXMEM:.2f} GB -> {memory_GB:.2f}"
                            + " GB")
                        self.MAXMEM = memory_GB
                elif " - done" in row :
                    stored_parts.append(row.split(" - done")[0])
            prt("stored_parts:", stored_parts)
        self.__update_partition(stored_parts)
        # +++++++++ Started Green tensor computation? ++++++++++++++++++++
        prt("stage :", self.__stage)
        if self.__stage == 2 :
            prt("The Green tensor computation is not started.")
            self.__update_text( )
        else :
            if self.__stage == 3 :
                self.__update_text( blocks[3] )
                prt("The Green tensor computation is in progress...")
                self.pick_up_Green_tensor_computation()
            else : # self.__stage == 4
                prt("The Green tensor computation is finished.\n"
                    + "Ready for field computations.")
                rows3 = [ r.strip() for r in blocks[3].split("\n") ]
                self.__update_exc_from_text( rows3 )
                if len(blocks) == 4 : # No Field computation
                    self.__FC_dict = None
                    self.__update_text( blocks[3] )
                else : # len(blocks) == 5
                    txt = blocks[4]
                    rows = txt.split("\n")
                    trouve = False
                    for r,row in enumerate(rows) :
                        if "List of the field computations" in row :
                            rb = r+1
                            trouve = True
                            break
                    if trouve :
                        self.__FC_dict = dict()
                        self.__update_text( blocks[3], txt )
                        labels = [ r.strip() for r in rows[rb:] ]
                        labels = [ r for r in labels if len(r)>0 ]
                        for lbl in labels :
                            self.__FC_dict[lbl] = FieldComputation(
                                                        self,
                                                        lbl)
                    else : # Should not happend
                        self.__FC_dict = None
                        self.__update_text( blocks[3] )
        prt("End of FieldComputationInProgress.update_from_file")
    #---------------------------------------------------------------------
    def prt(self, *args) :
        return self.__prt(*args)
    @property
    def head_file_path(self) :
        return os.path.join( self.__CP.result_path,
                             FieldComputationInProgress.FNAME )
    @property
    def label(self) :
        return self.__CP.label
    @property
    def result_path(self) :
        return self.__CP.result_path
    @property
    def Green_tensor_path(self) :
        return self.__CP.Green_tensor_path
    @property
    def field_path(self) :
        return self.__CP.field_path
    @property
    def creation_date_time(self) :
        return self.__creation
    @property
    def log(self) :
        return self.__text
    @property
    def history(self) :
        return self.__text
    @property
    def structure(self) :
        return self.__MS
    @property
    def structure_type(self) :
        return self.__case
    @property
    def time_grid(self) :
        return self.__CP.time_grid
    @property
    def space_grid(self) :
        return self.__CP.space_grid
    @property
    def is_3d(self) :
        return self.__CP.is_3d
    @property
    def case(self) :
        return self.__case
    @property
    def stage(self) :
        if self.__stage == 2 :
            return "Stage: computation of Green tensors not started."
        if self.__stage == 3 :
            return "Stage: computation of Green tensors not finished."
        if self.__stage == 4 :
            return ("Stage: computation of Green tensors finished.\n"
                    + "       Ready for field computation(s)")
    @property
    def possible_excitations(self) :
        msg = "Possible excitations: "
        if self.__stage < 4 :
            return msg+"none. Green tensors are not all computed."        
        rows = [ r.strip() for r in self.__text.split("\n")
                 if "in interface" in r ]
        rows = [ f"\n\t{i:2d} > {r}" for i,r in enumerate(rows) ]
        return msg + "".join(rows)
    @property
    def fmt_struc_file_path(self) :
        return self.__fmt_struc_file_path
    @property
    def fmt_coef_file_pathes(self) :
        if self.__stage < 4 :
            self.prt("FieldComputationInProgress.fmt_coef_file_pathes "
                     + "::\n\tWarning: Green tensors are not all "
                     + "computed.")
            return tuple()
        return tuple( self.__fmt_coef_file_pathes )
    @property
    def slices(self) :
        return tuple( self.__slices )
    @property
    def parts(self) :
        return tuple( self.__parts )
    @property
    def excitation_numbers(self) :
        if self.__stage < 4 :
            self.prt("FieldComputationInProgress.excitation_numbers "
                     + "::\n\tWarning: Green tensors are not all "
                     + "computed.")
            return tuple()
        return tuple( self.__exc_numbers )
    #---------------------------------------------------------------------
    @staticmethod
    def stage_text(number) :
        lsz = FieldComputationInProgress.LINESIZE
        text = FieldComputationInProgress.STAGES[number]
        bars = "|"*lsz + "\n"
        szmx = lsz - 4
        txt = f"STAGE {number} - {text}"
        csz = len(txt)
        if csz <= szmx :
            spd = (lsz-csz)//2
            return bars + spd*" " + txt + "\n" + bars
        else :
            rows = []
            words = text.split()
            cr = f"STAGE {number} -"
            lcr = len(cr)
            bks = lcr*" "
            szmx -= lcr
            csz,csz0 = 0,0
            for wd in words :
                ln = len(wd)+1
                csz1 = csz0 + ln
                if csz1 <= szmx :
                    cr += " " + wd
                    csz0 = csz1
                else :
                    if csz0 > csz : csz = csz0
                    rows.append(cr)
                    cr = bks + " " + wd
                    csz0 = ln
            if csz0 > 0 : 
                if csz0 > csz : csz = csz0
                rows.append(cr)
            spd = (lsz-lcr-csz)//2
            txt = ("\n" + spd*" ").join(rows)
            return bars + spd*" " + txt + "\n" + bars
    #---------------------------------------------------------------------
    # FIELD COMPUTATION    
    #---------------------------------------------------------------------
    def setFieldComputation(self, label, excitations) :
        """Init a new field computation of label 'label'
           'excitations' contains triplet(s) (no, ranges, array)
            > no is the number displayed by the 'possible_excitations'
              method;
            > ranges (it_min, it_max, [i_ymin, iy_max,] ix_min, ix_max )
            > array is a ndarray of floats of shape
                     (it_max-it_min, [iy_max-iy_min,] ix_max-ix_min )
        """
        msg = "FieldComputationInProgress.setFieldComputation :: "
        if self.__FC_dict is None :
            self.__FC_dict = dict()
            self.__text += f"List of the field computations:\n"
        elif label in self.__FC_dict.keys() :
            msg += f"Aborted: '{label}' already exists."
            self.__prt(msg)
            return
        self.__text += f"\t{label}\n"
        self.__FC_dict[label] = FieldComputation(self, label, excitations)
        self.save()
    #---------------------------------------------------------------------
    def __call__(self, label) :
        """Access to field computation 'label'."""
        try :
            FC = self.__FC_dict[label]
        except Exception as err :
            msg = ("FieldComputationInProgress :: field computation "
                   + f"'{label}' does not exist.\n\t'{err}'")
            self.__prt(msg)
            FC = None
        return FC 
    #---------------------------------------------------------------------
    @property
    def list_of_excitation_labels(self) :
        if self.__FC_dict is None :
            self.prt("No field computation. Sorry!")
            return tuple()
        L = list( self.__FC_dict.keys() )
        L.sort()
        return tuple(L)
#=========================================================================
class FieldComputation :
    """Field computation of label 'label', managed by 'parent', an 
       instance of FieldComputationInProgress. 
       'excitations' contains triplet(s) (no, ranges, array)
            > no is the number displayed by 'parent.possible_excitations'
            > ranges (it_min, it_max,  [i_ymin, iy_max,] ix_min, ix_max)
            > array is a ndarray of floats of shape
                     (it_max-it_min, [iy_max-iy_min,] ix_max-ix_min)
       If 'excitations' is not given, the FieldComputation is assumed to
       be defined in an existing head file.
    """
    PLATE_FIELD_FMT = "{}_at_z_{:07.3f}_mm.npy"
    PIPE_FIELD_FMT = "{}_at_r_{:07.3f}_mm.npy"
    FNAME = "excitations_parameters.txt"
    PLT_OPT = {"size":14, "family":"Arial", "weight":"bold"}
    #---------------------------------------------------------------------
    def __init__(self, parent, label, excitations=None) :
        msg = "FieldComputation constructor :: Error\n\t"
        self.__FCIP = parent
        self.__label = str(label)
        self.__path = os.path.join(parent.field_path, label)
        if not os.path.isdir( self.__path ) :
            os.mkdir( self.__path )
            self.prt(msg.replace(" Error","")
                     + f"'{self.__path}' created.")
        self.__head_path = os.path.join(self.__path, self.FNAME)
        if parent.case == "Plate" :
            self.__fmt = os.path.join(self.__path, self.PLATE_FIELD_FMT)
        else : # parent.case == "Pipe"
            self.__fmt = os.path.join(self.__path, self.PIPE_FIELD_FMT)
        if excitations is None : # read from file
            if not os.path.isfile(self.__head_path) :
                msg += (f"'{self.FNAME}' does not exist. Cannot create"
                        + "new instance.")
                raise ValueError(msg)
            with open(self.__head_path, "r", encoding="utf8") as strm :
                head_text = strm.read()
            self.__update_from_text( head_text )
        else :
            self.__slices = []
            self.__tab_pth = []
            self.__coef_fmt = []
            for rk, ranges, tab_exc in excitations :
                try :
                    coef_fmt = parent.fmt_coef_file_pathes[rk]
                    S,I,_ = parent.excitation_numbers[rk]
                except Exception as err :
                    msg += f"rank {rk} incorrect.\n\t'{err}'"
                    raise ValueError(msg)
                try :
                    if parent.is_3d :
                        bt,et,by,ey,bx,ex = ranges
                        slc = (slice(bt,et), slice(by,ey), slice(bx,ex))
                        # Note that y is before x.
                    else : # 2d
                        bt,et,bx,ex = ranges
                        slc = (slice(bt,et), slice(bx,ex))
                except Exception as err :
                    msg += f"ranges {ranges} incorrect.\n\t'{err}'"
                    raise ValueError(msg)
                try :
                    shp1 = tab_exc.shape
                    shp2 = tuple( sl.stop-sl.start for sl in slc )
                    assert shp1 == shp2
                except Exception as err :
                    msg += f"excitation array incorrect.\n\t'{err}'"
                    raise ValueError(msg)
                tab_pth = os.path.join(self.__path,
                                       f"{S}_interface{I}_excitation.npy")
                np.save(tab_pth, tab_exc) # array saved on disk
                self.__slices.append( slc )
                self.__tab_pth.append( tab_pth )
                self.__coef_fmt.append( coef_fmt )
            self.save()
    #---------------------------------------------------------------------
    def compute(self, fields_to_compute, normal_positions_mm,
                max_memory_GB="auto") :
        """Compute the fields of labels in 'fields_to_compute' at normal
           positions (z for plates and r for pipes) in
           'normal_positions_mm'. The results are saved in .npy files.
        """
        msg = "FieldComputation.compute :: Error\n\t"
        # Fields already computed?
        fields_to_compute = np.array(fields_to_compute, dtype=str)
        normal_positions_mm = np.array(normal_positions_mm, dtype=float)
        nb_fld = len(fields_to_compute)
        nb_pos = len(normal_positions_mm)
        E = np.ones( (nb_fld, nb_pos), dtype=bool)
        for jf,fld in enumerate(fields_to_compute) :
            for jp,pos in enumerate(normal_positions_mm) :
                pth = self.__fmt.format(fld, pos)
                if os.path.isfile( pth ) :
                    self.prt(f"Case '{fld}' at position {pos:.2f} mm "
                             + "already computed")
                    E[jf,jp] = False
        fields_to_compute = fields_to_compute[ np.where(E.any(axis=1)) ]
        if len(fields_to_compute) == 0 :
            self.prt(msg.replace("Error\n\t",
                                 "All cases are already computed."))
            return
        normal_positions_mm = normal_positions_mm[
                                                np.where(E.any(axis=0)) ]
        # Time and space grids
        tm_gd = self.__FCIP.time_grid
        sp_gd = self.__FCIP.space_grid
        # Memory
        mem_by_field = 16 * tm_gd.nt * sp_gd.nx / self.__FCIP.GIGA
        if self.__FCIP.is_3d : mem_by_field *= sp_gd.ny
        if max_memory_GB == "auto" :
            max_memory_GB = self.__FCIP.MAXMEM
        else :
            max_memory_GB = min(max_memory_GB, self.__FCIP.MAXMEM)
            
        if mem_by_field > max_memory_GB :
            msg += (f"Needed memory by field ~ {mem_by_field:.2f} GB."
                    + "\n\tToo big. Not yet implemented")
            raise ValueError(msg)
        fields_to_compute = tuple(fields_to_compute)
        nb_fld, nb_pos = len(fields_to_compute), len(normal_positions_mm)
        max_cases = round( max_memory_GB // mem_by_field )
        nb_cases = nb_fld * nb_pos
        self.prt(f"{nb_cases} / {max_cases} cases possible.")
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        if nb_cases > max_cases : # Computation must be divided...
            nb_frac = round( np.ceil( nb_cases / max_cases ) )
            if nb_frac <= nb_pos : # ...with respect to position only
                # because nb_cases / max_cases <= nb_pos
                #         nb_fld*mem_by_field <= max_memory_GB
                # Number of normal positions by computation:
                rzstp = round( max_memory_GB // (mem_by_field*nb_fld) )
                for nrz in range(0, nb_pos, rzstp ) :
                    pos = normal_positions_mm[ nrz : nrz+rzstp ]
                    self.compute(fields_to_compute, pos)
            else : # ...with respect to fields for each position
                for no_fld in range(0, nb_fld, max_cases) :
                    ftc = fields_to_compute[ no_fld : no_fld+max_cases ]
                    for nrz in range(nb_pos) :
                        pos = [ normal_positions_mm[nrz] ]
                        self.compute(ftc, pos)
            return # Finished                    
        #xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
        BA_KS = np.ndarray( (tm_gd.ns,) + sp_gd.shape + (nb_fld, nb_pos),
                            dtype=np.complex128 )
        self.prt(f"Size of the big array {BA_KS.shape} ~ "
                 + f"{BA_KS.nbytes/self.__FCIP.GIGA:.3f} GB")
        # Name for each element of the partition
        Lprt = self.__FCIP.parts
        # Slices for each element of the partition
        Lslc = self.__FCIP.slices
        # Structure path
        struct_fmt = self.__FCIP.fmt_struc_file_path
        idx = struct_fmt.index("Green_")-1       
        #*****************************************************************
        if self.__FCIP.is_3d :
            #msg += "3d case not yet properly implemented. Sorry!"
            #raise ValueError(msg)
            ###################################################
            ############### NOT YET TESTED ####################
            ###################################################
            beg = time()
            self.prt(f"[{now()}] Excitation in the KS-domain...")
            L_tab_exc = []
            # Remind the y index before the x index
            tab_etx = np.zeros( (tm_gd.nt, sp_gd.ny, sp_gd.nx),
                                dtype=np.float64 )
            for slc, tab_pth in zip(self.__slices, self.__tab_pth) :
                # Excitation in the time-space domain
                etx = tab_etx.copy()
                etx[slc] = np.load(tab_pth)
                etx = sp_gd.cent2sort(etx, axes=(1,2))
                # Excitation in the Laplace-space domain
                esx = tm_gd.LT(etx)
                del etx
                esx[-1,...] = 0.0 # last value must be zero
                # Excitation in the Laplace-wavenumber domain
                esk = sp_gd.fft(esx, axes=(1,2))
                del esx
                esk[:,:,sp_gd.nx_max] = 0.0 # Value at kxmax is zero
                esk[:,sp_gd.ny_max,:] = 0.0 # Value at kymax is zero
                L_tab_exc.append(esk)
            dur = time()-beg
            self.prt(f'[{dur:.1f}"] ...done.\n'
                     + f"[{now()}] Beginning of the field computation "
                     + "by part...") 
            for slc, part in zip(Lslc, Lprt) :
                self.prt(f"[{now()}] part '{part}'")
                beg = time()
                esk, co_pth = L_tab_exc[0] , self.__coef_fmt[0]
                co_pth = co_pth.format(part)
                self.prt(f"\tReading '...{co_pth[idx:]}'")
                # Warning: in C, (it, ix, iy)
                C = np.einsum("ikj,ijkl->ijkl", esk[slc],
                              np.load(co_pth) )
                for esk, co_pth in zip(
                        L_tab_exc[1:], self.__coef_fmt[1:]):
                    co_pth = co_pth.format(part)
                    C += np.einsum("ikj,ijkl->ijkl", esk[slc],
                                   np.load(co_pth) )
                dur = time()-beg
                struct_pth = struct_fmt.format(part)
                self.prt(f'[{dur:.1f}"] ...coefficients done.\n'
                         + f"\tReading '...{struct_pth[idx:]}'...")
                with open(struct_pth, "br") as strm :
                    my_struct = pickle.load(strm)
                dur = time()-beg
                self.prt(f'[{dur:.1f}"] ...done.')
                for iz, zr_mm in enumerate(normal_positions_mm) :
                    self.prt(f"\t{fields_to_compute} at z/r ="
                             + f" {zr_mm:.3f} mm...")
                    zr_obs = 1e-3*zr_mm
                    cslc = slc + (slice(None), iz )
                    BA_KS[cslc] = my_struct.fields(
                                            zr_obs, C,
                                            wanted=fields_to_compute,
                                            output = "array"
                                            ).swapaxes(1,2)
                    dur = time()-beg
                    self.prt(f'[{dur:.1f}"] ...done.')
                del my_struct, C
            self.prt(f"{now()} - Coming back to (x,s)-Domain...")
            # (k,s)-Domain -> (x,s)-Domain
            beg = time()
            BA_XS = sp_gd.ifft(BA_KS, axes=(1,2))
            BA_XS = sp_gd.sort2cent(BA_XS, axes=(1,2)) # Centering space
                                                       # values        
        #*****************************************************************
        else : # 2d
            beg = time()
            self.prt(f"[{now()}] Excitation in the KS-domain...")
            L_tab_exc = []
            tab_etx = np.zeros( (tm_gd.nt, sp_gd.nx), dtype=np.float64 )
            for slc, tab_pth in zip(self.__slices, self.__tab_pth) :
                # Excitation in the time-space domain
                etx = tab_etx.copy()
                etx[slc] = np.load(tab_pth)
                etx = sp_gd.cent2sort(etx, axis=1)
                esx = tm_gd.LT(etx)
                del etx
                esx[-1,...] = 0.0 # last value must be zero
                # Excitation in the Laplace-wavenumber domain
                esk = sp_gd.fft(esx, axis=1)
                del esx
                esk[:,sp_gd.n_max] = 0.0 # Value at kmax is zero
                L_tab_exc.append(esk)
            dur = time()-beg
            self.prt(f'[{dur:.1f}"] ...done.\n'
                     + f"[{now()}] Beginning of the field computation "
                     + "by part...") 
            for slc, part in zip(Lslc, Lprt) :
                self.prt(f"[{now()}] part '{part}'")
                beg = time()
                esk, co_pth = L_tab_exc[0] , self.__coef_fmt[0]
                co_pth = co_pth.format(part)
                self.prt(f"\tReading '...{co_pth[idx:]}'")
                C = np.einsum("ij,ijk->ijk", esk[slc], np.load(co_pth) )
                for esk, co_pth in zip(
                        L_tab_exc[1:] ,self.__coef_fmt[1:]) :
                    co_pth = co_pth.format(part)
                    C += np.einsum("ij,ijk->ijk", esk[slc],
                                   np.load(co_pth) )
                dur = time()-beg
                struct_pth = struct_fmt.format(part)
                self.prt(f'[{dur:.1f}"] ...coefficients done.\n'
                         + f"\tReading '...{struct_pth[idx:]}'...")
                with open(struct_pth, "br") as strm :
                    my_struct = pickle.load(strm)
                dur = time()-beg
                self.prt(f'[{dur:.1f}"] ...done.')
                for iz, zr_mm in enumerate(normal_positions_mm) :
                    self.prt(f"\t{fields_to_compute} at z/r ="
                             + f" {zr_mm:.3f} mm...")
                    zr_obs = 1e-3*zr_mm
                    cslc = slc + (slice(None), iz )
                    BA_KS[cslc] = my_struct.fields(
                                            zr_obs, C,
                                            wanted=fields_to_compute,
                                            output = "array")
                    dur = time()-beg
                    self.prt(f'[{dur:.1f}"] ...done.')
                del my_struct, C
            self.prt(f"{now()} - Coming back to (x,s)-Domain...")
            # (k,s)-Domain -> (x,s)-Domain
            beg = time()
            BA_XS = sp_gd.ifft(BA_KS, axis=1)
            BA_XS = sp_gd.sort2cent(BA_XS, axis=1) # Centering space
                                                   # values
        # 3d and 2d
        del BA_KS
        dur = time()-beg
        self.prt(f'[{dur:.1f}"] ...done. '
                 + 'Coming back to (x,t)-Domain...')
        # (x,s)-Domain -> (x,t)-Domain
        BA_XT = tm_gd.iLT(BA_XS, axis=0)
        del BA_XS
        dur = time()-beg
        self.prt(f'[{dur:.1f}"] ...done. Saving files...')
        for jf, fld in enumerate( fields_to_compute ) :
            for jp, pos in enumerate( normal_positions_mm ) :
                pth = self.__fmt.format(fld, pos)
                np.save(pth, BA_XT[..., jf, jp])
                dur = time()-beg
                self.prt(f'[{dur:.1f}"] ...{pth[idx:]} saved')
        del BA_XT
    #---------------------------------------------------------------------
    def save( self ) :
        msg = "FieldComputation.save :: Error\n\t"
        nb = len(self.__tab_pth)
        text = f"Excitation with {nb} component"
        if nb >=2 : text += "s"
        text += ":"
        for slc, tp, cf in zip(self.__slices, self.__tab_pth,
                               self.__coef_fmt) :
            idx =  " ".join( [ f"{sl.start}:{sl.stop}" for sl in slc ] )
            text += "\n\t" + "\n\t\t".join([os.path.basename(cf),
                                            os.path.basename(tp),
                                            idx])
        text += "\n"
        with open(self.__head_path, "w", encoding="utf8") as strm :
            strm.write(text)
    #---------------------------------------------------------------------
    def __update_from_text( self, head_text ) :
        msg = "FieldComputation.__update_from_text :: Error\n\t" 
        rows = [ r.strip() for r in head_text.split("\n") ]
        try :
            nb = int( rows[0].split()[2] )
        except Exception as err:
            msg += f"Cannot extract the number of components\n\t'{err}'"
            raise ValueError(msg)
        self.__slices = []
        self.__tab_pth = []
        self.__coef_fmt = []
        for rk in 1+3*np.arange(nb) :
            try : r1,r2,r3 = rows[rk:rk+3]
            except Exception as err:
                msg += f"Cannot read component #{rk//3+1}\n\t'{err}'"
                raise ValueError(msg)
            r1 = os.path.join(self.__FCIP.Green_tensor_path, r1)
            r2 = os.path.join(self.__FCIP.field_path, self.__label, r2)
            r3 = [ s.split(":") for s in r3.split() ]
            r3 = [ slice(int(b),int(e)) for b,e in r3 ]
            b = r1.index("Green_tensors")-1
            self.prt("\n\t + ".join( [ msg.replace(
                                            " Error\n\t",
                                            f"\n\tComponent #{rk//3+1}"),
                                       "..."+r1[b:], "..."+r2[b:],
                                       f"{r3}" ] ) )
            self.__slices.append( tuple(r3) )
            self.__tab_pth.append(r2)
            self.__coef_fmt.append(r1)
    #---------------------------------------------------------------------
    def prt(self, *args) : # verbose or not transmitted by the parent
        return self.__FCIP.prt(*args) 
    #---------------------------------------------------------------------
    @property
    def field_path(self) : return self.__path
    @property
    def name(self) : return self.__label
    @property
    def label(self) : return self.__label
    @property
    def computed_fields(self) :
        """Returns the .npy file names corresponding to computed fields.
        """
        L = [ nm for nm in os.listdir(self.__path)
              if "_mm.npy" in nm and "_at" in nm ]
        return L
    @property
    def computed_cases(self) :
        """Returns the pairs (field name, normal position in mm)
           corresponding to computed fields.
        """
        C = [ nm.replace("_mm.npy", "").replace("_at_r_", " ").replace(
                         "_at_z_", " ").split()
              for nm in self.computed_fields ]
        return tuple( (F,float(rz)) for F,rz in C )
    #---------------------------------------------------------------------
    def plot_excitations(self, coefficient=1.0, show_now = True) :
        """Plots of excitations arrays, for visual checking."""
        msg = "FieldComputation.plot_excitations :: Error\n\t"
        opt = self.PLT_OPT
        tm_gd = self.__FCIP.time_grid
        sp_gd = self.__FCIP.space_grid
        dxs2,dts2 = 0.5e3 * sp_gd.dx, 0.5e6 * tm_gd.dt
        if self.__FCIP.is_3d :
            dys2 = 0.5e3 * sp_gd.dy
            msg += "3d case not yet implemented. Sorry!"
            raise ValueError(msg)
            ## TO DO ##
        else : # 2d case
            figs, axes, ims, dvds, caxs = [],[],[],[],[]
            for slc, tp in zip(self.__slices, self.__tab_pth) :
                it_min, it_max = slc[0].start,  slc[0].stop 
                ix_min, ix_max = slc[1].start,  slc[1].stop 
                tab_exc = np.load( tp )
                val_max = coefficient * np.abs(tab_exc).max()                
                label = os.path.basename(tp).replace("_excitation.npy","")
                fig = plt.figure(f"Stored excitation {label}",
                                 figsize = (8.5,7.5) )
                figs.append( fig )
                ax = fig.subplots(1,1)
                axes.append( ax )
                fig.subplots_adjust(0.1,0.08,0.9,0.92)
                ax.set_xlabel("Space $x$ [mm]", **opt)
                ax.set_ylabel("Time $t$ [µs]", **opt)
                ax.set_title(f"Excitation function '{label}'", **opt)
                xmn, xmx = (1e3*sp_gd.Xc[ix_min] - dxs2,
                            1e3*sp_gd.Xc[ix_max] - dxs2)
                tmx, tmn = (1e6*tm_gd.T[it_min] - dts2,
                            1e6*tm_gd.T[it_max] - dts2)
                im = ax.imshow(tab_exc, cmap="seismic",
                               vmin=-val_max, vmax=val_max,
                               aspect="auto", interpolation="none",
                               extent=(xmn, xmx, tmx, tmn) )
                ims.append( im )
                dvd = make_axes_locatable(ax)
                dvds.append( dvd )
                cax = dvd.append_axes('right', size='2%', pad=0.06)
                if abs(coefficient) < 1 :
                    plt.colorbar(im, cax=cax, extend="both")
                else :
                    plt.colorbar(im, cax=cax)
                ax.grid()
        if show_now : plt.show()
#=========================================================================
if __name__ == "__main__" :
    CP = ComputationParameters("test", 100.0, 20.0, 1.0, 0.5, 5.0)
    print(CP)
    plate_pth = "../Data/Plates/test_23-06-09.txt"
    read_plate = USMultilayerPlate.import_from_file(plate_pth)
    print(read_plate)
    FCIP = FieldComputationInProgress(CP, read_plate, verbose=True)
    print(FCIP.creation_date_time)
    print(FCIP.log)
    # For more complicated tests, see basic example in the directory
    # Examples_without_GUI/Field_computation
#=========================================================================
