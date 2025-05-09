# Version 1.1 - 2023, December 4
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import os, sys
import numpy as np
#++++ TraFiC location ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Relative path to TraFiC code:
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import TraFiC_init
from TimeGridClass import TimeGrid
from SpaceGridClasses import Space1DGrid, Space2DGrid
#=========================================================================
class ComputationParameters :
    """Management of input and output files for field computations.
       label: label of this set of parameters
       duration_µs: Total duration of the simulation, in microseconds
       delay_µs: duration before t = 0 such that any signal is zero
                 before t0 = -delay
       max_frequency_MHz: Maximum frequency in MegaHertz
       structure: "plate" (default) or "pipe"
       max_length_m: between source(s) and observation point(s), in meters
                     value in 2D, pair in 3D. For pipes, the second value
                     is the reference radius in meters necessary to
                     determine the angular discretization step.
       min_wavelength_mm: Minimum wavelength in millimeters
                          value in 2D, pair in 3D
       result_path: directory containing the expected results
                    if result_path is "auto":
                          <TraFiC root>/Data/Results/<label>
       material_path: directory containing the material files
                      if material_path is "auto":
                          <TraFiC root>/Data/Materials/
       plate_path: directory containing the plate files
                   if plate_path is "auto":
                          <TraFiC root>/Data/Plates/
       pipe_path:  directory containing the pipe files
                   if pipe_path is "auto":
                          <TraFiC root>/Data/Pipes/
    """
    #---------------------------------------------------------------------
    # root of TraFiC is one level upper than the current file
    ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    RND_F = 1.0e4  # Sampling frequency rounded to the nearest
                   # multiple of RND_F, in Hertz
    RND_X = 1.0e-5 # Discretization step(s) rounded to the nearest
                   # multiple of RDN_X, in meters
    TITLE = "Multilayer Plate/Pipe Computation:"
    #---------------------------------------------------------------------
    def __init__(self, label, duration_µs, delay_µs, max_frequency_MHz, \
                 max_length_m, min_wavelength_mm, result_path="auto",
                 material_path="auto", structure="undefined",
                 plate_path=None, pipe_path=None, verbose=False,
                 ) :
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        # Label of the set of parameters
        self.__lbl = label
        # Duration in seconds
        duration_s = 1e-6*duration_µs
        # Delay in seconds
        delay_s = 1e-6*delay_µs
        # Sampling frequency in Hz (twice the maximum frequency)
        rnd_f = ComputationParameters.RND_F # for round
        fs = rnd_f * np.ceil( 2.0*1e6*max_frequency_MHz / rnd_f - 1e-4)
        Ts = 1.0 / fs
        # Time grid :
        self.__tg = TimeGrid(duration_s, Ts, -delay_s)
        # Type of structure :
        if structure == "undefined" : # not specified
            if plate_path is not None :
                if pipe_path is not None :
                    # Both plate and pipe paths are specified
                    msg = "ComputationParameters builder :: Error:\n" + \
                          "\tBoth plate and pipe paths are specified\n"+\
                          "\tfor an undefined structure"
                    raise ValueError(msg)
                else : # pipe_path is None
                    structure = "plate"
            else : # plate_path is None
                if pipe_path is not None :
                    structure = "pipe"
                else : # Default case
                    structure = "plate"
                    plate_path = "auto"
        elif structure == "plate" :
            if plate_path is None :
                plate_path = "auto"
        elif structure == "pipe" :
            if pipe_path is None :
                pipe_path = "auto"
        else :            
            msg = "ComputationParameters builder :: Error:\n" + \
                  f"\tstructure parameter '{structure}' must be\n" + \
                   "either 'plate' or 'pipe'."
            raise ValueError(msg)
        # Discretization step(s) in x-direction in m
        # (half the minimum wavelength(s))
        try :
            x_max = float( max_length_m )
            self.__3d = False
        except Exception as err1 :
            try :
                x_max, y_max = [ float(v) for v in max_length_m ]
                self.__3d = True
            except Exception as err2 :
                msg = "ComputationParameters constructor: errors on " + \
                     f"max_length_m:\n\t'{err1}'\n\t'{err2}'"
                raise ValueError(msg) 
        rnd_x = ComputationParameters.RND_X # for round
        if self.__3d :
            if structure == "plate" :
                dx, dy = [ rnd_x * \
                           np.floor( 0.5e-3*float(v) / rnd_x + 1e-4) \
                           for v in min_wavelength_mm ]
                # Number of discretization points
                nx = 2*round(np.ceil( x_max / dx ))
                ny = 2*round(np.ceil( y_max / dy ))
            else : # structure == "pipe"
                lbdax, lbday = min_wavelength_mm
                dx = rnd_x * np.floor(0.5e-3*float(lbdax)/rnd_x + 1e-4)
                self.__ref_radius_m = y_max # reference radius in m
                r_ref = 1e3*y_max # reference radius in mm
                y_max = np.pi  # max angle
                lbday /= r_ref # angular wavelength
                # Number of discretization points
                nx = 2*round(np.ceil( x_max / dx ))
                ny = 2*round(np.ceil( 2*y_max / lbday ))
                dy = 2*y_max / ny
            # Space Grid
            self.__sg = Space2DGrid(nx, ny, dx, dy)
        else :
            dx = rnd_x * \
                 np.floor( 0.5e-3*float(min_wavelength_mm)/rnd_x + 1e-4)
            # Number of discretization points
            nx = 2*round(np.ceil( x_max / dx ))
            # Space Grid
            self.__sg = Space1DGrid(nx, dx)
        # Directory pathes    
        if result_path == "auto" :
            result_path = os.path.join(self.ROOT, "Data", \
                                       "Results", label)
        RP, dirs = result_path, []
        while not os.path.isdir(RP) and RP != "" :
            dirs.append(RP)
            RP = os.path.dirname(RP)
        for dr in reversed(dirs) :            
            os.mkdir(dr)
            self.__prt(f"Directory '{dr}' created.")
        self.__RP = result_path
        self.__GTP = os.path.join(self.__RP, "Green_tensors")
        self.__FP = os.path.join(self.__RP, "Fields")
        for pth in (self.__GTP, self.__FP) :
            if not os.path.isdir(pth) :
                os.mkdir(pth)
                self.__prt(f"Directory '{pth}' created.")        
        if material_path == "auto" :
            material_path = os.path.join(self.ROOT, "Data", \
                                         "Materials", "")
        if not os.path.isdir(material_path) :
            msg = "TraFiC :: ComputationParameters :: error:\n" + \
                 f"\t'{material_path}' does not exist."
            raise ValueError(msg)
        self.__MP = material_path
        self.__struct = structure
        if structure == "plate" :
            if plate_path == "auto" :
                plate_path = os.path.join(self.ROOT, "Data", "Plates", "")
            if not os.path.isdir(plate_path) :
                msg = "TraFiC :: ComputationParameters :: error:\n" + \
                     f"\t'{plate_path}' does not exist."
                raise ValueError(msg)
            self.__PP = plate_path
        else : # structure == "pipe" :
            if pipe_path == "auto" :
                pipe_path = os.path.join(self.ROOT, "Data", "Pipes", "")
            if not os.path.isdir(pipe_path) :
                msg = "TraFiC :: ComputationParameters :: error:\n" + \
                     f"\t'{pipe_path}' does not exist."
                raise ValueError(msg)
            self.__PP = pipe_path
    #---------------------------------------------------------------------
    @property
    def time_grid(self) : return self.__tg
    @property
    def tm_gd(self) : return self.__tg
    @property
    def Ts_µs(self) :
        """Sampling period (in microseconds)."""
        return 1e6 * self.__tg.Ts
    @property
    def fs_MHz(self) :
        """Sampling frequency (in MegaHertz)."""
        return 1e-6 / self.__tg.Ts
    @property
    def duration_µs(self) :
        """Sampling period (in microseconds)."""
        return 1e6*self.__tg.duration
    @property
    def delay_µs(self) :
        """Delay before t=0 (in microseconds)."""
        return -1e6*self.__tg.t0
    @property
    def tmax_µs(self) :
        """Maximum time value (in microseconds)."""
        return np.round(1e6*(self.__tg.duration + self.__tg.t0),6)
    @property
    def gamma_MHz(self) :
        """Gamma parameter (in MegaHertz)."""
        return 1e-6*self.__tg.gamma
    @property
    def ns(self) :
        """Number of values in the Laplace domain."""
        return self.__tg.ns
    @property
    def nt(self) : 
        """Number of time values."""
        return self.__tg.nt
    @property
    def T(self) : 
        """Time values (in seconds)."""
        return self.__tg.time_values
    @property
    def T_µs(self) : 
        """Time values (in microseconds)."""
        return 1e6 * self.__tg.time_values
    @property
    def space_grid(self) : return self.__sg
    @property
    def sp_gd(self) : return self.__sg
    @property
    def is_3d(self) : return self.__3d
    @property
    def is_2d(self) : return not self.__3d
    @property
    def nx(self) :
        """Number of space values in the x direction."""
        return self.__sg.nx
    @property
    def dx(self) :
        """Discretization step in the x direction (in meters)."""
        return self.__sg.dx
    @property
    def xmax(self) :
        """Maximum value in the x direction (in meters)."""
        if self.__3d :
            return self.__sg.x_grid.xmax
        else : # 2d
            return self.__sg.xmax
    @property
    def dx_mm(self) :
        """Discretization step in the x direction (in meters)."""
        return 1e3*self.__sg.dx
    @property
    def xmax_mm(self) :
        """Maximum value in the x direction (in millimeters)."""
        if self.__3d :
            return 1e3*self.__sg.x_grid.xmax
        else : # 2d
            return 1e3*self.__sg.xmax
    @property
    def ny(self) :
        """Number of space values in the y direction."""
        if self.__3d :
            return self.__sg.ny
        msg = "ComputationParameters.ny :: Error:\n\t" + \
              "2D case, y-axis is not defined."
        raise ValueError(msg)
    @property
    def dy(self) :
        """Discretization step in the y direction (in meters). 3D only."""
        if self.__3d :
            return self.__sg.dy
        msg = "ComputationParameters.dy :: Error:\n\t" + \
              "2D case, y-axis is not defined."
        raise ValueError(msg)
    @property
    def ymax(self) :
        """Maximum value in the y direction (in meters). 3D only."""
        if self.__3d :
            return self.__sg.y_grid.xmax
        msg = "ComputationParameters.ymax :: Error:\n\t" + \
              "2D case, y-axis is not defined."
        raise ValueError(msg)
    @property
    def reference_radius(self) :
        """Reference radius in meters, for 3D pipes only.
           Undefined for other cases."""
        if self.__3d and self.__struct == "pipe" :
            return self.__ref_radius_m
        msg = "ComputationParameters.reference_radius :: Error:" + \
              "\n\tThis is not à 3D pipe."
        raise ValueError(msg)
    @property
    def dy_mm(self) :
        """Discretization step in the y direction (in millimeters).
           3D only."""
        if self.__3d :
            if self.__struct == "pipe" :
                return 1e3*self.__ref_radius_m*self.__sg.dy
            else : # self.__struct == "plate"
                return 1e3*self.__sg.dy
        msg = "ComputationParameters.dy_mm :: Error:\n\t" + \
              "2D case, y-axis is not defined."
        raise ValueError(msg)
    @property
    def da_degree(self) :
        """Angular discretization step (in degrees).
           3D pipes only. Undefined for other cases."""
        if self.__3d and self.__struct == "pipe" :
            return np.degrees( self.__sg.dy )
        msg = "ComputationParameters.da_degree :: Error:" + \
              "\n\tThis is not à 3D pipe."
        raise ValueError(msg)
    @property
    def ymax_mm(self) :
        """Maximum value in the y direction (in millimeters). 3D only."""
        if self.__3d :
            if self.__struct == "pipe" :
                return 1e3*self.__ref_radius_m*np.pi
                     # self.__sg.y_grid.xmax = np.pi
            else : # self.__struct == "plate"
                return 1e3*self.__sg.y_grid.xmax
        msg = "ComputationParameters.ymax_mm :: Error:\n\t" + \
              "2D case, y-axis is not defined."
        raise ValueError(msg)
    @property
    def amax_degree(self) :
        """Angular discretization step (in degrees).
           3D pipes only. Undefined for other cases."""
        if self.__3d and self.__struct == "pipe" :
            return 180.0 # np.degrees( self.__sg.y_grid.xmax )
        msg = "ComputationParameters.amax_degree :: Error:" + \
              "\n\tThis is not à 3D pipe."
        raise ValueError(msg)
    @property
    def label(self) : return self.__lbl
    @property
    def result_path(self) : return self.__RP
    @property
    def abs_result_path(self) :
        return os.path.abspath(self.__RP)
    @property
    def Green_tensor_path(self) : return self.__GTP
    @property
    def field_path(self) : return self.__FP
    @property
    def material_path(self) : return self.__MP
    @property
    def abs_material_path(self) :
        return os.path.abspath(self.__MP)
    @property
    def structure_type(self) : return self.__struct
    @property
    def structure_path(self) : return self.__PP
    @property
    def plate_path(self) :
        if self.__struct == "plate" :
            return self.__PP
        else :
            return None
    @property
    def pipe_path(self) :
        if self.__struct == "pipe" :
            return self.__PP
        else :
            return None
    @property
    def abs_structure_path(self) :
        return os.path.abspath(self.__PP)
    @property
    def abs_plate_path(self) :
        if self.__struct == "plate" :
            return os.path.abspath(self.__PP)
        else :
            return None
    @property
    def abs_pipe_path(self) :
        if self.__struct == "pipe" :
            return os.path.abspath(self.__PP)
        else :
            return None
    @property
    def maximum_wave_velocity(self) :
        "Maximum wave velocity [mm/µs] (1D or 2D vector)."
        if self.__3d and self.__struct == "plate" :
            x_max = np.array( [self.xmax_mm, self.ymax_mm] )
        else : # 2D or 3D pipe
            x_max = self.xmax_mm         
        t_max = self.tmax_µs
        return np.round( x_max / t_max, 2 )
    #---------------------------------------------------------------------
    @property
    def head_text(self) :
        strs = 80*"*"+"\n"
        txt =  strs + self.TITLE + f" '{self.__lbl}'\n" + \
              strs + "Computation parameters:\n\t" + \
              f"Duration: {self.duration_µs:.2f} µs\n\t" + \
              f"Delay: {self.delay_µs:.2f} µs\n\t" + \
              f"Gamma: {self.gamma_MHz:.7f} MHz\n\t" + \
              f"Number of s values: {self.ns}\n\t" + \
              f"dt: {self.Ts_µs:.7f} µs\n\t"
        if self.__3d :
            if self.__struct == "plate" :
                txt += f"2D Space grid: {self.nx} x {self.ny} nodes" + \
                       f"\n\txmax: {self.xmax_mm:.3f} mm\n\t" + \
                       f"dx: {self.dx_mm:.7f} mm\n\t" + \
                       f"ymax: {self.ymax_mm:.3f} mm\n\t" + \
                       f"dy: {self.dy_mm:.7f} mm\n"
            else : # self.__struct == "pipe" :
                rrf = 1e3*self.__ref_radius_m
                txt += f"2D Space grid: {self.nx} x {self.ny} nodes" + \
                       f"\n\tzmax: {self.xmax_mm:.3f} mm\n\t" + \
                       f"dz: {self.dx_mm:.7f} mm\n\t" + \
                       f"nmax: {self.ny//2} \n\t" + \
                       f"da: {self.da_degree:.7f}° " + \
                       f"({self.dy_mm:.7f} mm for the ref. " + \
                        "radius)\n\t" + \
                       f"Reference radius: {rrf:.7f} mm\n"
        else : # 2d
            if self.__struct == "plate" :
                txt += f"1D Space grid: {self.nx} nodes\n\t" + \
                       f"xmax: {self.xmax_mm:.3f} mm\n\t" + \
                       f"dx: {self.dx_mm:.7f} mm\n"
            else : # self.__struct == "pipe" :
                txt += f"1D Space grid: {self.nx} nodes\n\t" + \
                       f"zmax: {self.xmax_mm:.3f} mm\n\t" + \
                       f"dz: {self.dx_mm:.7f} mm\n"
        txt += strs + f"Materials in: {self.abs_material_path}\n"
        if self.__struct == "plate" : 
            txt += "Plates"
        else :
            txt += "Pipes"
        txt += f" in: {self.abs_structure_path}\n" + \
               f"Results in: {self.abs_result_path}\n" + strs[:-1] + "\n"               
        return txt
    #---------------------------------------------------------------------
    def __str__(self) :
        return self.head_text
    #---------------------------------------------------------------------
    @staticmethod
    def from_head_text( text, verbose=False ) :
        msg = "ComputationParameters.from_head_text :: error:\n\t"
        title = ComputationParameters.TITLE
        if verbose :
            prt = print
        else :
            prt = lambda *args : None
        dico = dict()
        label = None
        rows = [ r.strip() for r in text.split("\n") ]
        for r in rows :
            if title in r :
                label = r.split(title)[1].strip( \
                                         ).replace("'","").replace('"',"")
            elif "in:" in r :
                try : k,v = [ m.strip() for m in r.split("in:") ]
                except : pass
                dico[k.lower()] = v                
            elif ":" in r :
                try : k,v = [ m.strip() for m in r.split(":") ]
                except : pass
                dico[k.lower()] = v
        if label is None :
            msg += "label is not defined"
            raise ValueError(msg)
        prt( dico )
        is_3d = '2d space grid' in dico.keys() 
        is_2d = '1d space grid' in dico.keys()
        if ( is_3d and is_2d ) or ( not is_3d and not is_2d ) :
            msg += f"dimension of space grid 1d={is_2d} and 2d={is_3d}"
            raise ValueError(msg)
        # Pathes :
        RP = dico.get('results')
        if RP is None : RP = 'auto'
        MP = dico.get('materials')
        if MP is None : MP = 'auto'
        PP = dico.get('plates')
        if PP is None : 
            PP = dico.get('pipes')
            if PP is None :
                struct = "plate"
                plate_path = "auto"
                pipe_path = None
            else :
                struct = "pipe"
                plate_path = None
                pipe_path = PP
        else :
            struct = "plate"
            plate_path = PP
            pipe_path = None
        try :
            # Time parameters
            dur, unit = dico['duration'].split()
            dur = float(dur)
            if unit == "µs" :
                d = dur
            elif unit == "ms" :
                d = 1e3*dur
            elif unit == "s" :
                d = 1e6*dur
            delay, unit = dico['delay'].split()
            delay = float(delay)
            if unit == "µs" :
                tau = delay
            elif unit == "ms" :
                tau = 1e3*delay
            elif unit == "s" :
                tau = 1e6*delay
            dt, unit = dico['dt'].split()
            dt = float(dt)
            if unit == "µs" :
                fmax = 0.5/dt
            elif unit == "ms" :
                fmax = 0.5e-3/dt
            elif unit == "s" :
                fmax = 0.5e-6/dt
            # Space parameters
            try :
                dx, unit = dico['dx'].split()
            except Exception as err1 :
                try :
                    dx, unit = dico['dz'].split()
                except Exception as err2 :
                    msg += f"'{err1}' and\n\t'{err2}'"
                    raise ValueError(msg)                    
            dx = float(dx)
            if unit == "mm" :
                lbdx = 2.0*dx
            elif unit == "m" :
                lbdx = 2.0e3*dx
            try : # plate
                xmax, unit = dico['xmax'].split()
                xmax = float(xmax)
            except Exception as err1 : 
                try : # pipe
                    xmax, unit = dico['zmax'].split()
                    xmax = float(xmax)
                except Exception as err2 :
                    msg += f"'{err1}' and\n\t'{err2}'"
                    raise ValueError(msg)                    
            if unit == "mm" :
                Lx = 1e-3*xmax
            elif unit == "m" :
                Lx = xmax          
            if is_2d: # dx only
                nx = int(dico['1d space grid'].split()[0])
                return ComputationParameters(label, d, tau, fmax, Lx, \
                                             lbdx, RP, MP, struct, \
                                             plate_path, pipe_path)
            else : # (dx,dy)
                try : # plate
                    dy, unit = dico['dy'].split()
                    dy = float(dy)
                except Exception as err1 : 
                    try : # pipe
                        da = dico['da'].split()[0]
                        if "°" in da : # degrees
                            da = np.radians(float(da.replace("°","")))
                        else : # radians
                            da = float(da)
                        rref, unit = dico['reference radius'].split()
                        rref = float(rref)  
                        if unit == "mm" :
                            Ly = 1e-3*rref
                        elif unit == "m" :
                            Ly = rref
                        dy = rref*da
                    except Exception as err2 :
                        msg += f"'{err1}' and\n\t'{err2}'"
                        raise ValueError(msg)  
                if unit == "mm" :
                    lbdy = 2.001*dy
                elif unit == "m" :
                    lbdy = 2.001e3*dy
                try : # plate
                    ymax, unit = dico['ymax'].split()
                    ymax = float(ymax)
                    if unit == "mm" :
                        Ly = 1e-3*ymax
                    elif unit == "m" :
                        Ly = ymax  
                except Exception as err1 : 
                    try : # pipe
                        nmax = dico['nmax']
                        nmax = int(nmax)
                    except Exception as err2 :
                        msg += f"'{err1}' and\n\t'{err2}'"
                        raise ValueError(msg) 
                nx,_,ny,_ = dico['2d space grid'].split()
                nx, ny = int(nx), int(ny)               
                return ComputationParameters(label, d, tau, fmax, \
                                             (Lx,Ly), (lbdx,lbdy),\
                                              RP, MP, struct, \
                                              plate_path, pipe_path)
        except Exception as err :
            msg += f"'{err}'"
            raise ValueError(msg)
#=========================================================================
if __name__ == "__main__" :
    # Computation parameters for plates
    for L_m,WL_mm in ((0.6,2.4),((0.6,0.48),(2.4,3.2))) :
        CP = ComputationParameters("test", 120.0, 20.0, 0.5, \
                                   L_m, WL_mm, verbose=True)
        print(CP)
        print(f"Maximum wave velocity: {CP.maximum_wave_velocity} mm/µs")
        copy_CP = ComputationParameters.from_head_text(CP.head_text)
        print("Test of ComputationParameters.from_head_text:", \
              CP.head_text == copy_CP.head_text)
    # Computation parameters for pipes
    for L_m,WL_mm in ((0.6,2.4),((0.6,0.05),(2.4,3.2))) :
        CP = ComputationParameters("test", 120.0, 20.0, 0.5, \
                                   L_m, WL_mm, \
                                   structure="pipe", verbose=True)
        print(CP)
        print(f"Maximum wave velocity: {CP.maximum_wave_velocity} mm/µs")
        copy_CP = ComputationParameters.from_head_text(CP.head_text)
        print("Test of ComputationParameters.from_head_text:", \
              CP.head_text == copy_CP.head_text)
#=========================================================================
