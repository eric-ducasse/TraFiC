# Version 1.00 - 2023, June 27
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
from SpaceGridClasses import Space1DGrid, Space2DGrid
from TraFiC_utilities import MAXMEM
#=========================================================================
class ComputationParameters :
    """Management of input and output files for field computations.
       label : label of this set of parameters
       time_grid : is the time grid (TimeGrid instance)
       space_grid : is the space grid (Space1DGrid or Space2DGrid
                                       instance)
       result_path : directory containing the expected results
                     if result_path is "auto":
                          <TraFiC root>/Data/Results/<label>
       material_path : directory containing the material files
                     if material_path is "auto":
                          <TraFiC root>/Data/Materials/
    """
    #---------------------------------------------------------------------
    # root of TraFiC is one level upper than the current file
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #---------------------------------------------------------------------
    def __init__(self, label, time_grid, space_grid, result_path="auto",
                 material_path="auto", verbose=False) :
        self.__lbl = label
        self.__tg = time_grid
        self.__sg = space_grid
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        if result_path == "auto" :
            result_path = os.path.join(self.root, "Data", \
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
            material_path = os.path.join(self.root, "Data", \
                                         "Materials", "")
        if not os.path.isdir(material_path) :
            msg = "TraFiC :: ComputationParameters :: error:\n" + \
                 f"\t'{material_path}' does not exist."
            raise ValueError(msg)
        self.__MP = material_path
    #---------------------------------------------------------------------
    @property
    def time_grid(self) : return self.__tg
    @property
    def tm_gd(self) : return self.__tg
    @property
    def space_grid(self) : return self.__sg
    @property
    def sp_gd(self) : return self.__sg
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
    def maximum_wave_velocity(self) :
        "Maximum wave velocity [mm/µs] (1D or 2D vector)."
        if isinstance(self.__sg, Space1DGrid) :
            x_max = np.array( [self.__sg.xmax] )
        else :
            x_max = np.array( [self.__sg.x_grid.xmax, \
                               self.__sg.y_grid.xmax] )            
        t_max = self.__tg.t0 + self.__tg.d
        return np.round( 1e-3 * x_max / t_max, 2 )
    #---------------------------------------------------------------------
    @property
    def head_text(self) :
        Ts_µs = 1e6*self.__tg.Ts
        duration_µs = 1e6*self.__tg.duration
        delay_µs = -1e6*self.__tg.t0
        gamma_MHz = 1e-6*self.__tg.gamma
        ns = self.__tg.ns
        strs = 70*"*"+"\n"
        txt =  "Multilayer Plate/Pipe Computation\n" + strs + \
               "Computation parameters:\n\t" + \
              f"Duration: {duration_µs:.2f} µs\n\t" + \
              f"Delay: {delay_µs:.2f} µs\n\t" + \
              f"Gamma: {gamma_MHz:.7f} MHz\n\t" + \
              f"Number of s values: {ns}\n\t" + \
              f"dt: {Ts_µs:.7f} µs\n\t"
        if isinstance(self.__sg, Space1DGrid) :
            nx = self.__sg.nx
            dx_mm = 1e3*self.__sg.dx
            xmax_mm = 1e3*self.__sg.xmax
            txt += f"1D Space grid: {nx} nodes\n\t" + \
              f"xmax: {xmax_mm:.3f} mm\n\t" + \
              f"dx: {dx_mm:.7f} mm\n"
        elif isinstance(self.__sg, Space2DGrid) :
            nx = self.__sg.nx
            ny = self.__sg.ny
            dx_mm = 1e3*self.__sg.dx
            dy_mm = 1e3*self.__sg.dy
            xmax_mm = 1e3*self.__sg.x_grid.xmax
            ymax_mm = 1e3*self.__sg.y_grid.xmax
            txt += f"2D Space grid: {nx} x {ny} nodes\n\t" + \
              f"xmax: {xmax_mm:.3f} mm\n\t" + \
              f"dx: {dx_mm:.7f} mm\n\t" + \
              f"ymax: {ymax_mm:.3f} mm\n\t" + \
              f"dy: {dy_mm:.7f} mm\n"
        txt += strs + f"Materials in: {self.abs_material_path}\n" + \
                f"Results in: {self.abs_result_path}\n" + strs[:-1]                
        return txt
    #---------------------------------------------------------------------
    def __str__(self) :
        return self.head_text
#=========================================================================
if __name__ == "__main__" :
    from TimeGridClass import TimeGrid
    TG = TimeGrid(1.2e-4, 1e-6, 0.2e-4)
    for SG in (Space1DGrid(1000, 1.2e-3),
               Space2DGrid(1000, 600, 1.2e-3, 1.6e-3)) :
        CP = ComputationParameters("test", TG, SG, verbose=True)
        print(CP)
        print(f"Maximum wave velocity: {CP.maximum_wave_velocity} mm/µs")
#=========================================================================
