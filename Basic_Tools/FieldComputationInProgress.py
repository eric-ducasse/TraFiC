# Version 1.00 - 2023, July 7
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import os, sys, psutil
import numpy as np
#++++ TraFiC location ++++++++++++++++++++++++++++++++++++++++++++++++++++
# Relative path to TraFiC code:
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root)
import TraFiC_init
from USMultilayerPlate import USMultilayerPlate
from USMultilayerPipe import USMultilayerPipe
from ComputationParameters import ComputationParameters
from TraFiC_utilities import now
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
    AVAILABLERAM = round(0.5*psutil.virtual_memory().free*2**-30,2)
                       # 50% of the virtual availbale memory
    MAXMEM = min(MAXMEM, AVAILABLERAM)
    FNAME = "Computation_in_progress.txt"
    LINESIZE = USMultilayerPlate.LINESIZE
    STAGES = ["Creation", \
              "Definition of the computation parameters", \
              "Definition of the structure", \
              "Computation of the Green tensors", \
              "Ready for field computations for given sources" ]
    #---------------------------------------------------------------------
    def __init__(self, comp_params_or_path, structure=None, \
                 verbose=False) :
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
                msg += "Error:\n\tThe 1st parameter is not a " + \
                       "'ComputationParameters'\n\tinstance."
                raise ValueError(msg)
            self.__CP = comput_params
            # Multilayer structure
            if isinstance( structure, USMultilayerPlate ) :
                self.__case = "Plate"
            elif isinstance( structure, USMultilayerPipe ) :
                self.__case = "Pipe"
                msg += "Sorry!\n\tComputation in pipes not yet available."
                print(msg)
                return
            else :
                msg += "Error:\n\tThe 2nd parameter is neither a " + \
                       "'USMultilayerPlate' nor a" + \
                       "\n\t'USMultilayerPipe' instance."
                raise ValueError(msg)
            self.__MS = structure
            # Head file exists?
            if os.path.isfile( self.head_file_path ) :
                msg += f"Warning:\n\tComputation '{self.label}' " + \
                       "already started."
                prt( msg )
                self.update_from_file( ) # Update with check
            else :
                self.__stage = 2 # Ready for Green tensor computation
                self.__creation = now(True)
                self.__update_text()
                self.save()
    #---------------------------------------------------------------------
    def __update_text(self, additional_text="" ) :
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
        self.__text += additional_text
        if self.__stage == 4 : # finished Green tensor computation
            self.__text += self.stage_text(4)
    #---------------------------------------------------------------------
    def save( self ) :        
        with open( self.head_file_path, "w", encoding="utf8" ) as strm :
            strm.write( self.__text )
    #---------------------------------------------------------------------
    def compute_Green_tensors( self, possible_excitations ) :
        """To start or continue the computation of Green tensors.
             possible_excitations is a list of pairs ( S, I ) where :
                > S is in {"Fx", "Fy", "Fz", "Ux", "Uy", "Uz"}
                > I is the number of the interface where a jump on S is
                  imposed.
                > (S, I) can be simply replaced by S for the interface
                  either between the first layer and the upper half-space
                  in a plate (I=0) or between the last layer and the
                  external space in a pipe (I=-1)
                > Vacuum/Fluid : "Fz" only
                  Wall/Solid : "Uz" only
                  Vacuum/Solid : "Fx", "Fy", "Fz" only
                  Fluid/Fluid : "Fz", "Uz" only
                  Fluid/Solid : "Fx", "Fy", "Fz", "Uz" only
                  Solid/Solid : all are possible
        """
        msg = "FieldComputationInProgress.computeGreenTensors :: "
        prt = self.__prt
        if self.__stage == 2 : # New Green tensor computation
            self.__stage = 3
            text_stage = self.STAGES[3]
            prt(f"+++ text_stage 3 +++\n{text_stage}")
            self.__text += self.stage_text(3)
            self.__exc_numbers = []
            Imax = len(self.__MS.layers)
            if self.__case == "Plate" :
                default_I = 0
                L = USMultilayerPlate.THS # Top half-space
                R = USMultilayerPlate.BHS # Top half-space
                matL = self.__MS.topUSmat # Top half-space
                matR = self.__MS.bottomUSmat # Bottom half-space
            else : # self.__case == "Pipe"
                default_I = Imax
                R = USMultilayerPipe.INCYL # Inner Cylinder
                L = USMultilayerPipe.EXTSP # External space
                matL = self.__MS.internalUSmat # External space
                matR = self.__MS.externalUSmat # Inner Cylinder
            for S_I in possible_excitations :
                try :
                    S,I = S_I
                    S,I = str(S), int(I)
                except : # Default interface
                    S = S_I
                    I = default_I
                if I == 0 :
                    # First interface
                    prt((S,I,L))
                elif I == -1 or I == Imax :
                    # Last interface
                    prt((S,I,R))
                else : #between 2 layers
                    prt((S,I))
                    
                
    
    #---------------------------------------------------------------------
    def pick_up_Green_tensor_computation( self ) :
        msg = "FieldComputationInProgress.pick_up_Green_tensor_" + \
              "computation\n:: Error:\n\t"
        prt = self.__prt
        prt("Beginning of FieldComputationInProgress." + \
            "pick_up_Green_tensor_computation")
        #### TO DO ####
        prt("End of FieldComputationInProgress." + \
            "pick_up_Green_tensor_computation")
    #---------------------------------------------------------------------
    def update_from_file(self, file_path=None ) :
        """Updates the FieldComputationInProgress for file_path.
           The default value of file_path is the attribute head_file_path.
        """
        msg = "FieldComputationInProgress.update_from_file :: Error:\n\t"
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
                    msg += "Unexpected new stage"
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
            blocks.append( "\n".join(rows[nb:]) )
            cs = False
        stage = stages[-1]
        if not np.allclose( stages, np.arange(stage+1) ) :
            msg += f"stages is {stages}"
            raise ValueError(msg)
        # ++++++++++ Initializations +++++++++++++++++++++++++++++++++++++
        self.__stage = stage
        self.__creation = blocks[0].strip()
        self.__CP = ComputationParameters.from_head_text(blocks[1])
        if check :
            if old_CP.head_text != self.__CP.head_text :
                msg1 = msg + "Warning:\n\tThe new and old " + \
                       "computation parameters differ."
                prt(msg1)
        self.__MS = USMultilayerPlate.import_from_text(blocks[2])
        if check :
            if old_MS.write_in_file() != self.__MS.write_in_file() :
                msg += "Warning:\n\tThe new and old computation " + \
                       "parameters differ."
                prt(msg)
        # +++++++++ Started Green tensor computation? ++++++++++++++++++++
        if self.__stage == 2 :
            prt("The Green tensor computation is not started.")
            self.__update_text(self)
        else :
            self.__update_text(self, blocks[3] )
            if self.__stage == 3 :
                prt("The Green tensor computation is in progress...")
                self.pick_up_Green_tensor_computation()
            else : # self.__stage == 4
                prt("The Green tensor computation is finished.\n" + \
                    "Ready for field computations.")
        prt("End of FieldComputationInProgress.update_from_file")
    #---------------------------------------------------------------------
    @property
    def head_file_path(self) :
        return os.path.join( self.__CP.result_path, \
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
