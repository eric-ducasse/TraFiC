# Version 0.84 - 2024, February 19
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import os, sys, time, platform, pickle
from PyQt5.QtWidgets import (QWidget, QFrame, QLineEdit, \
                             QPushButton, QFrame, QLabel, QComboBox, \
                             QGridLayout, QHBoxLayout, QVBoxLayout, \
                             QFileDialog, QMessageBox, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QEvent, QRect, QSize, pyqtSignal, QTimer
from matplotlib.backends.backend_qt5agg import  \
    FigureCanvasQTAgg as FigureCanvas,          \
    NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
if __name__ == "__main__" :
    root = os.path.abspath("..")
    sys.path.append(root)
    import TraFiC_init
from MaterialClasses import *
from Small_Widgets import (QVLine, QHLine, Selector, PopUp_Select_Item, \
                           PopUp_Interrupt)
from MaterialEdit import Entry
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Mode_Frame(QWidget):
    CASES = ["Fixed Frequency", "Fixed Wavenumber",
             "Fixed Phase Velocity"]
    UNITS = ["MHz", "mm^-1", "mm/µs"]
    #--------------------------------------------------------------------
    def __init__(self, mainWindow):
        QWidget.__init__(self, mainWindow)
        self.mw = mainWindow
        # Global layout
        self.layout = QGridLayout()
        # Discretization parameters
        but_disc = QPushButton("Show/Change Discretization", self)
        but_disc.released.connect(self.__change_discretization)         
        self.layout.addWidget(but_disc, 0, 0)
        # Choice of fixed parameter
        self.__prev_prm_idx = None
        self.prm_cmb = QComboBox(self)
        self.prm_cmb.activated.connect(\
                                   self.__change_fixed_parameter_name)
        # User Interface Initialization 
        self.__initUI()       
    #--------------------------------------------------------------------
    def __initUI(self):
        self.layout.addWidget(self.prm_cmb, 0, 1)

        # Final layout
        hlay = QHBoxLayout()
        hlay.addLayout(self.layout)
        hlay.addStretch()
        vlay = QVBoxLayout()
        vlay.addLayout(hlay)
        vlay.addStretch()
        self.setLayout(vlay)
        # Combo Box intialization
        self.prm_cmb.clear()
        self.prm_cmb.addItems(Mode_Frame.CASES)
        self.__prev_prm_idx = 0
        self.prm_cmb.setCurrentIndex(self.__prev_prm_idx)
    #--------------------------------------------------------------------
    def __change_discretization(self) :
        print("*** Mode_Frame.__change_discretization: TO DO ***")
    #--------------------------------------------------------------------
    def __change_fixed_parameter_name(self) :
        print("*** Mode_Frame.__set_fixed_parameter_name ***")
        case = self.prm_cmb.currentIndex()
        if self.__prev_prm_idx != case :
            print(f"{self.__prev_prm_idx} -> {case}")
            self.__prev_prm_idx = case
            self.change_fixed_parameter_name(case)
    #--------------------------------------------------------------------
    def change_fixed_parameter_name(self, case) :
        print("*** Mode_Frame.change_fixed_parameter_name: " + \
              "SHOULD NOT BE CALLED! ***")
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Mode_Shape_Frame(Mode_Frame):
    LABELS = ["  Frequency : ", "  Wavenumber : ", "  Phase velocity: "]
    IR = (2,2,0)
    RLABELS = [ ]
    for i in IR : RLABELS.append( LABELS[i] )
    RUNITS = [ Mode_Frame.UNITS[i] for i in IR ]
    IS = (1,0,1)
    SLABELS = [ ]
    for i in IS : SLABELS.append( LABELS[i] )
    SUNITS = [ Mode_Frame.UNITS[i] for i in IS ]
    START = [" [ Compute ] ","*** Go! ***"]
    ZERO_NUM = 1e-7
    IMKMIN = -0.046 # 46 Neper/m (divided by 10 each 5 cm)
    IMKMAX =  0.046
    IMFMIN = -1.0e-3 # No negative imaginary part for the eigenfrequency
    IMFMAX =  0.366  # 2.3 rad/µs (divided by 10 each µs)
    #--------------------------------------------------------------------
    def __init__(self, mainWindow):
        Mode_Frame.__init__(self, mainWindow)
        # Name of the fixed parameter
        self.__lab_fixprm = QLabel(self)
        # Value editor of the fixed parameter
        self.__edi_fixprm = QLineEdit(self)
        self.__edi_fixprm.editingFinished.connect(self.__update_param)
        self.__prev_val = None
        # Unit of the fixed parameter
        self.__uni_fixprm = QLabel(self)
        
        # Name of the first computed parameter
        self.__lab_cptprm = QLabel(self)
        # Starter of computation/Selector of eigenmode
        self.__cmb_mode = QComboBox(self)
        self.__cmb_mode.currentIndexChanged.connect(self.__update_modes)
        # Unit of the first computed parameter
        self.__uni_cptprm = QLabel(self)
        
        # Name of the second computed parameter
        self.__lab_other = QLabel(self)
        # Starter of computation/Selector of eigenmode
        self.__val_other = QLineEdit(self)
        self.__val_other.setEnabled(False)
        # Unit of the first computed parameter
        self.__uni_other = QLabel(self)
        
        # Energy velocity
        self.__lab_Ve = QLabel(self)
        # Starter of computation/Selector of eigenmode
        self.__val_Ve = QLineEdit(self)
        self.__val_Ve.setEnabled(False)
        # Unit of the first computed parameter
        self.__uni_Ve = QLabel(self)

        # Figure
        self.__figure = Mode_Shape_Figure(self)
        
        # Computation done ?
        self.__modes = None
        
        # User Interface Initialization 
        self.__initUI()       
    #--------------------------------------------------------------------
    def __initUI(self):
        # Buttons
        self.layout.addWidget(self.__lab_fixprm, 0, 2)
        self.layout.addWidget(self.__edi_fixprm, 0, 3)
        self.layout.addWidget(self.__uni_fixprm, 0, 4)
        self.layout.addWidget( QVLine(), 0, 5, 3, 1)
        self.layout.addWidget(self.__lab_cptprm, 0, 6)
        self.layout.addWidget(self.__cmb_mode, 0, 7)
        self.layout.addWidget(self.__uni_cptprm, 0, 8)
        self.layout.addWidget( QHLine(), 1, 2, 1, 7)
        self.layout.addWidget(self.__lab_other, 2, 2)
        self.layout.addWidget(self.__val_other, 2, 3)
        self.layout.addWidget(self.__uni_other, 2, 4)
        self.layout.addWidget(self.__lab_Ve, 2, 6)
        self.__lab_Ve.setText("Energy Velocity")
        self.layout.addWidget(self.__val_Ve, 2, 7)
        self.layout.addWidget(self.__uni_Ve, 2, 8)
        self.__uni_Ve.setText("mm/µs")
        # Figure
        self.layout.addWidget(self.__figure, 3, 0, 1, 9)
        # Update labels and units
        self.change_fixed_parameter_name(self.prm_cmb.currentIndex())
        self.__cmb_mode.clear()
        self.__cmb_mode.addItems(self.START)
        self.__cmb_mode.setEnabled(False)
    #--------------------------------------------------------------------
    def change_fixed_parameter_name(self, case) :
        print(Mode_Frame.CASES[case])
        self.__lab_fixprm.setText(Mode_Shape_Frame.LABELS[case])
        self.__uni_fixprm.setText(Mode_Shape_Frame.UNITS[case])
        self.__lab_cptprm.setText(Mode_Shape_Frame.RLABELS[case])
        self.__uni_cptprm.setText(Mode_Shape_Frame.RUNITS[case])
        self.__lab_other.setText(Mode_Shape_Frame.SLABELS[case])
        self.__uni_other.setText(Mode_Shape_Frame.SUNITS[case])
        self.__edi_fixprm.setText("") # Clear parameter value
        self.__update_param()
    #--------------------------------------------------------------------
    def __update_param(self, fmt=".3f") :
        try :
            value = complex(self.__edi_fixprm.text())
            if np.abs(value.imag) < self.ZERO_NUM*np.abs(value.real) :
                value = value.real
            to_write = ("{:"+fmt+"}").format(value)
            self.__edi_fixprm.setText(to_write)
        except Exception as err:
            # print(f"Mode_Shape_Frame.__update_param - error:\n\t{err}")
            self.__edi_fixprm.setText("")
            value = None
        if value == self.__prev_val :
            return # no change
        self.__modes = None
        self.__prev_val = value
        self.__cmb_mode.clear()
        self.__cmb_mode.addItems(self.START)
        self.__val_other.setText("")
        self.__val_Ve.setText("")
        self.__figure.set_mode( None )
        if value is None :
            self.__cmb_mode.setEnabled(False)
        else :
            self.__cmb_mode.setEnabled(True)
    #--------------------------------------------------------------------
    def __update_modes(self) :
        case = self.prm_cmb.currentIndex()
        # 0 -> Fixed frequency, 1 -> Fixed wawenumber,
        # 2 -> Fixed phase velocity
        if self.__modes is None :
            if self.__cmb_mode.currentIndex() == 1 : #go!
                if case == 0 : # Fixed frequency
                    f_Hz = 1e6*self.__prev_val
                    plate = self.mw.discretized_plate
                    self.__modes = plate.modes_for_given_frequency(f_Hz)
                    self.__modes.reverse()
                    self.__modes = [ m for m in self.__modes if \
                                self.IMKMIN < m.k.imag < self.IMKMAX ]
                    print("fixed frequency mode computation finished")
                    L_Vphi = [ f"{m.Vphi:.5f}" for m in self.__modes]
                    self.__cmb_mode.clear()
                    self.__cmb_mode.addItems(L_Vphi)
                elif case == 1 : # Fixed wawenumber
                    k_pmm = 1e3*self.__prev_val
                    plate = self.mw.discretized_plate
                    self.__modes = \
                                 plate.modes_for_given_wavenumber(k_pmm)
                    self.__modes.reverse()
                    self.__modes = [ m for m in self.__modes if \
                                self.IMFMIN < m.f.imag < self.IMFMAX ]
                    print("fixed wavenumber mode computation finished")
                    L_Vphi = [ f"{m.Vphi:.5f}" for m in self.__modes]
                    self.__cmb_mode.clear()
                    self.__cmb_mode.addItems(L_Vphi)
                elif case == 2 : # Fixed phase velocity
                    print("Fixed-phase-velocity computation not yet " + \
                          "available")
            return
        else : # A mode is selected
            if len(self.__modes) == 0 :
                msg = f"No mode found satisfying required conditions."
                qmb = QMessageBox(self) 
                qmb.setWindowTitle("Warning!")
                qmb.setIcon(QMessageBox.Warning)
                qmb.setText(msg.replace("\n","<br>"))
                qmb.setStandardButtons(QMessageBox.Ok);
                rep = qmb.exec_()
                return     
                
            no_mode = self.__cmb_mode.currentIndex()
            cur_mode = self.__modes[no_mode]
            if case == 0 : # Fixed frequency
                self.__val_other.setText(f"{cur_mode.k:.5f}")
            elif case == 1 : # Fixed wawenumber
                self.__val_other.setText(f"{cur_mode.f:.6f}")
            elif case == 2 : # Fixed phase velocity
                pass
            # Energy velocity
            if cur_mode.is_a_true_guided_mode :
                self.__val_Ve.setText(f"{cur_mode.Ve:.3f}")
            else :
                self.__val_Ve.setText("(undefined)")
            self.__figure.set_mode( cur_mode )
    #--------------------------------------------------------------------
    def reinit(self) :
        self.__edi_fixprm.setText("")
        self.__update_param()
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Mode_Shape_Figure(QFrame):
    DEFONT = {"size":12, "family":"Arial"} #, "weight": "bold"}
    DZ_MM = 0.01      # Default discretization step in z
    C_FLUID_HS = 5.0  # Ratio of the thickness of the fluid half-space by
                      #  the plate width
    C_EMPTY_HS = 0.13 # Ratio of the thickness of the empty half-space by
                      #  the plate width
    U_LABELS = ["Ux","Uy","Uz"]
    S_LABELS = ["Sxx","Sxy","Syy","Sxz","Syz","Szz"]
    P_LABELS = ["Px","Py","Pz","Ve","Et"]
    K2 = U_LABELS + P_LABELS # keys of length 2
    K3 = S_LABELS            # keys of length 3
    def __init__(self, parent=None, mode=None, zmin="auto", \
                 zmax="auto") :
        QFrame.__init__(self, parent)
        # Current mode
        self.__mode = mode
        if mode is not None :
            self.__update_fields(zmin, zmax)
        else :
            self.__data = None
        # Figure and canvas
        self.__figure  = Figure()
        self.__canvas  = FigureCanvas(self.__figure)
        # In-plane axes
        self.__xy_u = self.__figure.add_subplot(2,1,1)
        self.__xy_S = self.__xy_u.twinx() 
        # Out-of-plane axes
        self.__z__u  = self.__figure.add_subplot(2,1,2)
        self.__z_SP = self.__z__u.twinx()
        # Tool bas
        self.__toolbar = NavigationToolbar(self.__canvas, self)
        # Global Layout
        self.layout = QGridLayout()

        # Displacement selector
        self.__u_sel = Selector("Displacement", self.U_LABELS, self)
        self.__u_sel.released.connect(self.__change_u)
        self.__u_sel.stateChanged.connect(self.__update_u)
        self.__disp = True
        # Stresses selector
        self.__S_sel = Selector("Stress", self.S_LABELS, self)
        self.__S_sel.released.connect(self.__change_S)
        self.__S_sel.stateChanged.connect(self.__update_S)
        self.__S_sel.setEnabled(False)
        self.__stresses = False
        # Poynting selector
        self.__P_sel = Selector("Poynting", self.P_LABELS, self,
                                exclusive_items = [["Ve","Et"]])
        self.__P_sel.released.connect(self.__change_P)
        self.__P_sel.stateChanged.connect(self.__update_P)
        self.__P_sel.setEnabled(False)
        self.__Poynting = False

        # Export Button
        self.__but_exp = QPushButton( \
                            " Export Mode shape (numpy, Matlab) ", self)
        self.__but_exp.released.connect(self.__export)
        # User Interface Initialization
        self.__initUI()
    #--------------------------------------------------------------------
    def __initUI(self):
        # First Row
        hlay = QHBoxLayout()
        hlay.addWidget(self.__u_sel)
        hlay.addStretch() 
        hlay.addWidget(self.__S_sel)
        hlay.addWidget(QLabel(" or ")) 
        hlay.addWidget(self.__P_sel)        
        self.layout.addLayout(hlay, 0, 0)
        # Second Row
        self.__canvas.setFixedHeight(600)
        self.layout.addWidget(self.__canvas, 1, 0)
        # Third Row
        hbox = QHBoxLayout()
        hbox.addStretch()
        hbox.addWidget(self.__toolbar)
        hbox.addStretch()
        hbox.addWidget(self.__but_exp)
        self.layout.addLayout(hbox, 2, 0)
        self.setLayout(self.layout)
        # Figure
        self.__figure.subplots_adjust( left=0.1, right=0.9, \
                                       bottom=0.1, top=0.95, \
                                       hspace=0.25 )
        if self.__mode is None :
            self.__update_axes()
            self.__but_exp.setEnabled(False)
        else :
            self.__draw_mode()
            self.__but_exp.setEnabled(True)
    #--------------------------------------------------------------------
    def __update_u(self) :
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __update_S(self) :
        item_checked = False
        for case in self.__S_sel.items :
            if self.__S_sel.isChecked(case) :
               item_checked = True
        if not item_checked :
            self.__S_sel.setEnabled(False)
            self.__stresses = False
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __update_P(self) :
        item_checked = False
        for case in self.__P_sel.items :
            if self.__P_sel.isChecked(case) :
               item_checked = True
        if not item_checked :
            self.__P_sel.setEnabled(False)
            self.__Poynting = False
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __change_u(self) :
        self.__disp = not self.__disp
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __change_S(self) :
        self.__stresses = not self.__stresses
        self.__Poynting = False
        self.__P_sel.setEnabled(False)
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __change_P(self) :
        self.__stresses = False
        self.__Poynting = not self.__Poynting
        self.__S_sel.setEnabled(False)
        if self.__mode is None : self.__update_axes()
        else : self.__draw_mode()
    #--------------------------------------------------------------------
    def __update_fields(self, zmin="auto", zmax="auto") :
        fluid_ths, fluid_bhs = self.__mode.fluid_half_spaces
        Z_mm = self.__mode.Z
        w_mm = Z_mm[-1]-Z_mm[0]
        if fluid_ths :
            self.__z_min = Z_mm[0] - self.C_FLUID_HS*w_mm
            if fluid_bhs : # Fluid/Fluid
                self.__z_max = Z_mm[-1] + self.C_FLUID_HS*w_mm
            else : # Fluid/Empty
                self.__z_max = Z_mm[-1] + \
                               self.C_EMPTY_HS*w_mm*(1+self.C_FLUID_HS)
        elif fluid_bhs : # Empty/Fluid
            self.__z_max = Z_mm[-1] + self.C_FLUID_HS*w_mm
            self.__z_min = Z_mm[0] - \
                           self.C_EMPTY_HS*w_mm*(1+self.C_FLUID_HS)
        else : # Empty/Empty
            correc = self.C_EMPTY_HS*w_mm*(1+self.C_EMPTY_HS)
            self.__z_min = Z_mm[0] - correc
            self.__z_max = Z_mm[-1] + correc
        if zmin != "auto" :
            self.__z_min = max(zmin, self.__z_min)
        if zmax != "auto" :
            self.__z_max = min(zmax, self.__z_max)
        dico_mode = self.__mode.export(None, dz_mm=self.DZ_MM, \
                                         c_half_space=self.C_FLUID_HS)
        self.__data = dict()
        self.__data["z_mm"] = dico_mode["z_mm"]
        for k,v in dico_mode.items() :
            k2 = k[:2]
            if k2 in self.K2 :
                self.__data[k2] = dico_mode[k]
        for k,v in dico_mode.items() :
            k3 = k[:3]
            if k3 in self.K3 :
                self.__data[k3] = dico_mode[k]
        # Default result directory
        self.__specific_res_dir = False
        self.__res_dir = os.getcwd()
        try :
            mw = self.parent().mw
            self.__res_dir = mw.geom_frm.structure_file_path
            self.__res_dir = self.__res_dir.replace(".txt","")
            if os.path.isfile(self.__res_dir) :
                msg = "*** Mode_Shape_Figure.__update_fields ***" + \
                      f"\n\t{self.__res_dir} is a file!"
                raise ValueError(msg)
            self.__specific_res_dir = True
            print(f"*** Default Result Directory:\n\t{self.__res_dir}")
        except Exception as err :
            print(f"*** No Specific Default Result Directory:\n\t{err}")
        return
    #--------------------------------------------------------------------
    def __update_axes(self, draw=True) :
        for ax in [self.__xy_u, self.__xy_S, self.__z__u, self.__z_SP] :
            ax.clear()
        self.__xy_u.set_title("In plane", **self.DEFONT)
        self.__z__u.set_title("Out of plane", **self.DEFONT)
        self.__z__u.set_xlabel("Vertical Position $z$ [mm]", \
                               **self.DEFONT)     
        self.__xy_u.set_ylabel("In-plane Displacements [µm]", \
                               **self.DEFONT)      
        self.__z__u.set_ylabel("Vertical Displacement $u_z$ [µm]", \
                               **self.DEFONT)
        self.__xy_u.grid() ; self.__z__u.grid()
        if self.__stresses :     
            self.__xy_S.set_ylabel("In-plane Stresses [MPa]", \
                               **self.DEFONT) 
            self.__xy_S.yaxis.set_label_position("right")     
            self.__z_SP.set_ylabel("Out-of-plane Stresses [MPa]", \
                               **self.DEFONT)
            self.__z_SP.yaxis.set_label_position("right")
            self.__xy_S.grid(color="pink")
            self.__z_SP.grid(color="pink")
        elif self.__Poynting :     
            self.__xy_S.set_ylabel( \
                          "(Poynting vector) [W/mm²]", \
                          **self.DEFONT)  
            self.__xy_S.yaxis.set_label_position("right")    
            self.__z_SP.set_ylabel("")
            self.__z_SP.yaxis.set_label_position("right")
            self.__xy_S.grid(color="pink")
            self.__z_SP.grid(color="pink")
        else : # None
            self.__xy_S.set_yticks([])
            self.__z_SP.set_yticks([])
        if draw : self.__canvas.draw()
    #--------------------------------------------------------------------
    def set_mode(self, mode) :
        if mode is None :
            self.__but_exp.setEnabled(False)
            self.__update_axes()
        else :
            self.__but_exp.setEnabled(True)
            self.__mode = mode
            self.__update_fields()
            self.__draw_mode()
    #--------------------------------------------------------------------
    def __draw_mode(self) :
        self.__update_axes(draw=False)
        Z_mm = self.__data["z_mm"]
        z0, ze = self.__mode.Z[0],self.__mode.Z[-1]
        thck = ze-z0
        z0,ze = z0-thck,ze+thck
        b,e = (Z_mm>=z0).argmax(), (Z_mm>ze).argmax()-1
        # Displacements
        Ux, Uy, Uz = self.U_LABELS
        u_min,u_max = np.inf,-np.inf
        leg_xy, leg_z = False,False
        if self.__disp and self.__u_sel.isChecked(Ux):
            leg_xy = True
            Y = self.__data[Ux].real
            self.__xy_u.plot(Z_mm, Y, "-b", label = "Re($u_x$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
            Y = self.__data[Ux].imag
            self.__xy_u.plot(Z_mm, Y, "--b", label = "Im($u_x$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
        if self.__disp and self.__u_sel.isChecked(Uy):
            leg_xy = True
            Y = self.__data[Uy].real
            self.__xy_u.plot(Z_mm, Y, "-r", label = "Re($u_y$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
            Y = self.__data[Uy].imag
            self.__xy_u.plot(Z_mm, Y, "--r", label = "Im($u_y$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
        if self.__disp and self.__u_sel.isChecked(Uz):
            leg_z = True
            Y = self.__data[Uz].real
            self.__z__u.plot(Z_mm, Y, "-b", label = "Re($u_z$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
            Y = self.__data[Uz].imag
            self.__z__u.plot(Z_mm, Y, "--b", label = "Im($u_z$)")
            u_min,u_max = min(u_min,Y[b:e].min()),max(u_max,Y[b:e].max())
        if leg_xy : self.__xy_u.legend(loc="upper left")
        if leg_z :  self.__z__u.legend(loc="upper left") 
        self.__xy_u.set_xlim(self.__z_min, self.__z_max)    
        self.__z__u.set_xlim(self.__z_min, self.__z_max)
        if u_min == np.inf : u_min,u_max = -1,1 # No displacement drawn
        u_min,u_max = 1.05*u_min-0.05*u_max,1.05*u_max-0.05*u_min
        self.__xy_u.set_ylim(u_min,u_max)    
        self.__z__u.set_ylim(u_min,u_max)
        # Stresses
        if self.__stresses :
            Sxx,Sxy,Syy,Sxz,Syz,Szz = self.S_LABELS
            s_min,s_max = np.inf,-np.inf
            leg_xy, leg_z = False,False
            if self.__S_sel.isChecked(Sxx):
                leg_xy = True
                Y = self.__data[Sxx].real
                self.__xy_S.plot(Z_mm, Y, "-c", \
                                 label = r"Re($\sigma_{xx}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Sxx].imag
                self.__xy_S.plot(Z_mm, Y, "--c", \
                                 label = r"Im($\sigma_{xx}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if self.__S_sel.isChecked(Syy):
                leg_xy = True
                Y = self.__data[Syy].real
                self.__xy_S.plot(Z_mm, Y, "-m", \
                                 label = r"Re($\sigma_{yy}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Syy].imag
                self.__xy_S.plot(Z_mm, Y, "--m", \
                                 label = r"Im($\sigma_{yy}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if self.__S_sel.isChecked(Sxy):
                leg_xy = True
                Y = self.__data[Sxy].real
                self.__xy_S.plot(Z_mm, Y, "-g", \
                                 label = r"Re($\sigma_{xy}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Sxy].imag
                self.__xy_S.plot(Z_mm, Y, "--g", \
                                 label = r"Im($\sigma_{xy}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if self.__S_sel.isChecked(Sxz):
                leg_z = True
                Y = self.__data[Sxz].real
                self.__z_SP.plot(Z_mm, Y, "-c", \
                                 label = r"Re($\sigma_{xz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Sxz].imag
                self.__z_SP.plot(Z_mm, Y, "--c", \
                                 label = r"Im($\sigma_{xz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if self.__S_sel.isChecked(Syz):
                leg_z = True
                Y = self.__data[Syz].real
                self.__z_SP.plot(Z_mm, Y, "-m", \
                                 label = r"Re($\sigma_{yz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Syz].imag
                self.__z_SP.plot(Z_mm, Y, "--m", \
                                 label = r"Im($\sigma_{yz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if self.__S_sel.isChecked(Szz):
                leg_z = True
                Y = self.__data[Szz].real
                self.__z_SP.plot(Z_mm, Y, "-g", \
                                 label = r"Re($\sigma_{zz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
                Y = self.__data[Szz].imag
                self.__z_SP.plot(Z_mm, Y, "--g", \
                                 label = r"Im($\sigma_{zz}$)")
                s_min,s_max = min(s_min,Y[b:e].min()), \
                              max(s_max,Y[b:e].max())
            if leg_xy : self.__xy_S.legend(loc="upper right")
            if leg_z :  self.__z_SP.legend(loc="upper right") 
            s_min,s_max = 1.05*s_min-0.05*s_max,1.05*s_max-0.05*s_min
            self.__xy_S.set_ylim(s_min,s_max)    
            self.__z_SP.set_ylim(s_min,s_max)
        # Poynting vector
        if self.__Poynting :
            Px, Py, Pz, Ve, Et = self.P_LABELS
            p_min,p_max = np.inf,-np.inf
            e_min,e_max = 0.0,-np.inf
            leg_xy, leg_z = False,False
            if self.__P_sel.isChecked(Px):
                leg_xy = True
                Y = self.__data[Px].real
                self.__xy_S.plot(Z_mm, Y, "-c", label = "$P_x$")
                p_min,p_max = min(p_min,Y[b:e].min()), \
                              max(p_max,Y[b:e].max())
            if self.__P_sel.isChecked(Py):
                leg_xy = True
                Y = self.__data[Py].real
                self.__xy_S.plot(Z_mm, Y, "-m", label = "$P_y$")
                p_min,p_max = min(p_min,Y[b:e].min()), \
                              max(p_max,Y[b:e].max())
            if self.__P_sel.isChecked(Pz):
                leg_xy = True
                Y = self.__data[Pz].real
                self.__xy_S.plot(Z_mm, Y, "-g", label = "$P_z$")
                p_min,p_max = min(p_min,Y[b:e].min()), \
                              max(p_max,Y[b:e].max())
            if self.__P_sel.isChecked(Ve):
                leg_z = True
                Y = self.__data[Ve]
                self.__z_SP.plot(Z_mm, Y[0,:], "-r", linewidth=1.8, \
                                 label = "$v_{ex}$")
                self.__z_SP.plot(Z_mm, Y[1,:], ":r", linewidth=1.0, \
                                 label = "$v_{ey}$")
                e_min,e_max = min(e_min,Y[:,b:e].min()), \
                              max(e_max,Y[:,b:e].max())      
                self.__z_SP.set_ylabel( \
                                  "Local Energy Velocity [mm/µs]", \
                                  **self.DEFONT)
                self.__z_SP.yaxis.set_label_position("right")
            if self.__P_sel.isChecked(Et):
                leg_z = True
                Y = self.__data[Et]
                self.__z_SP.plot(Z_mm, Y, "-r", label = "$e_{tot}$")
                e_min,e_max = min(e_min,Y[b:e].min()), \
                              max(e_max,Y[b:e].max())      
                self.__z_SP.set_ylabel( \
                                  "Volume Energy [µJ/mm³]", \
                                  **self.DEFONT)
                self.__z_SP.yaxis.set_label_position("right")
            if leg_xy :
                self.__xy_S.legend(loc="upper right") 
                p_min = 1.05*p_min-0.05*p_max
                p_max = 1.05*p_max-0.05*p_min
                self.__xy_S.set_ylim(p_min,p_max)    
            if leg_z :
                self.__z_SP.legend(loc="upper right")
                e_min = 1.05*e_min-0.05*e_max
                e_max = 1.05*e_max-0.05*e_min
                self.__z_SP.set_ylim(e_min,e_max)
        # Interfaces between layers
        for ax in (self.__xy_u,self.__z__u) :
            for z in self.__mode.Z :
                ax.plot([z,z], [u_min,u_max], "k-", linewidth=0.5)
        self.__canvas.draw()
    #--------------------------------------------------------------------
    def __export(self) :
        if self.__specific_res_dir :
            if not os.path.isdir(self.__res_dir) :
                os.mkdir(self.__res_dir)
        def_file_path = os.path.join(self.__res_dir, self.__mode.name)
        file_path,_ = QFileDialog.getSaveFileName( \
                            self,\
                            "Export mode shape as:", \
                            def_file_path, \
                            "NPZ-file (numpy) (*.npz);;" + \
                            "MAT-file (Matlab) (*.mat)")
        if file_path == "" : return False # Abort saving
        if file_path.endswith(".mat") :
            fmt = "Matlab"
        elif file_path.endswith(".npz") :
            fmt = "numpy"
        else :
            ext = "."+file_path.split(".")[-1]
            msg = f"Unknown extension {ext}. Mode shape not saved"
            qmb = QMessageBox(self) 
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText(msg.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Ok);
            rep = qmb.exec_()
            return            
        self.__specific_res_dir = False
        self.__res_dir = os.path.dirname(file_path)
        self.__mode.export(file_path, file_format = fmt)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Dispersion_Curve_Frame(Mode_Frame):
    COLORS = ('#2080B0', '#FF8000', '#30A030', '#A82828', '#A068C0', \
              '#905850', '#E078C0', '#808080', '#C0C020', '#18C0D0', \
              '#0000F0', '#CC0000', '#007000', '#808000', '#00A0A0', \
              '#FF00FF', '#C05000', '#60A060', '#6060A0', '#A06060')*5
    MARGIN = 0.02 # For axis limits
    __nb_max_plot = 200
    #--------------------------------------------------------------------
    def __init__(self, mainWindow):
        Mode_Frame.__init__(self, mainWindow)
        self.__next_iter = QTimer(self)
        self.__next_iter.timeout.connect(self.__next_iteration)
        self.__iter_data = dict() # To store the computation data
        # Pop-up windows
        self.__popup = None # Previous computations
        self.__inter = None # Interrupt computation
        # Additional button
        self.but_open = QPushButton(\
                              "Open Started/Finished Computation", self)
        self.but_open.released.connect(self.__open_disp_comp)
        # Computation parameters
        self.__new_computation = None
        self.__disp_file_fmt = None
        self.__previous_case = -1
        self.__val_min = Entry(self, "", "", labelW=40, unitW=60)
        self.__val_max = Entry(self, "", "", labelW=40, unitW=60)
        self.__nb_int = Entry(self, "Nb int.", "", fmt="d",
                              labelW=50, unitW=1) # Integers
        self.__go_btn = QPushButton("", self)
        self.__go_btn.released.connect(self.__start_computing)
        self.__vphi_valmax = Entry(self, "", "", labelW=90, unitW=70)
        self.__att_valmax = Entry(self, r"Att. max.", "Neper/m", \
                                   labelW=90, unitW=80)
        for wg in (self.__val_min, self.__val_max, self.__nb_int, \
                   self.__vphi_valmax, self.__att_valmax) :
            wg.editingFinished.connect(self.__check_loop_param)
        # Display zone
        self.__edit_fmin = Entry(self, r"$f_{\min}$", "MHz", \
                                 labelW=30, unitW=40)
        self.__edit_fmax = Entry(self, r"$f_{\max}$", "MHz", \
                                 labelW=30, unitW=40)
        for wg in (self.__edit_fmin, self.__edit_fmax) :
            wg.editingFinished.connect(self.__update_F_axis)
        uw = 80
        self.__edit_vphimin = Entry(self, r"$v_{\phi\min}$", "mm/µs", \
                                 unitW=uw)
        self.__edit_vphimax = Entry(self, r"$v_{\phi\max}$", "mm/µs", \
                                 unitW=uw)
        for wg in (self.__edit_vphimin, self.__edit_vphimax) :
            wg.editingFinished.connect(self.__update_Vphi_axis)
        self.__edit_amin = Entry(self, r"$a_{\min}$", "Neper/m", \
                                 unitW=uw)
        self.__edit_amax = Entry(self, r"$a_{\max}$", "Neper/m", \
                                 unitW=uw)
        for wg in (self.__edit_amin, self.__edit_amax) :
            wg.editingFinished.connect(self.__update_Att_axis)
        self.__edit_vemin = Entry(self, r"$v_{e\min}$", "mm/µs", \
                                 unitW=uw)
        self.__edit_vemax = Entry(self, r"$v_{e\max}$", "mm/µs", \
                                 unitW=uw)
        for wg in (self.__edit_vemin, self.__edit_vemax) :
            wg.editingFinished.connect(self.__update_Ve_axis)
        self.__figure  = Figure()
        self.__ax_vphi = self.__figure.add_subplot(3,1,1)
        self.__ax_att = self.__figure.add_subplot(3,1,2)
        self.__ax_ve = self.__figure.add_subplot(3,1,3)
        self.__canvas  = FigureCanvas(self.__figure)
        # User Interface Initialization 
        self.__initUI() 
    #--------------------------------------------------------------------
    def __initUI(self):
        # Computation parameters
        self.layout.addWidget(self.but_open, 1, 0, 1, 1)
        self.__update_but_open()
        self.layout.addWidget(QLabel("Computation parameters:",self), \
                              0, 2, 1, 1)
        self.layout.addWidget(self.__val_min, 0, 3, 1, 1)
        self.layout.addWidget(self.__val_max, 1, 3, 1, 1)
        self.layout.addWidget(self.__nb_int, 0, 4, 1, 1)
        self.layout.addWidget(self.__go_btn, 1, 4, 1, 1)
        self.setGoButton(False)
        self.layout.addWidget(self.__vphi_valmax,0, 5, 1, 1)
        self.layout.addWidget(self.__att_valmax,1, 5, 1, 1)
        # Display zone
        for wg in (self.__edit_fmin, self.__edit_fmax, \
                   self.__edit_vphimin,self.__edit_vphimax, \
                   self.__edit_amin, self.__edit_amax, \
                   self.__edit_vemin, self.__edit_vemax, \
                   self.__go_btn) :
            wg.setEnabled(False)
        freq_layout = QHBoxLayout()
        freq_layout.addWidget(self.__edit_fmin)
        freq_layout.addStretch()
        freq_layout.addWidget(self.__edit_fmax)
        vphi_layout = QVBoxLayout()
        vphi_layout.addWidget(self.__edit_vphimax)
        vphi_layout.addStretch()
        vphi_layout.addWidget(self.__edit_vphimin)
        vphi_layout.addWidget(self.__edit_amax)
        vphi_layout.addStretch()
        vphi_layout.addWidget(self.__edit_amin)
        vphi_layout.addWidget(self.__edit_vemax)
        vphi_layout.addStretch()
        vphi_layout.addWidget(self.__edit_vemin)
        VSpace = QFrame(self)
        VSpace.setFixedSize(50,50)
        vphi_layout.addWidget(VSpace)
        self.layout.addLayout(vphi_layout, 2, 0, 1, 1)
        self.layout.addWidget(self.__canvas, 2, 1, 1, 5)
        self.layout.addLayout(freq_layout, 3, 1, 1, 5)
        self.__canvas.setFixedHeight(700)
        self.__canvas.setFixedWidth(1000)
        self.__figure.subplots_adjust(left = 0.1, right=0.98, \
                                      bottom = 0.1, top = 0.98, \
                                      hspace = 0.25)
        self.__update_axes()
        self.change_fixed_parameter_name(self.prm_cmb.currentIndex()) 
    #--------------------------------------------------------------------
    def reinit(self) :
        # reinit Tab because the plate has changed
        for wg in (self.__val_min, self.__val_max, self.__nb_int, \
                   self.__vphi_valmax, self.__att_valmax) :
            wg.value = None
        self.__update_axes(clear=True)
        self.setGoButton(False)
        self.__update_but_open()
        for wg in (self.__edit_fmin, self.__edit_fmax, \
                   self.__edit_vphimin,self.__edit_vphimax, \
                   self.__edit_amin, self.__edit_amax, \
                   self.__edit_vemin, self.__edit_vemax) :
            wg.value = None
            wg.setEnabled(False)
    #--------------------------------------------------------------------
    def __update_but_open(self) :
        """Are Previous sets of parameters available?"""
        cur_fold = self.mw.results_folder
        if not isinstance(cur_fold, str) or \
           not os.path.isdir(cur_fold) :
            print( "Dispersion_Curve_Frame.__update_but_open\n\t" + \
                  f"Folder '{cur_fold}' does not exist." )
            self.but_open.setEnabled(False)
            self.__avail_disp_param = None
            return
        case = self.__previous_case
        if case == 0 : # Fixed frequency
            # "fixed-f_{nb_val}values_{val_min}-{val_max}MHz" +
            # "_Vph-max{vphi_max}mpms_Att-max{att_max}pm"
            sep0 = "fixed-f_"
            sep3 = "MHz_Vph-max"
            sep4 = "mpms_Att-max"
            self.__head_disp_param = "Frequency range [MHz]" + \
                                     20*" " + \
                                     " | Max. Ph. Vel." + \
                                     " | Max. Attenuation"
            self.__fmt_disp_param = "{:4d} values from {:10.6f} to " + \
                                    "{:10.6f} | {:7.3f} mm/µs | " + \
                                    "{:7.3f} Neper/m "
        elif case == 1 : # Fixed wavenumber
            # "fixed-k_{nb_val}values_"{val_min}-{val_max}pmm" +
            # "_Vph-max{vphi_max}mpms_Att-max{att_max}pm"
            sep0 = "fixed-k_"
            sep3 = "pmm_Vph-max"
            sep4 = "mpms_Att-max"
            self.__head_disp_param = "Wavenumber range [/mm]" + \
                                     15*" " + \
                                     " | Max. Ph. Vel." + \
                                     " | Max. Attenuation"
            self.__fmt_disp_param = "{:4d} values from {:8.3f} to " + \
                                    "{:8.3f} | {:7.3f} mm/µs | " + \
                                    "{:7.3f} Neper/m "
        elif no_case == 2 : # Fixed phase velocity
            # "fixed-Vph_{nb_val}values_{val_min}-{val_max}mpms" +
            # "_F-max{freq_max}MHz_Att-max{att_max}pm"
            sep0 = "fixed-Vph_"
            sep3 = "mpms_F-max"
            sep4 = "MHz_Att-max"
            self.__head_disp_param = "Phase velocity range [mm/µs]" + \
                                     7*" " + \
                                     " | Max. Frequency" + \
                                     " | Max. Attenuation"
            self.__fmt_disp_param = "{:4d} values from {:7.3f} to " + \
                                    "{:7.3f} | {:10.6f} MHz | " + \
                                    "{:7.3f} Neper/m "
        sep1 = "values_"
        sep2 = "-"
        sep5 = "pm"
        Ld = [ nm for nm in os.listdir(cur_fold) if \
                               nm.startswith(sep0) and \
                               os.path.isdir(cur_fold+"/"+nm) ]
        if len(Ld) == 0 :
            self.but_open.setEnabled(False)
            self.__avail_disp_param = None
            return
        # Available sets of parameters
        adp = []
        b0 = len(sep0)
        for d in Ld :
            e0 = b0 + d[b0:].index(sep1)
            nb_val = int(d[b0:e0])
            b1 = e0 + len(sep1)
            e1 = b1 + d[b1:].index(sep2)
            val_min = float(d[b1:e1])
            b2 = e1 + len(sep2)
            e2 = b2 + d[b2:].index(sep3)
            val_max = float(d[b2:e2])
            b3 = e2 + len(sep3)
            e3 = b3 + d[b3:].index(sep4)
            val_max2 = float(d[b3:e3])
            b4 = e3 + len(sep4)
            e4 = b4 + d[b4:].index(sep5)
            att_max = float(d[b4:e4])
            adp.append([nb_val,val_min,val_max,val_max2,att_max])
        self.but_open.setEnabled(True)
        print("adp:\n" + "\n".join( \
                    [ self.__fmt_disp_param.format(*p) \
                                       for p in adp ] ) )
        self.__avail_disp_param = adp
    #--------------------------------------------------------------------
    def __open_disp_comp(self) :
        """Launch pop-up window to select previous set of parameters."""
        adp = self.__avail_disp_param
        head = self.__head_disp_param
        items = [ self.__fmt_disp_param.format(*p) for p in adp ]
        self.mw.setEnabled(False)
        self.__popup = PopUp_Select_Item(head, items, self)
        self.__popup.selected_item.connect(self.__selected_disp_param)
        self.__popup.show()
    #--------------------------------------------------------------------
    def __selected_disp_param(self, no) :
        sel_item = self.__avail_disp_param[no-1]
        print(sel_item)
        self.__val_min.value = sel_item[1]
        self.__val_max.value = sel_item[2]
        self.__nb_int.value = sel_item[0]-1
        self.__vphi_valmax.value = sel_item[3]
        self.__att_valmax.value = sel_item[4]
        self.mw.setEnabled(True)
        self.__check_loop_param() 
    #--------------------------------------------------------------------
    def __update_axes(self, clear=True) :
        ax_vphi,ax_att,ax_ve = self.__ax_vphi, self.__ax_att, \
                               self.__ax_ve
        if clear :
            for ax in (ax_vphi,ax_att,ax_ve) :
                ax.clear() ; ax.grid(True)
            ax_vphi.set_ylabel(r"Phase velocity $v_{\phi}$")
            ax_att.set_ylabel(r"Attenuation $a$")
            ax_ve.set_ylabel(r"Energy velocity $v_{e}$")
            ax_ve.set_xlabel(r"Frequency $f$ [MHz]")
        self.__canvas.draw()
    #--------------------------------------------------------------------
    def __set_F_range(self, vmin, vmax) :
        self.__edit_fmin.value = vmin
        self.__edit_fmax.value = vmax
        self.__update_F_axis()
    #--------------------------------------------------------------------
    def __update_F_axis(self) :
        vmin = self.__edit_fmin.value
        vmax = self.__edit_fmax.value
        if vmin is None or vmax is None : return
        a = self.MARGIN
        b = 1+a
        vmin,vmax = b*vmin-a*vmax, b*vmax-a*vmin
        for ax in (self.__ax_vphi, self.__ax_att, self.__ax_ve) :
            ax.set_xlim(vmin, vmax)
        self.__canvas.draw()
    #--------------------------------------------------------------------
    def __set_Vphi_range(self, vmin, vmax) :
        self.__edit_vphimin.value = vmin
        self.__edit_vphimax.value = vmax
        self.__update_Vphi_axis()
    #--------------------------------------------------------------------
    def __update_Vphi_axis(self) :
        vmin = self.__edit_vphimin.value
        vmax = self.__edit_vphimax.value
        if vmin is None or vmax is None : return
        a = self.MARGIN
        b = 1+a
        self.__ax_vphi.set_ylim( b*vmin-a*vmax, b*vmax-a*vmin)
        self.__canvas.draw()
    #--------------------------------------------------------------------
    def __set_Att_range(self, vmin, vmax) :
        self.__edit_amin.value = vmin
        self.__edit_amax.value = vmax
        self.__update_Att_axis()
    #--------------------------------------------------------------------
    def __update_Att_axis(self) :
        vmin = self.__edit_amin.value
        vmax = self.__edit_amax.value
        if vmin is None or vmax is None : return
        a = self.MARGIN
        b = 1+a
        self.__ax_att.set_ylim( b*vmin-a*vmax, b*vmax-a*vmin) 
        self.__canvas.draw()
    #--------------------------------------------------------------------
    def __set_Ve_range(self, vmin, vmax) :
        self.__edit_vemin.value = vmin
        self.__edit_vemax.value = vmax
        self.__update_Ve_axis()
    #--------------------------------------------------------------------
    def __update_Ve_axis(self) :
        vmin = self.__edit_vemin.value
        vmax = self.__edit_vemax.value
        if vmin is None or vmax is None : return
        a = self.MARGIN
        b = 1+a
        self.__ax_ve.set_ylim( b*vmin-a*vmax, b*vmax-a*vmin) 
        self.__canvas.draw()        
    #--------------------------------------------------------------------
    def change_fixed_parameter_name(self, case) :
        print(Mode_Frame.CASES[case])
        if case == self.__previous_case : return # No Change
        if self.__att_valmax.value is None :
            self.__att_valmax.value = 70.0
        if case == 0 : # Fixed frequency
            self.__val_min.name = r"$f_{\min}$"
            self.__val_min.unit = "MHz"
            self.__val_min.value = None
            self.__val_max.name = r"$f_{\max}$"
            self.__val_max.unit = "MHz"
            self.__val_max.value = None
            self.__vphi_valmax.name = r"$v_{\phi}$ max."
            self.__vphi_valmax.unit = "mm/µs"
            if self.__vphi_valmax.value is None or \
               self.__previous_case == 2 :
                self.__vphi_valmax.value = 10.0
        elif case == 1 : # Fixed wawenumber
            self.__val_min.name = r"$k_{\min}$"
            self.__val_min.unit = "rad/mm"
            self.__val_min.value = None
            self.__val_max.name = r"$k_{\max}$"
            self.__val_max.unit = "rad/mm"
            self.__val_max.value = None
            self.__vphi_valmax.name = r"$v_{\phi}$ max."
            self.__vphi_valmax.unit = "mm/µs"
            if self.__vphi_valmax.value is None or \
               self.__previous_case == 2 :
                self.__vphi_valmax.value = 10.0
        elif case == 2 : # Fixed phase velocity
            self.__val_min.name = r"$v_{\phi\min}$"
            self.__val_min.unit = "mm/µs"
            self.__val_min.value = None
            self.__val_max.name = r"$v_{\phi\max}$"
            self.__val_max.unit = "mm/µs"
            self.__val_max.value = None
            self.__vphi_valmax.name = r"$f$ max."
            self.__vphi_valmax.unit = "MHz"
            if self.__vphi_valmax.value is None or \
               self.__previous_case in (0,1) :
                self.__vphi_valmax.value = 1.0
        self.__previous_case = case # For the next call
        self.__check_loop_param()
        self.__update_but_open()
    #--------------------------------------------------------------------
    def setGoButton(self, ok) :
        btn = self.__go_btn
        if isinstance(ok, bool):
            if ok :
                btn.setText("Start/Continue Computation")
                btn.setStyleSheet(self.mw.OK_COLOR)
            else :
                btn.setText("parameterization in progress...")
                btn.setStyleSheet(self.mw.KO_COLOR)
            font = QFont()
            font.setBold(ok)
            btn.setFont(font)
            btn.setEnabled(ok)
            return
        # ok must be a number between 0 and 1
        level = np.clip(ok, 0, 1)
        btn.setText(f"Computation: {100*level:.1f}% done")
        btn.setStyleSheet(self.mw.PR_COLOR)
        font = QFont()
        font.setBold(True)
        btn.setFont(font)
        btn.setEnabled(False)
    #--------------------------------------------------------------------
    @property
    def case(self) :
        no_case = self.prm_cmb.currentIndex()
        case = Mode_Frame.CASES[no_case].replace("Fixed ","")
        return no_case,case
    #--------------------------------------------------------------------
    def __check_loop_param(self) :
        no_case,case = self.case
        self.__param = {"Fixed Parameter" : case}
        par_min = self.__val_min.value
        if par_min is None :
            self.setGoButton(False) ; return
        par_max = self.__val_max.value
        if par_max is None :
            self.setGoButton(False) ; return
        if par_min > par_max :
            par_min,par_max = par_max,par_min
            self.__val_min.value = par_min
            self.__val_max.value = par_max
        nb_int =  self.__nb_int.value
        if nb_int is None :
            self.setGoButton(False) ; return
        if nb_int <= 1 :
            self.__nb_int.value = None
            self.setGoButton(False) ; return
        vmax = self.__vphi_valmax.value
        if vmax is None :
            self.setGoButton(False) ; return
        if not isinstance(vmax, float) or vmax <=0 :
            print(f"Incorrect vmax {vmax}")
            self.__vphi_valmax.value = None
            self.setGoButton(False) ; return
        amax = self.__att_valmax.value
        if amax is None :
            self.setGoButton(False) ; return
        if not isinstance(amax, float) or amax <=0 :
            print(f"Incorrect amax {amax}")
            self.__att_valmax.value = None
            self.setGoButton(False) ; return
        Vval = np.linspace(par_min, par_max, nb_int+1)
        if np.abs(Vval[0]) < 1e-7*np.abs(Vval[1]) :
            Vval = Vval[1:] # Zero value removed
        self.__param.update({case+" values": Vval, "Max. Att.": amax})
        if no_case <= 1 : # Fixed Frequency or Fixed Wavenumber
            self.__param["Max. Phase Velocity"] = vmax
        else : # Fixed slowness
            self.__param["Max. Frequency"] = vmax
        # print("Computation parameters:", self.__param)
        self.setGoButton(True) ; return
    #--------------------------------------------------------------------
    def __start_computing(self) :
        self.mw.setEnabled(False) # Blocks interactivity
                        # No action on application during computation
        msg = "*** Dispersion_Curve_Frame.__compute ***\n\t" + \
              "{} case not yet implemented."
        no_case, case = self.case
        key_vals = case + " values"
        Vval = self.__param[key_vals]
        nb_val = Vval.shape[0]
        if   nb_val <=   9 : nb_iter =  1
        elif nb_val <=  20 : nb_iter =  5
        elif nb_val <=  40 : nb_iter = 10
        elif nb_val <= 100 : nb_iter = 20
        elif nb_val <= 200 : nb_iter = 50
        else : nb_iter = 100
        if nb_val <= self.__nb_max_plot :
            plot_step = 1
        else :
            plot_step = (nb_val//self.__nb_max_plot) + 1
        scale = np.linspace(0.0, 1.0, nb_iter+1).round(4)
        ratios = np.arange(1,nb_val+1)/nb_val
        indexes = [0] + \
                  [ np.argmax(ratios>d) for d in scale[1:-1] ] + \
                  [nb_val]
        L_Vval = [ Vval[b:e] for b,e in zip(indexes[:-1],indexes[1:]) ]
        self.__iter_data = {"no case": no_case, "nb iter":nb_iter, \
                            "no iter": -1, "scale" : scale, \
                            "List of value vectors": L_Vval, \
                            "Plot step": plot_step}
        self.__update_axes(clear=True)
        # **************** Fixed Frequency ****************
        if no_case == 0 :
            self.__set_F_range(Vval[0], Vval[-1])
            vmax = self.__param["Max. Phase Velocity"] # mm/µs            
            self.__set_Vphi_range(0, vmax)
            self.__set_Ve_range(0, vmax)
            amax = self.__param["Max. Att."] # Neper/m
            self.__set_Att_range(-amax, amax)
            modes0 = None
            rel_err, rke = 1e-1, 10.0 #
            cc_min_comp = 0.95
            self.__iter_data.update({"vmax":vmax, "amax":amax, \
                                     "modes0":modes0, \
                                     "rel_err":rel_err, \
                                     "rke":rke , \
                                     "cc_min_comp":cc_min_comp})
            # subfolder name:
            sbf_name = f"fixed-f_{nb_val}values_" + \
                       f"{Vval[0]:.6f}-{Vval[-1]:.6f}MHz"
            sbf_name += f"_Vph-max{vmax:.3f}mpms_Att-max{amax:.3f}pm"
        # **************** Fixed Wavenumber ****************
        elif no_case == 1 :  
            print(msg.format("Fixed Wavenumber"))
            # subfolder name:
            sbf_name = "fixed-k_{nb_val}values_" + \
                       f"{Vval[0]:.3f}-{Vval[-1]:.3f}pmm"
            sbf_name += f"_Vph-max{vmax:.3f}mpms_Att-max{amax:.3f}pm"
            return                  
        # **************** Fixed Phase Velocity ****************
        elif no_case == 2 :  
            print(msg.format("Fixed Phase Velocity"))
            # subfolder name:
            sbf_name = "fixed-Vph_{nb_val}values_" + \
                       f"{Vval[0]:.3f}-{Vval[-1]:.3f}mpms"
            sbf_name += f"_F-max{vmax:.3f}MHz_Att-max{amax:.3f}pm"
            return
        else :
            print(f"Unexpected case no.{no_case}")
            return
        # ++++++++++++++++ folder containing files ++++++++++++
        cur_fold = self.mw.results_folder
        plate_pth = self.mw.geom_frm.structure_file_path
        last_change = time.ctime(os.path.getmtime(plate_pth))
        print(f"last plate change: {last_change}")
        plate_name = os.path.basename(plate_pth).replace(".txt","")
        sbf_pth = cur_fold + "/" + sbf_name
        hf_pth = sbf_pth + "/head_file.txt"
        self.__new_computation = True  
        if os.path.isdir(sbf_pth) :
            print(f"Folder\n{sbf_name}\nallready exists")
            if os.path.isfile(hf_pth) :
                with open(hf_pth, "r") as strm :
                    pp, lc = [ m.strip() for m in strm.readlines() ]
                if pp == plate_pth and lc == last_change :
                    print("Computation previously started")
                    self.__new_computation = False
                else :
                    print(f"{pp}\n***{plate_pth}\n" + \
                          f"{lc}\n***{last_change}")
        else :
            print(f"Folder\n{sbf_name}\nis created")
            os.mkdir(sbf_pth)
        if self.__new_computation :
            for nm in os.listdir(sbf_pth) :
                os.remove( sbf_pth + "/" + nm )
            with open(hf_pth, "w") as strm :
                strm.write(plate_pth+"\n"+last_change)
        self.__disp_file_fmt = sbf_pth + "/"+ plate_name + \
                               "_{0:03d}.pckl"        
        self.__inter = PopUp_Interrupt("Stop Reading/Computation!", self)
        self.__inter.stop.connect(self.__stop)
        self.__inter.show()  
        self.__update_display_and_next_iteration()
    #--------------------------------------------------------------------
    def __stop(self) :
        self.mw.setEnabled(True) # Restores interactivity
        for wg in (self.__edit_fmin, self.__edit_fmax, \
                   self.__edit_vphimin, self.__edit_vphimax, \
                   self.__edit_amin, self.__edit_amax, \
                   self.__edit_vemin, self.__edit_vemax) :
            wg.setEnabled(True)
        self.__next_iter.stop()  # Stop timer
        return
    #--------------------------------------------------------------------
    def __next_iteration(self) :
        # Importation here to avoid circular dependency
        from Plane_Guided_Mode import Plane_Guided_Mode
        P = self.__iter_data
        try :
            no_iter, nb_iter = P["no iter"], P["nb iter"]
        except Exception as err :
            msg = "Dispersion_Curve_Frame.__next_iteration :: error:"
            msg += f"\n\t{err}"
            print(msg)
            return
        if no_iter == nb_iter : # loop ended
            self.__inter.close()
            self.__stop()
            return
        # Path of the file containing the modes
        disp_file_path = self.__disp_file_fmt.format(no_iter+1)
        plate = self.mw.discretized_plate
        no_case,plot_step = P["no case"], P["Plot step"]
        idx = sum([len(e) for e in \
                   P["List of value vectors"][:no_iter] ])
        ax_vphi,ax_att,ax_ve = self.__ax_vphi, self.__ax_att, \
                               self.__ax_ve
        if os.path.exists(disp_file_path) :
        # +++++++++++++++ Computation already done ++++++++++++++++++++++
            with open(disp_file_path, "rb") as strm :
                print(f"Reading '{os.path.basename(disp_file_path)}'...")
                while True:
                    try:
                        saved_modes = pickle.load(strm)
                    except EOFError:
                        break # End of file
                    # Plot and restoration of already computed modes
                    modes0 = []
                    for dm,c in zip(saved_modes, self.COLORS) :
                        if dm is None :
                            modes0.append(None)
                            continue
                        m = Plane_Guided_Mode.from_dict(plate, dm)
                        modes0.append(m)
                        if idx%plot_step > 0 : continue # no plot
                        if m.is_a_true_guided_mode :
                            mrk = "."
                            ax_ve.plot([m.f], [m.Vex], mrk, color=c, \
                                   markersize=3.0)
                        else :
                            mrk = "x"
                        ax_vphi.plot([m.f], [m.Vphi], mrk, color=c, \
                                   markersize=3.0)
                        ax_att.plot([m.f], [-1e3*m.k.imag], mrk, \
                                    color=c, markersize=3.0)
                    idx += 1
            P["modes0"] = modes0 # Updating last modes
            self.__update_display_and_next_iteration()
            return
        # +++++++++++++++ Computation of modes +++++++++++++++++++++++++
        print(f"Computing '{os.path.basename(disp_file_path)}'...")
        temp_disp_file_path = os.path.dirname(disp_file_path) + "/" + \
                              "computation_in_progress.temp"
        # **************** Fixed Frequency ****************
        pickle_mode = "wb"
        if no_case == 0 : 
            Vval = 1.0e6*P["List of value vectors"][no_iter] # MHz -> Hz
            modes0, vmax, amax = P["modes0"], P["vmax"], 1e-3*P["amax"]
            cc_min_comp = P["cc_min_comp"]
            for f in Vval :
                # 1/ Modes computation
                new_modes = plate.modes_for_given_frequency(f) #, \
                                         #   rel_err = P["rel_err"], \
                                         #   rel_kappa_err = P["rke"] )                # 2/ follow-up of the modes
                modes1 = []
                if modes0 is not None :
                    for m0 in modes0 :
                        if m0 is None or len(new_modes)==0 :
                            modes1.append(None)
                        else :             
                            i1,cc1 = m0.nearest_mode_index(new_modes)
                            if cc1 > cc_min_comp : # Mode found
                                m1 = new_modes.pop(i1)
                                modes1.append(m1)
                            else :
                                modes1.append(None)
                # 3/ Sorting of the remaining modes by increasing
                #    phase velocity
                K = np.array([ m.k for m in new_modes ]) # mm^-1
                Vphi = np.array([ m.Vphi for m in new_modes ]) # mm/µs
                #    Rejection of modes that are too attenuated or have
                #    too high phase velocity
                Vidx = np.where( (np.abs(K.imag)<=amax)& \
                                 (Vphi<=vmax) )[0]
                new_modes = [new_modes[no] for no in Vidx]
                #    Modes to be treated
                #¨Vk,Vvphi = K[Vidx],Vphi[Vidx]
                Vvphi = Vphi[Vidx]
                #    Sorting by increasing phase velocity
                Vidx0 = Vvphi.argsort()
                new_modes = [new_modes[no] for no in Vidx0]
                modes0 = modes1 + new_modes
                # 4/ Plot of calculated modes
                saved_modes = []
                for no,(m,c) in enumerate(zip(modes0, self.COLORS)) :
                    if m is None :
                        saved_modes.append(None)
                        continue
                    if idx%plot_step == 0 :
                        if m.is_a_true_guided_mode :
                            mrk = "."
                            ax_ve.plot([m.f], [m.Vex], mrk, color=c, \
                                       markersize=3.0)
                        else :
                            mrk = "x"
                        ax_vphi.plot([m.f], [m.Vphi], mrk, color=c, \
                                       markersize=3.0)
                        ax_att.plot([m.f], [-1e3*m.k.imag], mrk, color=c, \
                                       markersize=3.0)
                    m.name = f"#{no}"
                    saved_modes.append(m.parameters_to_be_saved())
                # 5/ Saving modes
                with open(temp_disp_file_path, pickle_mode) as strm :
                    pickle.dump(saved_modes, strm, \
                                pickle.HIGHEST_PROTOCOL)
                pickle_mode = "ab" # append
                idx += 1
            P["modes0"] = modes0
            # To ensure that the computation is complete
            os.rename(temp_disp_file_path, disp_file_path)
        # **************** Fixed Wavenumber ****************
        elif no_case == 1 :    
            print("Fixed Wavenumber iterations : TO DO ***")
        # **************** Fixed Phase Velocity ****************
        elif no_case == 2 :  
            print("Fixed Phase Velocity iterations : TO DO ***")
        self.__update_display_and_next_iteration()
    #--------------------------------------------------------------------
    def __update_display_and_next_iteration(self) :
        self.__canvas.draw()
        self.__iter_data["no iter"] += 1
        no_iter = self.__iter_data["no iter"]
        level = self.__iter_data["scale"][no_iter]
        self.setGoButton(level)
        self.__next_iter.start(1)
    #--------------------------------------------------------------------
    @staticmethod
    def from_pickle_modes_to_csv(dir_path, csv_path=None, verbose=True,
                            F_digit=7, K_digit=7, V_digit=5, A_digit=5,
                            Neper_m=True, save_ReK=False) :
        """Export computed modes to dispersion curves in a CSV file."""
        from Plane_Guided_Mode import Plane_Guided_Mode
        if Neper_m : # attenuation per meter
            ca = 1.0
            ua = "m^-1"
        else : # attenuation per millimeter
            ca = 1.0e-3
            ua = "mm^-1"            
        if verbose :
            prt = print
        else :
            prt = lambda *args : None
        # Head file
        hfp = dir_path + "/head_file.txt"
        if os.path.isfile(hfp) :
            with open(hfp, "r") as strm :
                hf_text = strm.read()
            prt(f"head_file.txt contents:\n{hf_text}")
        # Note that the existence of 'head_file.txt' is not necessary
        LP = [ nm for nm in os.listdir(dir_path) if
               nm.endswith(".pckl") ]
        LP.sort()
        plate_name = LP[0][:-9]
        prt("Plate name:", plate_name)
        if csv_path is None :
            csv_path = dir_path + "/" + plate_name + ".csv"
        prt("CSV file name:", os.path.basename(csv_path))
        csv_text = ""
        dir_name = os.path.basename(dir_path)
        if dir_name.startswith("fixed-f") :
            head = "Frequency [MHz]"
            head_fmt = "{:."+str(F_digit-1)+"f}"
            if save_ReK :
                data = ";Wavenumber [mm^-1]"
                str_no_mode = ";;;;"
                fmt_no_true = ";{:."+str(K_digit-1)+"f}"
                fmt_mode = ";{:."+str(K_digit-1)+"f}"
                def from_dict(dict_mode, ca=ca) :
                    freq = dict_mode['f_MHz']
                    k = dict_mode['k_mm^-1']
                    energy_vel = dict_mode['Ve_mm/µs']
                    phase_vel = dict_mode['Vph_mm/µs']
                    att = -k.imag*1e3*ca
                    return freq, [k.real,phase_vel,att,energy_vel] 
            else :
                data = ""                
                str_no_mode = ";;;"
                fmt_no_true = ""
                fmt_mode = ""
                def from_dict(dict_mode, ca=ca) :
                    freq = dict_mode['f_MHz']
                    k = dict_mode['k_mm^-1']
                    energy_vel = dict_mode['Ve_mm/µs']
                    phase_vel = dict_mode['Vph_mm/µs']
                    att = -k.imag*1e3*ca
                    return freq, [phase_vel,att,energy_vel]
            data += f";Phase Vel. [mm/µs];Att. [{ua}]; En. Vel. [mm/µs]"
            fmt_no_true += ";{:."+str(V_digit-1)+"f};{:."+\
                          str(A_digit-1)+"f};"
            fmt_mode += ";{:."+str(V_digit-1)+"f};{:."+str(A_digit-1)+\
                       "f};{:."+str(V_digit-1)+"f}"              
        elif dir_name.startswith("fixed-k") :
            head = "Wavenumber [/mm]"
            head_fmt = "{:."+str(K_digit-1)+"f}"
            data = ";Frequency [MHz];Phase Vel. [mm/µs];" + \
                   "Att. [Neper/m]; En. Vel. [mm/µs]"
            str_no_mode = ";;;;"
            fmt_no_true = ";{:."+str(F_digit-1)+"f};{:."+ \
                          str(V_digit-1)+"f};{:."+str(A_digit-1)+"f};"
            fmt_mode = ";{:."+str(F_digit-1)+"f};{:."+ \
                          str(V_digit-1)+"f};{:."+str(A_digit-1)+ \
                          "f};{:."+str(V_digit-1)+"f}"
            def from_dict(dict_mode) :
                freq = dict_mode['f_MHz']
                k = dict_mode['k_mm^-1']
                energy_vel = dict_mode['Ve_mm/µs']
                phase_vel = dict_mode['Vph_mm/µs']
                att = -freq.imag/phase_vel*1e3
                return k, [freq,phase_vel,att,energy_vel]         
        elif dir_name.startswith("fixed-Vph") :
            head = "Phase Vel. [mm/µs]"
            head_fmt = "{:."+str(V_digit-1)+"f}"
            data = ";Frequency [MHz];Att. [Neper/m]; En. Vel. [mm/µs]"
            str_no_mode = ";;;"
            fmt_no_true = ";{:."+str(F_digit-1)+"f};{:."+ \
                          str(A_digit-1)+"f};"
            fmt_mode = ";{:."+str(F_digit-1)+"f};{:."+ \
                       str(A_digit-1)+"f};{:."+str(V_digit-1)+"f}"
            def from_dict(dict_mode) :
                freq = dict_mode['f_MHz']
                k = dict_mode['k_mm^-1']
                energy_vel = dict_mode['Ve_mm/µs']
                phase_vel = dict_mode['Vph_mm/µs']
                att = -k.imag*1e3
                return phase_vel, [freq,att,energy_vel]
        nb_modes = 0
        rows = []
        for nm in LP :
            if not nm.startswith(plate_name) :
                continue
            disp_file_path = dir_path + "/" + nm
            with open(disp_file_path, "rb") as strm :
                prt(f"Reading '{nm}'...")
                while True:
                    try:
                        saved_modes = pickle.load(strm)
                    except EOFError:
                        break # End of file
                    # Plot and restoration of already computed modes
                    fixed_value = None
                    modes = []
                    for dm in saved_modes :
                        if dm is None :
                            modes.append(None)
                            continue
                        fixed_value, data_m = from_dict(dm)
                        modes.append(data_m)
                    if fixed_value is None : continue
                    nb_modes = max(nb_modes, len(saved_modes))
                    row = head_fmt.format(fixed_value)
                    for m in modes :
                        if m is None :
                            row += str_no_mode
                        elif m[-1] is None :
                            row += fmt_no_true.format(*m[:-1])
                        else :
                            row += fmt_mode.format(*m)
                    rows.append(row)
        prt(f"{len(rows)} lines")
        text = "\n".join([head + nb_modes*data] + rows)
        with open( csv_path, "w") as strm :
            strm.write(text)
        return True, csv_path
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    import matplotlib.pyplot as plt
    import os
    print(f"Current dir.: '{os.getcwd()}'")
    res_dir = "../Data/Results/Plexiglas_4mm"
    msg = res_dir+"\n\t"
    lsd = [ n for n in os.listdir(res_dir) \
                          if os.path.isdir(res_dir+"/"+n) ]
    msg += "\n\t".join(["[{}] '{}'".format(i,n) for (i,n) \
                                  in enumerate(lsd,1) ])
    print(msg)
    rep = input("Choix -> ")
    dp = res_dir+"/"+lsd[int(rep)-1]
    test,csv_pth = \
        Dispersion_Curve_Frame.from_pickle_modes_to_csv(dp,V_digit=9, \
                                                        Neper_m=False, \
                                                        save_ReK=True)
    def flt(s) :
        if s == "" : return None
        return float(s)
    if test :
        data = []                       
        with open( csv_pth, "r") as strm :
            first_r = strm.readline()
            en_tete = first_r.strip().split(";")
            for r in strm.readlines() :
                try :
                    numbers = [ flt(x) for x in r.strip().split(";") ]
                    data.append(numbers)
                except :
                    print(f"Cannot interpret '{r}'")
        if "Wavenumber [mm^-1]" in first_r :
            p = 4
        else :
            p = 3
        nb_max = len(en_tete) // p
        nb_modes = np.array( [ len(r)//p for r in data ] )
        fd2_c = np.array( [ -1, 0,  1])/2
        fd2_1 = np.array( [ -3, 4, -1])/2
        plt.figure("Vitesses de groupe et d'énergie", figsize=(18,9))
        #axF, axE, axC = [ plt.subplot(1,3,i) for i in (1,2,3)]
        axF, axE = [ plt.subplot(2,1,i) for i in (1,2)]
        for no in range(1, nb_max+1) :
            idx = [0,p*no-2,p*no] # indexes of f, Vphi, Ve
            d = (nb_modes < no).argmin() 
            M = []
            for r in data[d:] :
                try :
                    m = [ r[i] for i in idx ]
                except :
                    print("no:", no, "; idx:", idx, "; d:", d)
                    print("len(r):", len(r))
                    raise ValueError
                if m[1] is None : break
                M.append(m)
            M = np.array(M)
            nv, _ = M.shape
            DF = np.empty( (nv,3) )
            DF[0,:] = fd2_1
            DF[1:-1,:] = fd2_c
            DF[-1,::-1] = -fd2_1
            Vk = 2*np.pi*M[:,0]/M[:,1]
            if nv > 3 :
                K = np.empty( (3,nv) )
                for r in (0,1,2) :
                    K[r,1:-1] = Vk[r:r+nv-2]
                K[:,0] = K[:,1]
                K[:,-1] = K[:,-2]
                dw = 2*np.pi*(M[1,0]-M[0,0])
                Sg = np.einsum("ij,ji->i", DF, K) / dw
                Cg = 1.0/Sg
                axE.plot(M[:,0], Cg, ":", linewidth=2.5)
            axF.plot(M[:,0], M[:,1], "-")
            axE.plot(M[:,0], M[:,2], "-", linewidth=1.0)
            #axC.plot(M[:,0], 100*(Cg-M[:,2])/M[:,2], "-")
        plt.subplots_adjust(0.05,0.1,0.99,0.99,0.25)
        for ax,vi,va in ((axF,0,10), (axE,0,10)) : #, (axC,-0.1,0.1)) :
            ax.grid()
            ax.set_ylim(vi,va)
            ax.set_xlabel("Frequency $f$ [MHz]", size=14)
        axF.set_ylabel("Phase Velocity $v_{\phi}$ [mm/µs]", size=14)
        axE.set_ylabel("Energy & Group Velocities [mm/µs]", size=14)
        #axC.set_ylabel("Difference [%]", size=14)
        axE.legend(["Group Velocity $v_g$","Energy Velocity $v_e$"])
        #axC.legend(["$(v_g-v_e)/v_e$"])
        plt.show()
                
        
