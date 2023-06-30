# Version 0.84 - 2023, June 29
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import os, platform
from PyQt5.QtWidgets import (QWidget, QFrame, QAction, QLineEdit, \
                             QPushButton, QFrame, QLabel, QComboBox, \
                             QGridLayout, QHBoxLayout, QVBoxLayout, \
                             QFileDialog, QMessageBox, QInputDialog)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QEvent, QRect, QSize, pyqtSignal
if __name__ == "__main__" :
    import sys
    root = os.path.abspath("..")
    sys.path.append(root)
    import TraFiC_init
from MaterialEdit import Material_App
from MaterialClasses import *
from USMultilayeredPlate import USMultilayeredPlate as USMP
from Small_Widgets import QVLine, QHLine
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Geometry_Frame(QWidget):
    THS = "Upper Half-Space"
    BHS = "Lower Half-Space"
    MAX_LAYER = 30
    RB = 1 # Row(s) before multilayer structure
    FONTSIZE = 11
    FONTNAME = "Arial"
    BUT_W = 250
    BUT_H = 28
    #----------------------------------------------------------------------
    def __init__(self, mainWindow):
        QWidget.__init__(self, mainWindow)
        self.mw = mainWindow
        self.__fileLoc = self.mw.PLATE_DIR
        self.__fileName = self.mw.UNTITLED
        # Layout to display layers
        self.grid = QGridLayout()
        self.grid.setAlignment(Qt.AlignVCenter)
        # First row(s)
        align = Qt.AlignLeft|Qt.AlignVCenter
        self.chk_but = QPushButton(self)
        self.setCheckButton(False) # Incomplete definition
        self.chk_but.released.connect(self.isCompletelyDefinedStructure)
        self.grid.addWidget( self.chk_but, 0, 0, 1, 2, align)
        # Layer frames
        self.__layers = [] # No Layer at the beginning
        lay_frm = Layer_Frame(self, self.THS, \
                              Layer_Frame.VACUUM, \
                              half_space=True)
        # (Automatically append to self.__layers)
        self.grid.addWidget( QHLine(), self.RB, 0, 1, 4)
        self.grid.addWidget( lay_frm, self.RB+1, 1, 1, 1, align)
        # Interface positions
        pos = pos_label()
        self.__pos = [ pos ]
        pos.setEnabled(False)
        for pos in self.__pos : pos.setEnabled(False)
        # Insert layer
        ins = QPushButton("Insert Layer", self)
        self.__ins = [ ins ]
        ins.released.connect(lambda : self.insertLayer(1))
        no_r = self.RB+2
        self.grid.addWidget( pos, no_r, 0, 1, 1, align)
        self.grid.addWidget( QHLine(), no_r, 1, 1, 2)
        self.grid.addWidget( ins, no_r, 3, 1, 1, align)
        # a single layer and lower space at the beginning
        layers_to_append = [ (None,None,None,False), \
                             (self.BHS, \
                              Layer_Frame.VACUUM, \
                              None, True) ]
        self.__rebuildLayers(layers_to_append)
        # Initialisation of self.__modif
        self.set_modif(False)
        # User Interface Initialization 
        self.__initUI()       
    #----------------------------------------------------------------------
    def __initUI(self):
        # Size
        w = round(0.97*self.mw.WIDTH)
        h = round(0.80*self.mw.HEIGHT)
        h = max(h, self.MAX_LAYER*2*Layer_Frame.HEIGHT)
        self.resize(w, h)
        # Check button
        self.chk_but.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        self.chk_but.setFixedSize(self.BUT_W, self.BUT_H)
        # Layout
        right = QHBoxLayout()
        right.addStretch()
        self.grid.addLayout(right,0,2)
        self.setLayout(self.grid)
    #----------------------------------------------------------------------
    @property
    def structure_file_path(self) :
        rfp = os.path.relpath(self.__fileLoc+"/"+self.__fileName)
        return rfp.replace("\\", "/")
    #----------------------------------------------------------------------
    def set_new_file(self) :
        self.__fileName = self.mw.UNTITLED
        self.__deleteLayers()
        # Top Half-Space: Vacuum
        self.__layers[0].set_material(Layer_Frame.VACUUM) 
        layers_to_append = [ (None,None,None,False), \
                             (self.BHS, \
                              Layer_Frame.VACUUM, \
                              None, True) ]
        self.__rebuildLayers(layers_to_append)
        self.set_modif(False)
    #----------------------------------------------------------------------
    @property
    def modif(self) :
        return self.__modif
    #----------------------------------------------------------------------
    def set_modif(self, true_false) :
        self.__modif = true_false
        self.mw.update_statusBar()
    #----------------------------------------------------------------------    
    @property
    def shortpath(self) :
        file_path = self.structure_file_path  
        if len(file_path)<99 :
            path = file_path
        else :
            dm = max( 47 - lennam//2 , 5 )
            path = self.__fileLoc[:dm]+"[...]"+self.__fileLoc[-dm:]+\
                   "/"+self.__fileName
        if self.__modif == True : path = "***"+path+"***"
        return path
    #----------------------------------------------------------------------
    # Called in the constructor of Layer_Frame
    def append_layer(self, new_layer) :
        self.__layers.append(new_layer)
        self.set_modif(True)
    #----------------------------------------------------------------------
    @property
    def number_of_layers(self) :
        return len(self.__layers)-2
    #----------------------------------------------------------------------
    @property
    def top_half_space_material(self) :
        return self.mat_of_name(self.__layers[0].mat_name)
    #----------------------------------------------------------------------
    @property
    def bottom_half_space_material(self) :
        return self.mat_of_name(self.__layers[-1].mat_name)
    #----------------------------------------------------------------------
    def material_of_layer(self, number) :
        """Numbering of layers begins at 1."""
        return self.mat_of_name(self.__layers[number].mat_name)
    #----------------------------------------------------------------------
    def width_of_layer(self, number) :
        """Numbering of layers begins at 1."""
        return self.__layers[number].width_in_mm   
    #----------------------------------------------------------------------
    def __update_pos(self):
        self.__pos[0].value_in_mm = 0.0
        cur_val = self.__pos[0].value_in_mm
        ok = True
        for lay,pos in zip(self.__layers[1:-1],self.__pos[1:]) :
            if ok :
                w = lay.width_in_mm
                ok = w is not None
            if ok :
                cur_val += w
                pos.value_in_mm = cur_val
            else :
                pos.value_in_mm = None 
        # Global checking of the structure definition
        self.isCompletelyDefinedStructure(False)
    #----------------------------------------------------------------------
    def update_list_of_materials(self):
        undef = Layer_Frame.UNDEFINED
        not_cons = Layer_Frame.HS_CONDITIONS + [undef]
        list_of_materials = []
        for lay in self.__layers :
            mn = lay.mat_name
            if mn not in not_cons and mn not in list_of_materials :
                list_of_materials.append(mn)
        list_of_materials.sort()
        list_of_materials.append(undef)
        for lay in self.__layers :
            lay.update_combo(list_of_materials)
    #----------------------------------------------------------------------
    def mat_of_name(self, mat_name) :
        if mat_name == Layer_Frame.UNDEFINED : return None
        if mat_name in Layer_Frame.HS_CONDITIONS : return mat_name
        for lay in self.__layers :
            if lay.mat_name == mat_name :
                return lay.material
        print("Geometry_Frame.mat_of_name - Warning:\n\t" +
              f"Material name '{mat_name}' not found")
        return None
    #----------------------------------------------------------------------
    def __deleteLayers(self, line=1):
        #Grid dimensions
        rows = len(self.__layers)*2
        cols = self.grid.columnCount()
        lay = self.__layers.pop() # Lower Half-Space
        delete_layers = [lay.args()]
        lay.deleteWidget()        
        for nr in (self.RB+rows,self.RB+rows-1) :
            for nc in range(cols-1,-1,-1) :
                item = self.grid.itemAtPosition(nr,nc)
                if item is None :
                    pass # No object at this position
                elif item.widget() is not None: # widget to remove
                    widgetToRemove = item.widget()
                    widgetToRemove.setParent(None)
                    self.grid.removeWidget(widgetToRemove)
                else : # layout to remove
                    item.setParent(None)  
        for lay,pos,nrx in zip(self.__layers[:line-1:-1], \
                               self.__pos[:line-1:-1], \
                               range(self.RB+rows-2,self.RB+1,-2) ) :
            delete_layers.append(lay.args())
            lay.deleteWidget()
            pos.deleteWidget()
            self.__layers.pop()
            self.__pos.pop()
            self.__ins.pop()        
            for nr in (nrx,nrx-1) :
                for nc in range(cols-1,-1,-1) :
                    item = self.grid.itemAtPosition(nr,nc)
                    if item is None :
                        pass # No object at this position
                    elif item.widget() is not None: # widget to remove
                        widgetToRemove = item.widget()
                        widgetToRemove.setParent(None)
                        self.grid.removeWidget(widgetToRemove)
                    else : # layout to remove
                        item.setParent(None)
        delete_layers.reverse()     
        return delete_layers
    #----------------------------------------------------------------------
    def __rebuildLayers(self, layers_to_append):
        nb = len(self.__layers)     
        align = Qt.AlignLeft|Qt.AlignVCenter
        row = self.RB+2*nb
        for i,(n,mat,w,hs) in enumerate(layers_to_append[:-1], nb) :
            if n is not None or hs:
                print("Geometry_Frame.__rebuildLayers error:\n\t" + \
                      f"'{n}' should be None and {hs} should be False")
            # Layer frame
            lay = Layer_Frame(self, i, mat, w, hs)
            # (Automatically append to self.__layers)
            lay.changedWidth.connect(self.__update_pos)
            # Position in thickness
            pos = pos_label()
            self.__pos.append(pos)
            pos.setEnabled(False)
            # Insert layer
            ins = QPushButton("Insert Layer")
            self.__ins.append(ins)
            ins.released.connect(lambda no=i+1: self.insertLayer(no))
            # Update grid
            row += 1
            self.grid.addWidget( lay, row, 1, 1, 1, align)
            row += 1
            self.grid.addWidget( pos, row, 0, 1, 1, align)
            self.grid.addWidget( QHLine(), row, 1, 1, 2)
            self.grid.addWidget( ins, row, 3, 1, 1, align)            
        # Lower half space
        n,mat,w,hs = layers_to_append[-1]
        lay = Layer_Frame(self, n, mat, w, hs)
        # (Automatically append to self.__layers)
        row += 1
        self.grid.addWidget( lay, row, 1, 1, 1, align)
        row += 1
        bot = QVBoxLayout() ; 
        bot.addStretch() ; 
        self.grid.addLayout(bot,row,0)
        if len(self.__layers) == 3 : # A monolayer
            self.__layers[1].del_but.setEnabled(False)
            self.__layers[1].del_but.setStyleSheet(Layer_Frame.GRAY)
        self.__update_pos()
        # Global checking of the structure definition included in
        # self.__update_pos()
    #----------------------------------------------------------------------
    def deleteLayer(self, lay_frm):
        lay_idx = self.__layers.index(lay_frm)
        delete_layers = self.__deleteLayers(lay_idx)
        self.__rebuildLayers(delete_layers[1:])
        self.set_modif(True)  
    #----------------------------------------------------------------------
    def insertLayer(self, lay_idx): 
        self.__layers[1].del_but.setEnabled(True)
        self.__layers[1].del_but.setStyleSheet(Layer_Frame.TEXTCOLOR)               
        delete_layers = self.__deleteLayers(lay_idx)
        self.__rebuildLayers([(None,None,None,False)]+delete_layers)
        self.set_modif(True)  
    #----------------------------------------------------------------------
    def setCheckButton(self, ok) :
        if ok :
            self.chk_but.setText("Plate completely defined")
            self.chk_but.setStyleSheet(self.mw.OK_COLOR)
        else :
            self.chk_but.setText("Definition in progress...")
            self.chk_but.setStyleSheet(self.mw.KO_COLOR)
            self.mw.folderMenu.setEnabled(False)
    #----------------------------------------------------------------------
    def isCompletelyDefinedStructure(self, message_box=True) :
        if len(self.__layers) < 3 or not self.__layers[-1].hs :
            return # Rebuilding in progress
        ok,msg,_,_,_,_ = self.is_completely_defined()
        if ok :
            pass
        else :
            if message_box :
                qmb = QMessageBox(self)
                qmb.setWindowTitle("Warning!")
                qmb.setIcon(QMessageBox.Warning)
                qmb.setText(msg.replace("\n","<br>"))
                qmb.setStandardButtons(QMessageBox.Ok)
                qmb.setDefaultButton(QMessageBox.Ok)
                rep = qmb.exec_()                
        self.mw.setCompletelyDefinedStructure(ok)
    #----------------------------------------------------------------------
    def is_completely_defined(self) :
        msg = "Multilayer Plate Checking:"
        layer_data, material_data = [], []
        OK = True
        top_hs_mat = self.__layers[0].material
        if top_hs_mat in Layer_Frame.HS_CONDITIONS :
            top_hs_mat_name = top_hs_mat
            if top_hs_mat_name == Layer_Frame.WALL :
                # Rigid Wall
                if not isinstance(self.__layers[1].material, Fluid) :
                    OK = False
                    msg += "\n+ First solid layer in contact with " + \
                           "rigid wall."                    
        elif top_hs_mat == None :
            OK = False
            top_hs_mat_name = Layer_Frame.UNDEFINED
            msg += "\n+ Material of top half-space undefined."
        else :
            material_data.append(top_hs_mat)
            top_hs_mat_name = self.__layers[0].mat_name
        for lay in self.__layers[1:-1] :
            nm,w,mat = lay.name, lay.width_in_mm, lay.material
            if w is None :
                OK = False
                msg += f"\n+ Width of {nm} undefined."
                w = "(undefined width)"
            else :
                w = f"Width: {w:.3f} mm"
            if mat is None :
                OK = False
                msg += f"\n+ Material of {nm} undefined."
                mat = '(undefined material)'
            else :
                if mat not in material_data :
                    material_data.append(mat)
                mat = f"Material: {lay.mat_name}"
            layer_data.append([w,mat])
        bottom_hs_mat = self.__layers[-1].material
        if bottom_hs_mat in Layer_Frame.HS_CONDITIONS :
            bottom_hs_mat_name = bottom_hs_mat
            if bottom_hs_mat_name == Layer_Frame.WALL :
                # Rigid Wall
                if not isinstance(self.__layers[-2].material, Fluid) :
                    OK = False
                    msg += "\n+ Last solid layer in contact with " + \
                           "rigid wall." 
        elif bottom_hs_mat == None :
            OK = False
            bottom_hs_mat_name = Layer_Frame.UNDEFINED
            msg += "\n+ Material of bottom half-space undefined."
        else :
            if bottom_hs_mat not in material_data :
                material_data.append(bottom_hs_mat)
            bottom_hs_mat_name = self.__layers[-1].mat_name
        print(f"[{OK}] "+msg)        
        self.setCheckButton(OK)
        return OK, msg, top_hs_mat_name, layer_data, \
               bottom_hs_mat_name, material_data
    #----------------------------------------------------------------------
    def write_in_file(self, file_path=None, ask_file_path=False, \
                      ask_results_folder=False) :
        OK, msg, ths, layer_data, bhs, materials = \
            self.is_completely_defined()
        if not OK :
            qmb = QMessageBox(self)
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText( msg.replace("\n","<br>") + \
                         "<br><br><b>Continue?</b>" )
            qmb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
            qmb.setDefaultButton(QMessageBox.Ok)
            rep = qmb.exec_()
            if rep != QMessageBox.Ok : return False # Abort saving
        text = USMP.export_to_text(layer_data, materials, ths, bhs)
        new_file_path = True
        if file_path is None :
            if self.__fileName == self.mw.UNTITLED or ask_file_path :
                file_path = QFileDialog.getSaveFileName( \
                            self,\
                            "Save as:", \
                            self.mw.PLATE_DIR, \
                            "Text files (*.txt);;All files (*)")[0]
                if file_path == "" : return False # Abort saving
            else :
                file_path = self.structure_file_path
                new_file_path = False        
        if new_file_path : # Update file path
            file_path = file_path.replace("\\","/")
            self.__fileLoc = os.path.dirname(file_path)
            self.__fileName = os.path.basename(file_path)
            self.mw.update_recent_files()
        # Select Results Folder
        if new_file_path or ask_results_folder :
            default_folder_name = self.__fileName.replace(".txt","")
            ok = self.mw.select_folder(suggested_name=\
                                       default_folder_name, \
                                       save_plate_file=False)
            if not ok : return False # Abort saving 
        rf_relative_path = os.path.relpath(self.mw.results_folder)
        rf_relative_path = rf_relative_path.replace("\\","/")
        text += f"Last Results Folder: {rf_relative_path}\n"
        with open(file_path, "w", encoding="utf8") as f :
            f.write(text)
        self.set_modif(False)
        return True
    #----------------------------------------------------------------------
    def read_from_file(self, file_path) :
        error_msg = f"File '{file_path}'\ndoes not seem to " + \
                     "correspond to a multilayer plate:"
        pos_enc = ("utf8","cp1252")
        results_folder = None
        for enc in pos_enc :
            try :
                with open(file_path, "r", encoding=enc) as strm :
                    rows = []
                    for r in strm :
                        r = r.strip()
                        if "results folder" in r.lower() :
                           results_folder = r.split(":")[1].strip()
                           print("Results folder:", results_folder)
                        else :
                            rows.append(r.lower())
                ok = True
                break
            except :
                ok = False
        if not ok :
            msg = error_msg + f"\n\tencoding error: not in {pos_enc}."
            return False, msg
        txt = "\n".join(rows)
        #------------------------------------------------------------------
        layer_data, materials, mat_ths, mat_bhs = \
                    USMP.import_from_text(txt, raised_errors=False)
        if isinstance(layer_data, bool) : # necessary False
            return False, materials
        #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.__layers[0].set_material(mat_ths)
        args_lay = []
        for w_m, mat in layer_data :
            args_lay.append( [None, mat, w_m, False] )
        args_lay.append( [self.BHS, mat_bhs, None, True] ) 
        #------------------------------------------------------------------
        # Rebuilding the layers
        self.__deleteLayers()
        self.__rebuildLayers(args_lay)
        # Updating file path
        self.__fileLoc = os.path.dirname(file_path)
        self.__fileName = os.path.basename(file_path)
        self.mw.update_recent_files()
        self.set_modif(False)
        if results_folder is None : # Select a results folder and save
            self.write_in_file(file_path, ask_results_folder=True)
        else :
            self.mw.select_folder(folder_path=results_folder, \
                                  save_plate_file=False)
        return True,[]
#========================================================================
class Layer_Frame(QFrame) :
    """To define and visualize the geometry and the material of a layer.
    """
    changedWidth = pyqtSignal(int)
    VACUUM = "[ Vacuum ]"
    WALL = "[ Rigid Wall ]"
    HS_CONDITIONS = [VACUUM, WALL]
    UNDEFINED = "(Undefined)"
    FONTSIZE = 11
    FONTNAME = "Arial"
    TEXTCOLOR = "color: rgb(160,0,160)"
    GRAY = "color: rgb(160,160,160)"
    WIDTH = 900
    LABEL_W = 200
    HEIGHT = 40
    LABEL_H = 30
    #----------------------------------------------------------------------
    def __init__(self, parent, name_or_number, material=None, \
                 thickness=None, half_space=False) :        
        self.__mat = None        # Material of the layer, modified below
        self.__thck = thickness  # Thickness of the layer [m]
        self.__hs = half_space   # Half-space or not ?
        self.name = name_or_number
        QFrame.__init__(self, parent)
        self.geom_frm = parent
        self.geom_frm.append_layer(self)
        # Layout                 
        self.lay = QGridLayout()
        self.lay.setAlignment(Qt.AlignHCenter|Qt.AlignVCenter)
        # Layer name
        self.label = QLabel(self)
        self.label.setText(self.name)
        self.label.setAlignment(Qt.AlignVCenter)
        # Layer width
        if half_space :
            if material is None :
                material = self.VACUUM # Vacuum by default
            self.width_edit = QFrame(self)
            self.width_edit.setFixedSize(pos_label.WTOT,pos_label.HTOT)
            self.del_but = QFrame(self)
        else :
            self.width_edit = pos_label("w")
            if thickness is None :
                self.width_edit.value_in_mm = None
            else :
                self.width_edit.value_in_mm = 1e3*thickness
            self.width_edit.changedValue.connect(self.update_width)
            self.del_but = QPushButton("Delete Layer",self)
            self.del_but.released.connect( \
                lambda : self.geom_frm.deleteLayer(self) )
        # Material Combo Box
        self.mat_cmb = QComboBox(self)
        self.__prev_choice = None
        self.mat_cmb.activated.connect(self.__change_mat)
        self.set_material(material)
        # Material Button
        self.mat_but = QPushButton("Edit", self)
        self.mat_but.released.connect(self.begin_edit_material)
        self.edit_material_window = None
        # Initialization of the user interface
        self.__initUI()
    #----------------------------------------------------------------------
    def __initUI(self) :
        align = Qt.AlignLeft|Qt.AlignVCenter
        # Size
        self.setMinimumHeight(self.HEIGHT)
        self.setMaximumHeight(self.HEIGHT)
        self.setMinimumWidth(self.WIDTH)
        self.setMaximumWidth(self.WIDTH)
        # Label of the layer
        self.label.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        self.label.setStyleSheet(self.TEXTCOLOR)
        self.label.setFixedSize(self.LABEL_W, self.LABEL_H)
        self.label.setAlignment(align)
        # Material Selection
        mat_lbl = QLabel(self)
        mat_lbl.setText("Material: ")
        mat_lbl.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        mat_lbl.setStyleSheet(self.TEXTCOLOR)
        mat_lbl.setFixedSize(round(0.35*self.LABEL_W), self.LABEL_H)
        self.mat_cmb.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        self.mat_cmb.setStyleSheet(self.TEXTCOLOR)
        self.mat_cmb.setFixedSize(self.LABEL_W, self.LABEL_H)
        self.mat_but.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        self.mat_but.setStyleSheet(self.TEXTCOLOR)
        self.mat_but.setFixedSize(round(0.3*self.LABEL_W), self.LABEL_H)
        # Delete Button
        self.del_but.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
        self.del_but.setStyleSheet(self.TEXTCOLOR)
        self.del_but.setFixedSize(round(0.6*self.LABEL_W), self.LABEL_H)
        # Layout
        top,bot,right = QVBoxLayout(),QVBoxLayout(),QHBoxLayout()
        top.addStretch() ; bot.addStretch() ; right.addStretch()
        self.lay.addLayout(top, 0, 0)
        self.lay.addWidget(self.label, 1, 0, 1, 2, align)
        self.lay.addWidget(self.width_edit, 1, 2, 1, 1, align)
        self.lay.addWidget(mat_lbl, 1, 3, 1, 1, align)
        self.lay.addWidget(self.mat_cmb, 1, 4, 1, 1, align)
        self.lay.addWidget(self.mat_but, 1, 5, 1, 1, align)
        self.lay.addWidget(self.del_but, 1, 6, 1, 1, align)
        self.lay.addLayout(right, 1, 7)
        self.lay.addLayout(bot, 2, 0)
        self.setLayout(self.lay)
    #----------------------------------------------------------------------
    def deleteWidget(self) :
        """Delete the object when the layer is removed."""
        for i in range(self.lay.count()-1,-1,-1) :
            layoutItem = self.lay.itemAt(i)
            if layoutItem.widget() is not None: # widget to remove
                widgetToRemove = layoutItem.widget()
                widgetToRemove.setParent(None)
                self.lay.removeWidget(widgetToRemove)
            else : # layout to remove
                layoutItem.setParent(None)  
        # Attributes
        self.label = None
        if isinstance(self.width_edit, pos_label) :
            self.width_edit.deleteWidget()
        self.width_edit = None
        self.mat_but = None
        self.del_but = None
        self.lay = None
    #----------------------------------------------------------------------
    @property
    def mat_name(self) :
        if self.__mat is None : return self.UNDEFINED
        elif isinstance(self.__mat, str) :
            return self.__mat
        else : return self.__mat.name
    @property
    def material(self) :
        return self.__mat
    @property
    def width_in_mm(self) :
        if self.__thck is None : return None
        return self.__thck * 1e3 # m -> mm
    @property
    def hs(self) : return self.__hs
    #----------------------------------------------------------------------
    @property
    def name(self) : return self.__name
    @name.setter
    def name(self, new_non) :
        if isinstance(new_non, str) :
            self.__name = new_non
        elif isinstance(new_non, int) and new_non > 0 :
            self.__name = f"Layer {new_non}"
        else :
            print(f"Error : name '{new_non}' not understood")
    #----------------------------------------------------------------------
    def update_width(self) :
        thck_mm = self.width_edit.value_in_mm
        if thck_mm is None :
            self.__thck = None
        else :
            self.__thck = 1e-3*thck_mm
        self.geom_frm.set_modif(True)
        self.changedWidth.emit(1)
    #----------------------------------------------------------------------
    def begin_edit_material(self) :
        self.geom_frm.mw.setEnabled(False)
        self.edit_material_window = Material_App(self)
        self.edit_material_window.show()
    #----------------------------------------------------------------------
    def end_edit_material(self, material) :
        self.geom_frm.mw.setEnabled(True)
        if material is None : return # Cancel
        self.set_material(material)
    #----------------------------------------------------------------------
    def set_material(self, material) :
        self.__mat = material
        self.geom_frm.update_list_of_materials()
        if material is None :
            self.mat_cmb.setCurrentText(self.UNDEFINED)
        elif material in self.HS_CONDITIONS :
            self.mat_cmb.setCurrentText(material)
        else :
            self.mat_cmb.setCurrentText(material.name)
        self.geom_frm.set_modif(True)
        # Global checking of the structure definition
        self.geom_frm.isCompletelyDefinedStructure(False)
    #----------------------------------------------------------------------
    def __change_mat(self) :
        new_mat_name = self.mat_cmb.currentText()
        if new_mat_name == '' : return # Empty combo
        if new_mat_name == self.__prev_choice : return # No change
        self.__prev_choice = new_mat_name
        self.set_material( self.geom_frm.mat_of_name(new_mat_name) )
    #----------------------------------------------------------------------
    def update_combo(self, list_of_mat) :
        previous_choice = self.mat_cmb.currentText()
        if self.__hs :
            list_of_mat = self.HS_CONDITIONS + list_of_mat
        self.mat_cmb.clear()
        self.mat_cmb.addItems(list_of_mat)
        self.mat_cmb.setCurrentText(previous_choice)
    #----------------------------------------------------------------------
    def args(self) :
        """Parameters to rebuild a new instance."""
        if "Layer " in self.__name :
            try :
                nb = int(self.__name[6:])
                half_space = False
                name_or_number = None
            except :
                half_space = True
                name_or_number = self.name
        else :
            half_space = True
            name_or_number = self.name
        return (name_or_number, self.__mat, self.__thck, half_space)            
#========================================================================
class pos_label(QFrame):
    """For thickness or position in mm."""
    changedValue = pyqtSignal(int)
    H = 24
    NB_CHAR = (2,5,3)
    W_by_CHAR = 13
    WTOT = (sum(NB_CHAR)+2)*W_by_CHAR
    HTOT = 30        
    FONTSIZE = 9
    FONTNAME = "Arial"
    TEXTCOLOR = "color: rgb(0,0,160)"
    #----------------------------------------------------------------------
    def __init__(self, name="z"):
        """name must be a single letter."""
        QFrame.__init__(self)
        self.__value_mm = None
        self.lay = QHBoxLayout()
        self.name = QLabel(self)
        self.setName(name)
        self.value_edit = QLineEdit(self)
        self.value_edit.editingFinished.connect(self.__update_value)
        self.unit = QLabel(self)
        self.unit.setText(" mm")
        self.__initUI()
    #----------------------------------------------------------------------
    def __initUI(self) :
        self.setContentsMargins(0,0,0,0)
        self.lay.setContentsMargins(0,0,0,0)
        self.lay.setSpacing(0)
        self.lay.setAlignment(Qt.AlignVCenter)
        align = Qt.AlignHCenter|Qt.AlignVCenter
        # Layout
        self.lay.addWidget(self.name, align)
        self.lay.addWidget(self.value_edit, align)
        self.lay.addWidget(self.unit, align)
        self.setLayout(self.lay)
        # sizes and fonts
        self.setFixedSize(self.WTOT, self.HTOT)
        for lbl,n in zip( (self.name, self.value_edit, self.unit),\
                          self.NB_CHAR ) :
            lbl.setFont( QFont(self.FONTNAME, self.FONTSIZE, \
                                  QFont.Bold))
            lbl.setStyleSheet(self.TEXTCOLOR)
            lbl.setFixedSize(n*self.W_by_CHAR, self.H)
            lbl.setAlignment(align)
    #----------------------------------------------------------------------
    def deleteWidget(self) :
        """Delete the object when the layer is removed."""
        for i in range(self.lay.count()-1,-1,-1) :
            layoutItem = self.lay.itemAt(i)
            if layoutItem.widget() is not None: # widget to remove
                widgetToRemove = layoutItem.widget()
                widgetToRemove.setParent(None)
                self.lay.removeWidget(widgetToRemove)
            else : # layout to remove
                layoutItem.setParent(None)  
        # Attributes
        self.name = None
        self.value_edit = None
        self.unit = None
        self.lay = None        
    #----------------------------------------------------------------------
    @property
    def value_in_mm(self) :
        return self.__value_mm
    @value_in_mm.setter
    def value_in_mm(self, val) :
        try : self.__value_mm = float(val)
        except : self.__value_mm = None
        self.__update_value(read=False)
    #----------------------------------------------------------------------
    def setEnabled(self, ok=True) :
        self.value_edit.setEnabled(ok)
    #----------------------------------------------------------------------
    def setName(self, new_name) :
        """new_name must be a single letter."""
        if isinstance(new_name, str) and len(new_name) == 1 :
            self.name.setText(new_name+" ")
    #----------------------------------------------------------------------
    def __update_value(self, read=True) :
        emit = False
        if read and self.value_edit.isModified() :
            txt = self.value_edit.text()
            try : val = float(txt.replace(",","."))
            except : val = None
            if val == self.__value_mm : return
            self.__value_mm = val
            emit = True
        if self.__value_mm is None : self.value_edit.setText("")
        else : self.value_edit.setText("{:.2f}".format(self.__value_mm))
        if emit : self.changedValue.emit(1)
