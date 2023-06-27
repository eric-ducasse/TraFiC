# Version 0.84 - 2022, September 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import os, platform
from PyQt5.QtWidgets import (QApplication, QMainWindow, \
                             QWidget, QScrollArea, QAction, \
                             QPushButton, QFrame, QLabel, \
                             QGridLayout, QHBoxLayout, QVBoxLayout, \
                             QFileDialog, QMessageBox, QInputDialog, \
                             QTabWidget, QLineEdit)
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QEvent, QRect, QSize
#---------- TraFiC classes ----------------------------------------------
if __name__ == "__main__" :
    import sys
    root = os.path.abspath("..")
    sys.path.append(root)
    import TraFiC_init
else :
    from TraFiC_init import root
#---------- TraFiC GUI classes ------------------------------------------
from Geometry_Frame import Geometry_Frame, Layer_Frame
from Mode_Frames import Mode_Shape_Frame, Dispersion_Curve_Frame
from Field_Frames import Source_Frame, Field_Computation_Frame
#---------- TraFiC computation classes ----------------------------------
from MaterialClasses import *
from Modes_Immersed_Multilayer_Plate import DiscretizedMultilayerPlate
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TraFiC_Application(QMainWindow):
    WIDTH = 1100
    HEIGHT = 800
    MAT_DIR = root + "/Data/Materials"
    PLATE_DIR = root + "/Data/Plates"
    PIPE_DIR = root + "/Data/Pipes"
    RECENT_FILES = root + "/Data/TraFiC_files/recent_files.txt"
    RESULT_DIR = root + "/Data/Results"
    RECENT_FOLDERS = root + "/Data/TraFiC_files/recent_folders.txt"
    MAX_FILES = 50  # Max number of recent files in the file
    NB_RF = 10      # Max number of recent files in the menu
    UNTITLED = "<untitled>.txt"
    DEF_DZ = 0.05 # Default discretization step in mm for mode computation
    ## DEF_DZ is very important (cost/accuracy)
    FDS_ORDER = 8 # Defaut order of the finite differences scheme
    OK_COLOR = \
            "color: rgb(0,100,0); background-color: rgb(200,255,200);"
    KO_COLOR = \
            "color: rgb(200,0,0); background-color: rgb(255,200,200);"
    PR_COLOR = \
            "color: rgb(0,0,200); background-color: rgb(200,230,255);"
    #--------------------------------------------------------------------
    def __init__(self):
        QMainWindow.__init__(self)

        # Menu bar
        self.menubar = self.menuBar() # Necessary for Mac OSX
        self.fileMenu = self.menubar.addMenu('File')
        self.folderMenu = self.menubar.addMenu('Results Folder')
        self.__crf = QAction("Current Results Folder:", parent=self)
        self.results_folder = None
        # Geometry and materials
        self.geom_frm = Geometry_Frame(self)
        geom_tab = QScrollArea(self)
        geom_tab.setWidget(self.geom_frm)
        # Mode Shape
        self.shp_frm = Mode_Shape_Frame(self)
        # Dispersion curve
        self.disp_frm = Dispersion_Curve_Frame(self)
        # Source definition
        self.src_frm = Source_Frame(self)
        # Field computation
        self.fc_frm = Field_Computation_Frame(self)

        # Tabs
        self.tabs = QTabWidget()        
        self.tabs.addTab(geom_tab,"Layers")      
        self.tabs.addTab(self.shp_frm,"Mode Shape")    
        self.tabs.addTab(self.disp_frm,"Dispersion Curves")    
        self.tabs.addTab(self.src_frm,"Source Definition")    
        self.tabs.addTab(self.fc_frm,"Field Computation")
        self.tabs.tabBarClicked.connect(self.__changeTabByClick)
        self.tabs.currentChanged.connect(self.__changeTab)
        self.__wanted_tab = 0
        self.__changeTab()
        self.setCentralWidget(self.tabs)

        # Parameters of the last mode computation
        self.__last_ths_mat = None # Material of the top half-space
        self.__last_bhs_mat = None # Material of the bottom half-space
        self.__last_layers  = None # (Material,width) of the layers
        self.__discretized_plate = None
        self.__previous_path = None
        
        self.__initUI()   # User Interface Initialization
        self.show()       # Shows the main window
    #--------------------------------------------------------------------
    def __initUI(self):
        self.resize(self.WIDTH, self.HEIGHT)
        self.center()
        self.setWindowTitle("Transient Field Computation (TraFiC)")
        self.statusBar()

        # Menu(s)        
        if platform.uname().system.startswith('Darw') : #Mac OSX
            self.menubar.setNativeMenuBar(False)

        ###### 'File' Menu
        # Open
        qo = QAction("Open Multilayer Plate",parent=self)
        qo.setShortcut('Ctrl+O')
        qo.triggered.connect(lambda _ : self.open_file(None))
        self.fileMenu.addAction(qo)
        # New File
        qn = QAction("New Multilayer Plate",parent=self)
        qn.setShortcut('Ctrl+N')
        qn.triggered.connect(self.new_file)
        self.fileMenu.addAction(qn)
        # Save
        qs = QAction("Save Multilayer Plate", parent=self)        
        qs.setShortcut('Ctrl+S') 
        qs.triggered.connect(self.save)
        self.fileMenu.addAction(qs)
        # Save As
        qsa = QAction("Save Multilayer Plate As...", parent=self)        
        qsa.setShortcut('Ctrl+Shift+S') 
        qsa.triggered.connect(self.save_as)
        self.fileMenu.addAction(qsa)
        # Quit
        qa = QAction("Exit", parent=self)        
        qa.setShortcut('Ctrl+Q') 
        qa.triggered.connect(self.close)
        self.fileMenu.addAction(qa)
        # Recent files
        if not os.path.isfile(self.RECENT_FILES) :
            with open(self.RECENT_FILES,"w", encoding="utf8") as f :
                f.write("")            
        self.fileMenu.addSeparator()
        tt = QAction("Recent Files:", parent=self)
        tt.setEnabled(False)
        self.fileMenu.addAction(tt)
        self.__recent_files = self.recent_files
        self.__menu_recent_files = []
        self.__update_menu_recent_files()
        ###### 'Results Folder' Menu
        # Current Results Folder
        self.__crf.setEnabled(False)
        self.folderMenu.addAction(self.__crf)
        # Select
        qf = QAction("Change Folder", parent=self)        
        qf.setShortcut('Ctrl+F') 
        qf.triggered.connect(self.select_folder)
        self.folderMenu.addAction(qf)
        # Recent folders
        if not os.path.isfile(self.RECENT_FOLDERS) :
            with open(self.RECENT_FOLDERS,"w", encoding="utf8") as f :
                f.write("")            
        self.folderMenu.addSeparator()
        tt = QAction("Recent Folders:", parent=self)
        tt.setEnabled(False)
        self.folderMenu.addAction(tt)
        self.__recent_folders = self.recent_folders
        self.__menu_recent_folders = []
        self.__update_menu_recent_folders()
        self.folderMenu.setEnabled(False)
    #--------------------------------------------------------------------
    def __update_menu_recent_files(self) :
        # Update the existing items
        n = 0 # if empty list
        for n,(item,rf) in enumerate(zip(self.__menu_recent_files, \
                                         self.__recent_files), 1) :
            lab = ".../" + os.path.basename(os.path.dirname(rf)) + \
                  "/" + os.path.basename(rf)
            item.setText(lab) # Change text
        # Create new items
        for i,rf in enumerate(self.__recent_files[n:],n) :
            lab = ".../" + os.path.basename(os.path.dirname(rf)) + \
                  "/" + os.path.basename(rf)
            item = QAction(lab, parent=self)
            item.triggered.connect( \
                            lambda _,no=i: self.open_recent_file(no))
            self.fileMenu.addAction(item)
            self.__menu_recent_files.append(item)
    #--------------------------------------------------------------------
    def __update_menu_recent_folders(self) :
        # Update the existing items
        n = 0 # if empty list
        for n,(item,rf) in enumerate(zip(self.__menu_recent_folders, \
                                         self.__recent_folders), 1) :
            lab = ".../" + os.path.basename(os.path.dirname(rf)) + \
                  "/" + os.path.basename(rf)
            item.setText(lab) # Change text
        # Create new items
        for i,rf in enumerate(self.__recent_folders[n:],n) :
            lab = ".../" + os.path.basename(os.path.dirname(rf)) + \
                  "/" + os.path.basename(rf)
            item = QAction(lab, parent=self)
            item.triggered.connect( \
                        lambda _,no=i: self.select_folder(number=no))
            self.folderMenu.addAction(item)
            self.__menu_recent_folders.append(item)
    #--------------------------------------------------------------------
    def update_recent_files(self) :
        new_file_path = self.geom_frm.structure_file_path
        if os.path.basename(new_file_path) == self.UNTITLED : return
        # Update the file
        last_recent_files = self.__all_last_files()
        with open(self.RECENT_FILES, "w", encoding="utf8") as f :
            L_rf = [new_file_path]
            f.write(new_file_path+"\n")
            n = 1
            for fp in last_recent_files :
                if fp != new_file_path :
                    L_rf.append(fp)
                    f.write(fp+"\n")
                    n += 1
                if n == self.MAX_FILES :
                    break
        # Update the recent files
        self.__recent_files = L_rf[:self.NB_RF]
        # Update the menu
        self.__update_menu_recent_files()
    #--------------------------------------------------------------------
    def update_recent_folders(self) :
        new_folder_path = self.results_folder
        self.__crf.setText(f"Current Results Folder: {new_folder_path}")
        # Update the file
        last_recent_folders = self.__all_last_folders()
        with open(self.RECENT_FOLDERS, "w", encoding="utf8") as f :
            L_rf = [new_folder_path]
            f.write(new_folder_path+"\n")
            n = 1
            for fp in last_recent_folders :
                if fp != new_folder_path :
                    L_rf.append(fp)
                    f.write(fp+"\n")
                    n += 1
                if n == self.MAX_FILES :
                    break
        # Update the recent files
        self.__recent_folders = L_rf[:self.NB_RF]
        # Update the menu
        self.__update_menu_recent_folders()
    #--------------------------------------------------------------------
    def update_statusBar(self) :
        try :
            self.statusBar().showMessage(self.geom_frm.shortpath)
        except :
            pass
    #--------------------------------------------------------------------
    @property
    def recent_files(self) :
        return self.__all_last_files(self.NB_RF)         
    #--------------------------------------------------------------------
    def __all_last_files(self, number=None) :
        if number is None: number = self.MAX_FILES
        with open(self.RECENT_FILES, "r", encoding="utf8") as f :
            file_pathes = [ row.strip() for row in f.readlines() ]
        return [ p for p in file_pathes if len(p) > 0 ][:number]
    #--------------------------------------------------------------------
    @property
    def recent_folders(self) :
        return self.__all_last_folders(self.NB_RF)         
    #--------------------------------------------------------------------
    def __all_last_folders(self, number=None) :
        if number is None: number = self.MAX_FILES
        with open(self.RECENT_FOLDERS, "r", encoding="utf8") as f :
            folder_pathes = [ row.strip() for row in f.readlines() ]
        return [ p for p in folder_pathes if len(p) > 0 ][:number]        
    #--------------------------------------------------------------------
    def __changeTabByClick(self, index) :
        """index : 0 -> Layers, 1 -> Mode Shape, 2 -> Dispersion Curves,
                   3 -> Source Definition, 4 -> Field Computation."""
        if index == -1 : return # No effect
        last_index = self.tabs.currentIndex()
        if index == last_index : return # No change            
        if index == 0 :
            self.__wanted_tab = index # Confirmed change
        elif index in (1,2) :
            if last_index in (1,2) : # No necessary checkN
                self.__wanted_tab = index # Confirmed change
            elif self.__well_defined_plate : # OK
                if last_index == 0 and self.geom_frm.modif :
                    msg = "<b><i>Unsaved Changes<b></i>:\n" + \
                          "Structure data must be saved before mode" + \
                          " computation.\nSave and continue?"
                    rep = self.approve(msg, no_button=False)
                    if rep == 1 : # Yes
                        self.save()
                    else : # rep==0 [Cancel]
                        return
                if self.geom_frm.structure_file_path == \
                   self.__previous_path : # Same plate as previously
                    self.__wanted_tab = index # Confirmed tab change
                    return                    
                # Check if the half-spaces are not solid.
                ths_mat = self.geom_frm.top_half_space_material
                if not (isinstance(ths_mat, str) or \
                        isinstance(ths_mat, Fluid) ) :
                    self.warning("The top half-space is " + \
                                 "<b>solid</b>:\n<b>" + \
                                 "Mode computation is not possible</b>")
                    return # Undo change
                bhs_mat = self.geom_frm.bottom_half_space_material
                if not (isinstance(bhs_mat, str) or \
                        isinstance(bhs_mat, Fluid) ) :
                    self.warning("The bottom half-space is " + \
                                 "<b>solid</b>:\n<b>" + \
                                 "Mode computation is not possible</b>")
                    return # Undo change
                # OK
                self.__previous_path = self.geom_frm.structure_file_path
                self.__update_mode_computation_data(ths_mat, bhs_mat)
                self.shp_frm.reinit()     # Re-init Mode-shape Frame
                self.disp_frm.reinit()     # Re-init Dispersion Frame
                self.__wanted_tab = index # Confirmed change
            else :
                self.warning("The plate is not well defined:\n<b>" + \
                             "Mode computation is not possible</b>")
        elif index in (3,4) :
            self.not_yet_available() # Undo change
    #--------------------------------------------------------------------
    def __changeTab(self) :
        index = self.tabs.currentIndex()
        if index == self.__wanted_tab : return
        self.tabs.setCurrentIndex(self.__wanted_tab)
        self.update_statusBar()
    #--------------------------------------------------------------------
    def setCompletelyDefinedStructure(self, ok) :
        print(f"TraFiC_Application.setCompletelyDefinedStructure({ok})")
        self.__well_defined_plate = ok
    #--------------------------------------------------------------------
    def __update_mode_computation_data(self, ths_mat, bhs_mat) :
        layers = [ (self.geom_frm.material_of_layer(i), \
                    self.geom_frm.width_of_layer(i)) for i in \
                          range(1, self.geom_frm.number_of_layers+1) ]
        old_plate = isinstance(self.__discretized_plate, \
                               DiscretizedMultilayerPlate)
        if old_plate :
            old_plate = (ths_mat==self.__last_ths_mat) and \
                        (bhs_mat==self.__last_bhs_mat) and \
                        (layers==self.__last_layers)
        if old_plate : return
        # Update the discretized multilayer plate
        # First layer
        mat,w_mm = layers[0]
        fds = self.FDS_ORDER
        n = max(round(w_mm/self.DEF_DZ), fds+4 )
        # Other layers
        new_plate = DiscretizedMultilayerPlate(mat, 1e-3*w_mm, n, fds)
        for mat,w_mm in layers[1:] :
            n = max(round(w_mm/self.DEF_DZ), fds+4 )
            new_plate.add_discretizedLayer(mat, 1e-3*w_mm, n)
        # Top half-space
        if ths_mat == Layer_Frame.VACUUM :
            pass
        elif ths_mat == Layer_Frame.WALL :
            new_plate.set_left_fluid("Wall")
        else : # Fluid
            new_plate.set_left_fluid(ths_mat)
        # Bottom half-space
        if bhs_mat == Layer_Frame.VACUUM :
            pass
        elif bhs_mat == Layer_Frame.WALL :
            new_plate.set_right_fluid("Wall")
        else : # Fluid
            new_plate.set_right_fluid(bhs_mat)
        print(new_plate)
        self.__discretized_plate = new_plate
        self.__last_ths_mat = ths_mat
        self.__last_bhs_mat = bhs_mat
        self.__last_layers = layers
    #--------------------------------------------------------------------
    @property
    def discretized_plate(self) : return self.__discretized_plate   
    #--------------------------------------------------------------------
    def open_recent_file(self, number) :
        self.open_file(self.__recent_files[number])       
    #--------------------------------------------------------------------
    def open_file(self, filePath=None) :
        print(f"*** TraFiC_App.open_file({filePath}) ***")
        if filePath is None :
            filePath = QFileDialog.getOpenFileName( \
                                self,\
                                "Open:", \
                                self.PLATE_DIR, \
                                "Text files (*.txt);;All files (*)")[0]
            if filePath == "" : return # Cancel
        self.__previous_path = None # For reinit mode computation
        last_file_path = self.geom_frm.structure_file_path
        ok,result = self.geom_frm.read_from_file(filePath)
        if ok :
            if last_file_path != self.geom_frm.structure_file_path :
                # Effective change. Switch on geometry tab
                self.__wanted_tab = 0
                self.__changeTab()
        else :
            qmb = QMessageBox(self) 
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText(result.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Ok);
            rep = qmb.exec_()
            return
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def new_file(self) :      
        if self.geom_frm.modif :
            rep = self.approve("<b><i>Unsaved Changes<b></i>:\n" + \
                            "<b>Save before creating new structure?</b>")
            if rep == 1 : # Yes
                self.save()
            elif rep == -1 : # No
                pass
            else : # rep==0 [Cancel]
                return # Abort new file
        self.__previous_path = None # Modification for computation
        self.geom_frm.set_new_file()         
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save(self) :
        self.__previous_path = None # Modification for computation
        self.geom_frm.write_in_file()      
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def save_as(self) :
        self.__previous_path = None # Modification for computation
        self.geom_frm.write_in_file(ask_file_path=True) 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def closeEvent(self, event) :        
        if self.geom_frm.modif :
            rep = self.approve("<b><i>Unsaved Changes<b></i>:\n" + \
                               "<b>Save before closing?</b>")
            if rep == 1 : # Yes
                self.save()
                QMainWindow.closeEvent(self, event)
            elif rep == -1 : # No
                QMainWindow.closeEvent(self, event)
            else : # rep==0 [Cancel]
                event.ignore() # Abort closing
        else :
            QMainWindow.closeEvent(self, event)
    #--------------------------------------------------------------------
    def select_folder(self, number=None, folder_path=None, \
                      suggested_name=None, save_plate_file=True) :
        print("*** TraFiC_App.select_folder ***")
        if number is not None : # Recent Results Folder
            folder_path = self.__recent_folders[number]
        if folder_path is not None :
            # Relative path with / separators stored
            folder_path = os.path.relpath(folder_path)
            folder_path = folder_path.replace("\\","/")
            if os.path.isdir(folder_path) :
                if self.results_folder != folder_path :
                    self.results_folder = folder_path
                    self.update_recent_folders()
                    if save_plate_file :                  
                        self.geom_frm.write_in_file()
            else :
                msg = f"Folder '{folder_path}' does not exist" + \
                      "<br><br><b>Create this?</b>"
                qmb = QMessageBox(self)
                qmb.setWindowTitle("Warning!")
                qmb.setIcon(QMessageBox.Warning)
                qmb.setText(msg)
                qmb.setStandardButtons(QMessageBox.Ok | \
                                       QMessageBox.Cancel);
                qmb.setDefaultButton(QMessageBox.Ok)
                rep = qmb.exec_()
                if rep != QMessageBox.Ok :
                    folder_path = None # Interactive Selector
                else :
                    os.mkdir(folder_path)
        if folder_path is None :        
            msg = "Choose a results folder"
            if suggested_name is not None :
                msg += f" (Suggested name: '{suggested_name}')"
            msg += ":"
            folder_path = QFileDialog.getExistingDirectory( \
                                self,\
                                msg, \
                                self.RESULT_DIR, \
                                QFileDialog.ShowDirsOnly)
            if folder_path == "" : return False # Abort
            # Relative path with / separators stored
            folder_path = os.path.relpath(folder_path)
            folder_path = folder_path.replace("\\","/")
            if self.results_folder != folder_path :
                self.results_folder = folder_path
                self.update_recent_folders()
        self.folderMenu.setEnabled(True)
        return True
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def warning(self, msg) :
        qmb = QMessageBox(self) 
        qmb.setWindowTitle("Warning!")
        qmb.setIcon(QMessageBox.Warning)
        qmb.setText("<b><i>Warning:</i></b><br>" + \
                    msg.replace("\n","<br>"))
        qmb.setStandardButtons(QMessageBox.Ok);
        rep = qmb.exec_()   
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def not_yet_available(self, widget=None) :
        if isinstance(widget,QLineEdit) :
            widget.setText("")
        qmb = QMessageBox(self) 
        qmb.setWindowTitle("Sorry!")
        qmb.setIcon(QMessageBox.Information)
        qmb.setText("<b><i>Not yet available</i></b>")
        qmb.setStandardButtons(QMessageBox.Ok);
        rep = qmb.exec_() 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def approve(self, msg, no_button=True) :
        """Pop-up window 'Yes[/No]/Cancel'."""
        qmb = QMessageBox(self)
        if no_button :
            qmb.setWindowTitle("Yes / No / Cancel")
            qmb.setIcon(QMessageBox.Question)
            qmb.setText(msg.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Yes | QMessageBox.No\
                                                   | QMessageBox.Cancel)
            qmb.setDefaultButton(QMessageBox.No)
            rep = qmb.exec_()
            return 1 * (rep == QMessageBox.Yes) - \
                   1 * (rep == QMessageBox.No)
        else :
            qmb.setWindowTitle("Yes / Cancel")
            qmb.setIcon(QMessageBox.Question)
            qmb.setText(msg.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Yes | QMessageBox.Cancel)
            qmb.setDefaultButton(QMessageBox.Yes)
            rep = qmb.exec_()
            return 1 * (rep == QMessageBox.Yes)        
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def center(self):
        '''Window centered in the screen'''
        desktop = QApplication.desktop()
        n = desktop.screenNumber(self.cursor().pos())
        screen_center = desktop.screenGeometry(n).center()
        screen_center.setX(round(0.9*screen_center.x()))
        screen_center.setY(round(0.8*screen_center.y()))
        geo_window = self.frameGeometry()
        geo_window.moveCenter(screen_center)
        pos_TL = geo_window.topLeft()
        pos_TL.setX(max(pos_TL.x(),0))
        pos_TL.setY(max(pos_TL.y(),0))
        self.move(pos_TL)

