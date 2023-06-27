# Version 4.12 - 2021, October 9
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, \
                             QScrollArea, \
                             QPushButton, QFrame, QAction, QVBoxLayout, \
                             QHBoxLayout, QLineEdit, QTabWidget, \
                             QGridLayout, QComboBox, QLabel, QCheckBox, \
                             QMessageBox, QFileDialog, QInputDialog)
from PyQt5.QtGui import  QFont
from PyQt5.QtCore import Qt, pyqtSignal
from matplotlib.backends.backend_qt5agg \
    import FigureCanvasQTAgg as FigureCanvas
# to insert matplotlib figures
from matplotlib.figure import Figure
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# basic tools
import os
from os.path import dirname, basename, abspath
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# To access the tree structure of files and folders
if __name__ == "__main__" :
    import sys
    root = abspath("..")
    sys.path.append(root)
    import TraFiC_init
else :
    from TraFiC_init import root
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Materials
from MaterialClasses import Material, Fluid, ImportMaterialFromFile
from MaterialClasses import AnisotropicElasticSolid as AESolid
from MaterialClasses import IsotropicElasticSolid as IESolid
from MaterialClasses import TransverselyIsotropicElasticSolid as TIESolid
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# global parameters
LSZ = 11 # label size
USZ = 10 # unit size
VERBOSE = False # messages, or not
ZERONUM = 1e-7
def PRT(*a) :
    if VERBOSE : print(*a)
    else : pass
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Material_App(QMainWindow) :
    """Material_App is the main window of a graphical application for
       entering the parameters of an elastic material"""
    FONT = QFont("Arial", 10, QFont.Bold)
    #--------------------------------------------------------------------
    __mat_dir = root + "/Data/Materials"
    #--------------------------------------------------------------------
    # list of attributes
    __attrNames = tuple("_Material_App__"+n for n in \
                         ("fileLoc", "fileName", "AEmaterial", \
                          "IEmaterial", "TIEmaterial", "Fmaterial", \
                          "title", "rhoEd", "tabs", "stifEd", "sca", \
                          "modif", "isoFrm", "tiFrm", "fluFrm", \
                          "previousTab", "call", "chkbx", "chk_but") )
    #--------------------------------------------------------------------
    # default values for a new anisotropic elastic material
    __newparam = \
        {"rho":None, "c11":None, "c22":None, "c33":None, "c12":None, \
         "c13":None, "c23":None, "c44":None, "c55":None, "c66":None, \
         "c14":0.0, "c15":0.0, "c16":0.0, "c24":0.0, "c25":0.0, \
         "c26":0.0, "c34":0.0, "c35":0.0, "c36":0.0, "c45":0.0, \
         "c46":0.0, "c56":0.0}
    #--------------------------------------------------------------------
    def __init__(self, calling_by=None):
        self.__call = calling_by # None or instance of Layer_Frame
        # Superclass constructor calling
        QMainWindow.__init__(self)
        
        # Window Geometry
        self.setGeometry(300, 300, 1200, 600) # top left corner 300, 300;
                                              # width: 1200; height: 600
        # Window Title
        if self.__call is None :
            self.setWindowTitle('Tool for Material Management')
        else :
            self.setWindowTitle( \
                f'Editing medium of "{self.__call.name}"')
        # Menus
        self.__createMenus()
        
        # New materials
        self.__AEmaterial = AESolid(self.__newparam, "New material")
        self.__IEmaterial = IESolid(dict(), "New material")
        self.__TIEmaterial = TIESolid(dict(), "New material")
        self.__Fmaterial = Fluid(dict(), "New material")
        self.__fileLoc = Material_App.__mat_dir
        self.__fileName = "(New File)"
        self.__modif = False

        # Push button for come back to TraFiC
        if self.__call is None :
            OK_Btn = QLabel()
        else :
            OK_Btn = QPushButton("OK: come back to TraFiC", self)
            OK_Btn.released.connect(self.__quit)
            OK_Btn.setFont( self.FONT )

        # Push button for cancelling
        if self.__call is None :
            ccl_Btn = QLabel()
        else :
            ccl_Btn = QPushButton("Cancel", self)
            ccl_Btn.released.connect(self.cancel)
            ccl_Btn.setFont( self.FONT )

        # Push button for checking
        self.__chk_but = QPushButton("Check and Validate Changes", self)
        self.__chk_but.setFont( self.FONT )
        self.__chk_but.setFixedSize(250, 30)        
        self.__chk_but.setStyleSheet("color: rgb(150,0,0); " + \
                                "background-color: rgb(255,230,230) ;")
        self.__chk_but.released.connect( \
                                lambda:self.updateMaterial(True,True) )

        # Title (material name)
        nom = QLabel("Name: ",self)
        self.__title = QLabel(self.__AEmaterial.name,self)
        self.__title.setFont(QFont("Arial",18, QFont.Bold))
        self.__title.setStyleSheet( "color: rgb(155, 0, 155);")
        
        # Push button to change name
        modifBtn = QPushButton("Change Name", self)
        modifBtn.clicked.connect(self.__changeName)

        # Input field for the mass density
        self.__rhoEd = MassDensityFrame(self)

        # Complex stiffnesses
        self.__chkbx = QCheckBox(self)
        self.__chkbx.stateChanged.connect(self.complex_on_off)
        label_chkbx = QLabel("Complex stiffnesses", self)
        label_chkbx.setFont(QFont("Arial",12, QFont.Bold))
        label_chkbx.setStyleSheet( "color: rgb(0, 0, 200);")
        
        # tabs zone
        self.__tabs = QTabWidget(self)
        self.__previousTab = self.__tabs.currentIndex()
        self.__tabs.tabBarClicked.connect(self.__changeTabByClick)
        self.__tabs.currentChanged.connect(self.__changeTab)
        # "Anisotropic Elastic" tab
        self.__stifEd = StiffnessesFrame(self.__tabs)
        self.__stifEd.update(self.__AEmaterial)
        self.__sca = QScrollArea(self)
        self.__sca.setWidget( self.__stifEd )
        self.__tabs.addTab(self.__sca, "Anisotropic Elastic")
        # "Transversely isotropic Elastic" tab
        self.__tiFrm = TransIsoFrame(self.__tabs)
        self.__tabs.addTab(self.__tiFrm, \
                              "Transversely Isotropic Elastic")
        # "Isotropic Elastic" tab
        self.__isoFrm = IsotropicFrame(self.__tabs,self)
        self.__tabs.addTab(self.__isoFrm,"Isotropic Elastic")
        # "Fluid" tab
        self.__fluFrm = FluidFrame(self.__tabs)
        self.__tabs.addTab(self.__fluFrm,"Fluid")

        # "Central Widget" (required in any main window)
        centralW = QWidget()
        self.setCentralWidget(centralW)

        # Global Layout
        vbox = QVBoxLayout()
        centralW.setLayout(vbox)

        # First horizontal layout
        hbox = QHBoxLayout()
        hbox.addWidget(self.__rhoEd) # from the left...
        hbox.addStretch()
        hbox.addWidget(label_chkbx)
        hbox.addWidget(self.__chkbx)
        hbox.addStretch()
        hbox.addWidget(nom) 
        hbox.addWidget(self.__title)
        hbox.addStretch() 
        hbox.addWidget(modifBtn)     # ... to the right
        # Second horizontal layout
        hbox2 = QHBoxLayout()
        hbox2.addStretch() 
        hbox2.addWidget(OK_Btn)
        hbox2.addWidget(ccl_Btn) 
        hbox2.addWidget(self.__chk_but)

        # Global vertical layout
        vbox.addLayout(hbox)           # from the top...
        vbox.addWidget(self.__tabs)    # (tabs)
        vbox.addLayout(hbox2)
        vbox.addStretch()              # ... to the bottom

        # Display
        self.__update()
        if self.__call is not None :
            mat = self.__call.material
            if mat is not None :
                self.__loadMaterial(mat)
        self.show()
    #--------------------------------------------------------------------
    @property
    def isomat(self) : # access from frame self.__isoFrm
        return self.__IEmaterial
    #--------------------------------------------------------------------
    def __quit(self):
        '''To quit or return to the calling window'''
        if self.__call is None :
            reponse = self.__saveWarning(\
            "Do you <B>really</B> want to quit <B>without saving</B>?")
            if reponse : self.close()
        else :
            self.close()
    #--------------------------------------------------------------------
    def set_complex_on_off(self) :
        self.__chkbx.setChecked(not self.iscomplex)
    #--------------------------------------------------------------------
    def complex_on_off(self) :
        print("Switch real/complex values for stiffnesses.\n\t" + \
              f"[{self.iscomplex}]")
        for frm in ( self.__stifEd, self.__tiFrm, self.__isoFrm, \
                     self.__fluFrm) :
            frm.update_complex()
    #--------------------------------------------------------------------
    @property
    def iscomplex(self) :
        return self.__chkbx.isChecked()
    #--------------------------------------------------------------------
    def cancel(self):
        self.__call.end_edit_material(None)
        QMainWindow.close(self)
    #--------------------------------------------------------------------
    def before_closing(self):
        '''To properly close the window'''
        PRT("+++ MaterialEdit.before_closing +++")
        if self.__call is None :
            return True
        if self.__call is not None : 
            index = self.__tabs.currentIndex()
            if index == 0 :
                mat = self.__AEmaterial
            elif index == 1 :
                mat = self.__TIEmaterial
            elif index == 2 :
                mat = self.__IEmaterial
            elif index == 3 :
                mat = self.__Fmaterial
            else : # unexpected tab number
                raise ValueError(f"unexpected tab number '{index}'")
            ok, msg = mat.check()
            if ok :
                self.__call.end_edit_material(mat)
                return True
            else :
                msg = "The material is not complete:\n-> " + \
                      msg.strip().replace("\n","; ")
                msg += "\nCome back to TraFiC without material?" + \
                       "\n(Otherwise, continue to edit material)"
                rep = self.approve(msg)
                if rep == 1 :
                    self.__call.end_edit_material(None)
                    return True
                else :
                    return False # Cancel the closing
    #--------------------------------------------------------------------
##    def close(self): Useless: close() automatically calls closeEvent()
##        '''To properly close the window'''
##        PRT("+++ MaterialEdit.close +++")
##        ok = self.before_closing()
##        if ok : QMainWindow.close(self)
    #--------------------------------------------------------------------
    def closeEvent(self, event):
        '''To properly close the window with the X red button.'''
        PRT("+++ MaterialEdit.closeEvent +++")
        ok = self.before_closing()
        if ok : QMainWindow.closeEvent(self, event)
    #--------------------------------------------------------------------        
    def __openDataFile(self):
        PRT("+++ Material_App.__openDataFile +++")
        reponse = self.__saveWarning("Do you <B>really</B> want"+\
                            " to open a new file <B>without saving</B>?")
        if not reponse : return
        # Open a file selector :
        fpath,_ = QFileDialog.getOpenFileName(None,
                'Material File Selector',
                self.__mat_dir,
                "Text Files (*.txt)")
        if fpath == "" : return # No selected file
        newMat,file,direc = ImportMaterialFromFile(fpath)
        self.__loadMaterial(newMat, file, direc)
    #--------------------------------------------------------------------        
    def __loadMaterial(self, newMat, file=None, direc=None):
        if direc is None :
            direc = self.__mat_dir
        modif = file is None # Material imported and not saved
        if modif :
            file = "<untitled-1>.txt"
        if isinstance(newMat,AESolid) :
            PRT("Anisotropic material loaded")
            self.__AEmaterial = newMat
            self.__fileLoc  = direc
            self.__fileName = file
            self.__modif = modif
            self.__tabs.setCurrentWidget(self.__sca)
            self.__update()
        elif isinstance(newMat,IESolid) :
            PRT("Isotropic material loaded")
            self.__IEmaterial = newMat
            self.__fileLoc  = direc
            self.__fileName = file
            self.__modif = modif
            self.__tabs.setCurrentWidget(self.__isoFrm)
            self.__update()
        elif isinstance(newMat,TIESolid) :
            PRT("Transversely isotropic material loaded")
            self.__TIEmaterial = newMat
            self.__fileLoc  = direc
            self.__fileName = file
            self.__modif = modif
            self.__tabs.setCurrentWidget(self.__tiFrm)
            self.__update()
        elif isinstance(newMat,Fluid) :
            PRT("Fluid material loaded")
            self.__Fmaterial = newMat
            self.__fileLoc  = direc
            self.__fileName = file
            self.__modif = modif
            self.__tabs.setCurrentWidget(self.__fluFrm)
            self.__update()
        else :
            qmb = QMessageBox(self) 
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText("The loaded material has not a known type!")
            qmb.setStandardButtons(QMessageBox.Ok);
            rep = qmb.exec_()
    #--------------------------------------------------------------------
    def __saveWarning(self,msg) :
        '''Ask confirmation if a modification is done.'''
        PRT("+++ Material_App.__saveWarning +++")
        if self.__modif :
            qmb = QMessageBox(self)
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText(msg.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel);
            qmb.setDefaultButton(QMessageBox.Cancel)
            rep = qmb.exec_()
            return rep == QMessageBox.Ok
        return True
    #--------------------------------------------------------------------
    @staticmethod
    def not_yet_available(widget) :
        if isinstance(widget,QLineEdit) :
            widget.setText("")
        qmb = QMessageBox(widget) 
        qmb.setWindowTitle("Sorry!")
        qmb.setIcon(QMessageBox.Information)
        qmb.setText("<B><I>Not yet available</I></B>")
        qmb.setStandardButtons(QMessageBox.Ok);
        rep = qmb.exec_()
    #--------------------------------------------------------------------
    def approve(self,msg) :
        '''Pop-up window 'Yes or No?'.'''
        PRT("+++ Material_App.approve +++")
        qmb = QMessageBox(self)
        qmb.setWindowTitle("Yes or no ?")
        qmb.setIcon(QMessageBox.Question)
        qmb.setText(msg.replace("\n","<br>"))
        qmb.setStandardButtons(QMessageBox.Yes | QMessageBox.No\
                               | QMessageBox.Cancel);
        qmb.setDefaultButton(QMessageBox.No)
        rep = qmb.exec_()
        return 1 * (rep == QMessageBox.Yes) - 1 * (rep == QMessageBox.No)
    #--------------------------------------------------------------------    
    def updateMaterial(self,verif,ctab) :
        if ctab : no = self.__tabs.currentIndex()
        else : no = self.__previousTab # before tab change
        PRT("+++ Material_App.updateMaterial [{}] +++".format(no))
        if no == 0 : # Anisotropic
            prm = self.__stifEd.readStiffnesses()
        elif no == 1 : # Transversely Isotropic 
            prm = self.__tiFrm.readStiffnesses()
        elif no == 2 : # Isotropic
            prm = self.__isoFrm.readAll()
        elif no == 3 : # Fluid
            prm = self.__fluFrm.readParameters()
        else : # unexpected tab number
            return
        # Mass density
        prm["rho"] = self.__rhoEd.readDensity()

        PRT("+++ Material_App.updateMaterial [fin de lecture] +++")
        # Updating the selected material... 
        if no == 0 : # Anisotropic 
            self.__AEmaterial = AESolid(prm, self.__title.text())
            test,msg = self.__AEmaterial.check()
        elif no == 1 : # Transversely Isotropic 
            self.__TIEmaterial = TIESolid(prm, self.__title.text())
            test,msg = self.__TIEmaterial.check()
        elif no == 2 : # Isotropic
            self.__IEmaterial = IESolid(prm, self.__title.text())
            test,msg = self.__IEmaterial.check()
        elif no == 3 : # Fluid
            self.__Fmaterial = Fluid(prm, self.__title.text())
            test,msg = self.__Fmaterial.check()
        if verif and not test :
            qmb = QMessageBox(self) 
            qmb.setWindowTitle("Warning!")
            qmb.setIcon(QMessageBox.Warning)
            qmb.setText("<B><I>Incomplete material</I></B>: "+\
                         msg.replace("\n","<br>"))
            qmb.setStandardButtons(QMessageBox.Ok);
            rep = qmb.exec_()
        self.__update()
        return test
    #--------------------------------------------------------------------        
    def __save(self):
        '''Saving in  a file.'''
        test = self.updateMaterial(True,True)
        if test :
            filePath = QFileDialog.getSaveFileName(self,\
                "Save as:",\
                self.__fileLoc,\
                "Text files (*.txt);;All files (*)")
            if isinstance(filePath,tuple) :
                filePath = filePath[0] # PyQt4/PyQt5
            no = self.__tabs.currentIndex()
            PRT("__save, tab no.{}, path : {}".format(no,filePath))
            if no == 0 : # Anisotropic
                self.__AEmaterial.save(filePath)
            elif no == 1 : # Transversely Isotropic
                self.__TIEmaterial.save(filePath)
            elif no == 2 : # Isotropic
                self.__IEmaterial.save(filePath)
            else : # n=3 Fluid
                self.__Fmaterial.save(filePath) 
            
            self.__fileLoc = dirname(filePath)
            self.__fileName = basename(filePath)
            self.__modif = False
            self.__update()
        return
    #--------------------------------------------------------------------    
    @property
    def __shortpath(self) :
        lenloc = len(self.__fileLoc)
        lennam = len(self.__fileName)        
        if lenloc+lennam <= 99 :
            path = self.__fileLoc+"/"+self.__fileName
        else :
            dm = max( 47 - lennam//2 , 5 )
            path = self.__fileLoc[:dm]+"[...]"+self.__fileLoc[-dm:]+\
                   "/"+self.__fileName
        if self.__modif == True : path = "***"+path+"***"
        return path
    #--------------------------------------------------------------------        
    def __update(self) :
        """Display updating."""
        PRT("+++ Material_App.__update +++")
        no = self.__tabs.currentIndex()
        if no == 0 :
            self.__title.setText(self.__AEmaterial.name)
            self.__rhoEd.update(self.__AEmaterial)
            self.__stifEd.update(self.__AEmaterial)
            verif,_ = self.__AEmaterial.check()
        if no == 1 :
            self.__title.setText(self.__TIEmaterial.name)
            self.__rhoEd.update(self.__TIEmaterial)
            self.__tiFrm.update(self.__TIEmaterial)
            verif,_ = self.__TIEmaterial.check()
        if no == 2 :
            self.__title.setText(self.__IEmaterial.name)
            self.__rhoEd.update(self.__IEmaterial)
            self.__isoFrm.update(self.__IEmaterial)
            verif,_ = self.__IEmaterial.check()
        elif no == 3 :
            self.__title.setText(self.__Fmaterial.name)
            self.__rhoEd.update(self.__Fmaterial)
            self.__fluFrm.update(self.__Fmaterial)
            verif,_ = self.__Fmaterial.check()
        if verif : # Green button      
            self.__chk_but.setStyleSheet("color: rgb(0,100,0); " + \
                            "background-color: rgb(230,255,230) ;")
        else : # Red button
            self.__chk_but.setStyleSheet("color: rgb(150,0,0); " + \
                            "background-color: rgb(255,230,230) ;")
        self.statusBar().showMessage(self.__shortpath)
    #--------------------------------------------------------------------
    def __newMaterial(self):
        response = self.__saveWarning( \
            "Do you <B>really</B> want to creat "\
            "a new material <B>without saving</B> ?")
        if not response : return
        types = ("Anisotropic elastic", \
                 "Transversely isotropic elastic", \
                 "Isotropic elastic", "Fluid")
        choice = dict()
        for i,e in enumerate(types) :
            choice[e] = i
        item, ok = QInputDialog.getItem(None, "Type of the new material?",\
                                              "Material", types, 0, False)
        if not ok : return
        no = choice[item]
        self.__tabs.setCurrentIndex(no)
        if no == 0 :
            self.__AEmaterial = AESolid(self.__newparam, "New material")
        elif no == 1 :
            self.__TIEmaterial = TIESolid(dict(), "New material")
        elif no == 2 :
            self.__IEmaterial = IESolid(dict(), "New material")
        elif no == 3 :
            self.__Fmaterial = Fluid(dict(), "New material")
        self.__fileName = "(New File)"
        self.__modif = True 
        self.__update() 
    #--------------------------------------------------------------------
    def detectedModif(self) :
        self.__modif = True        
        self.statusBar().showMessage(self.__shortpath)
    #--------------------------------------------------------------------
    def __changeName(self):
        """Open a window to change the material name"""
        text, result = QInputDialog.getText(self,\
                         "Changing the material name",
                         "Type the new material name:")
        if result:
            self.detectedModif()
            self.__title.setText(text)  
            self.updateMaterial(False,True)
    #--------------------------------------------------------------------    
    def __createMenus(self):
        # Menus
        menubar = self.menuBar()
        # File
        menuF = menubar.addMenu('File')
        # > New material
        a = QAction("New material", self)
        a.triggered.connect(self.__newMaterial)
        a.setShortcut('Ctrl+N')
        a.setStatusTip("Create a new material")
        menuF.addAction(a)        
        # > Open
        a = QAction("Open a material file", self)
        a.triggered.connect(self.__openDataFile)
        a.setShortcut('Ctrl+O')
        a.setStatusTip("Open a text file containing the " + \
                       "characteristics of a material")
        menuF.addAction(a)
        # > Save as
        a = QAction('Save as...', self)
        a.triggered.connect(self.__save)
        a.setShortcut('Ctrl+S')
        a.setStatusTip("Save the material parameters in a file")
        menuF.addAction(a)
        # > Quit
        if self.__call is None :
            a = QAction('Quit', self)
            a.setStatusTip("Quit the software")
        else :
            a = QAction("Come back to TraFiC", self)
            a.setStatusTip("Come bac to the TraFiC software")
        a.triggered.connect(self.__quit)
        a.setShortcut('Ctrl+Q')
        menuF.addAction(a)
    #--------------------------------------------------------------------
    def __changeTabByClick(self, index) :
        if index == -1 : return # No effect
        ptab = self.__previousTab
        if index == ptab : return # No change
        PRT(f"From tab {ptab} to tab {index} by click")
        self.updateMaterial(False,False) # Before changes
        if ptab == 2 :
            if index == 1 : # Conversion Iso -> TO
                msg = "Do you want to convert the <I>Isotropic " + \
                      "Elastic material</I> to <I>Transversely " + \
                      "isotropic Elastic material</I>?"
                rep = self.approve(msg)
                if rep == 1 :
                    PRT("Conversion Iso -> IT ...")
                    self.__TIEmaterial = self.__IEmaterial.export()
                    PRT("...effectuée")
            elif index == 0 : # Conversion Iso -> Ani
                msg = "Do you want to convert the <I>Isotropic " + \
                      "Elastic material</I> to " + \
                      "<I>Anisotropic Elastic material</I>?"
                rep = self.approve(msg)
                if rep == 1 :
                    PRT("Conversion Iso -> Ani ...")
                    self.__AEmaterial = \
                                 self.__IEmaterial.export().export()
            else : # Iso -> other
                pass
        elif ptab == 1 and index == 0 : # Conversion TI -> Ani
            msg = "Do you want to convert the <I>Transversely " + \
                  "Isotropic Elastic material" + \
                  "</I> to <I>Anisotropic Elastic material</I>?"
            rep = self.approve(msg)
            if rep == 1 :
                PRT("Conversion IT -> Ani ...")
                self.__AEmaterial = self.__TIEmaterial.export()
        self.__tabs.setCurrentIndex(index)
    #--------------------------------------------------------------------
    def __changeTab(self) :
        ptab = self.__previousTab
        ctab = self.__tabs.currentIndex()
        if ctab == ptab : return # no change
        PRT(f"From tab {ptab} to tab {ctab} without click")
        self.updateMaterial(False,False) # Before changes            
        self.__update() # Display refreshment
        self.__previousTab = ctab
        return
    #--------------------------------------------------------------------        
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "Material_App" : 
            if name in Material_App.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================       
class MassDensityFrame(QFrame) :
    """Petit cadre pour une masse volumique"""
    # liste des attributs
    __attrNames = ("_MassDensityFrame__edit",)
    #--------------------------------------------------------------------    
    def __init__(self,parent) :
        QFrame.__init__(self,parent)
        vbox = QVBoxLayout()
        label = QLabel("Mass density")
        label.setFixedSize(120,23)
        vbox.addWidget(label)
        hbox = QHBoxLayout()
        # rho
        f0 = Figure(facecolor = "none") 
        unit0 = FigureCanvas(f0)
        unit0.setParent(self)
        unit0.setFixedSize(20,20)
        f0.clear()
        f0.suptitle(r"$\rho$",fontsize=LSZ,
                      x=1.0, y=0.5, 
                      horizontalalignment='right',
                      verticalalignment='center')
        unit0.draw()
        hbox.addWidget(unit0)
        # Editing area
        self.__edit = QLineEdit()
        self.__edit.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        self.__edit.setFont(QFont("Times New Roman",12, QFont.Bold))
        self.__edit.setStyleSheet("""QLineEdit {color: rgb(0, 0, 200);}""")
        self.__edit.setFixedSize(83,20)
        hbox.addWidget(self.__edit)
        
        self.__edit.editingFinished.connect(self.__modifiedRho)
        # Unit
        unit = QLabel(self)
        unit.setFixedSize(50,20)
        unit.setText("mg/mm³")
        unit.setFont( QFont("Arial", USZ) )
        hbox.addWidget(unit)
        hbox.setSpacing(1)
        try : # PyQt5
            hbox.setContentsMargins(1,1,1,1)
        except : # PyQt4
            hbox.setMargin(1)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
        self.adjustSize()
    #--------------------------------------------------------------------    
    def __modifiedRho(self) :
        PRT("Modified mass density...")
        try :
            par = self.parent()
            nb = 1
            while not isinstance(par,Material_App) and nb <= 20 :
                par = par.parent()
                nb += 1
            if nb == 21 : raise
            par.detectedModif()
        except :
            PRT("The transmission of the information failed.")
        return
    #--------------------------------------------------------------------
    def readDensity(self) :
        text = self.__edit.text()
        try :
            value = float(text)
            return 1e3*value
        except :
            return None
    #--------------------------------------------------------------------
    def update(self, material) :
        test,msg = Material.check(material)
        if test :
            self.__edit.setText("{:.5f}".format(1e-3*material.rho))
        else :
            self.__edit.setText("")
    #--------------------------------------------------------------------
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "MassDensityFrame" : 
            if name in MassDensityFrame.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value) 
#========================================================================
#========================================================================      
class StiffnessFrame(QFrame) :
    """Little frame for typing stiffness value"""
    # liste des attributs
    __attrNames = tuple( "_StiffnessFrame__" + n for n in ("name",\
                        "value", "readOnly", "edit", "label", "canvas") )
    WTOT = 180   # Total Width
    HTOT = 68 # Total Height
    HLAB = 28 # Label Height
    WUNI = 35 # Unit Width
    HUNI = 20 # Unit Height
    #--------------------------------------------------------------------    
    def __init__(self, parent, name) :
        QFrame.__init__(self, parent)
        self.setStyleSheet("background:transparent")
        self.__name = name # stiffness name
        self.__value = None # value
        self.__readOnly = None
        self.__edit = QLineEdit()
        self.setFixedSize(self.WTOT,self.HTOT)
        vbox = QVBoxLayout()
        # Equation with matplotlib        
        hboxup = QHBoxLayout()
        self.__label = Figure(facecolor = "none")
        self.__canvas = FigureCanvas(self.__label)
        self.__canvas.setParent(self)
        self.__canvas.setFixedSize(self.WTOT,self.HLAB)
        self.setLabel(r"$c_{"+name[-2:]+r"}$")
        hboxup.addWidget(self.__canvas)
        hboxup.addStretch()
        vbox.addLayout(hboxup)
        
        hbox = QHBoxLayout()
        # Typing area (self.__edit)
        self.__edit.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        self.__edit.setFont(QFont("Times New Roman",12, QFont.Bold))
        self.__edit.setStyleSheet("color: rgb(0, 0, 200);")
        self.__edit.setFixedSize(self.WTOT-self.WUNI-20,self.HUNI)
        hbox.addWidget(self.__edit)
        self.__edit.editingFinished.connect(self.__modifiedCij)
        # Equation with matplotlib
        fig2 = Figure(facecolor = "none") # fond transparent
        unit = FigureCanvas(fig2)
        unit.setParent(self)
        unit.setFixedSize(self.WUNI,self.HUNI)
        fig2.suptitle(r"GPa",fontsize=USZ,
                      x=0.0, y=0.5, 
                      horizontalalignment='left',
                      verticalalignment='center')
        unit.draw()
        hbox.addWidget(unit)
        vbox.addLayout(hbox)
        self.setLayout(vbox)
    #--------------------------------------------------------------------
    def __modifiedCij(self) :
        PRT("Modified stiffness...")
        try :
            par = self.parent()
            nb = 1
            while not isinstance(par,Material_App) and nb <= 20 :
                par = par.parent()
                nb += 1
            if nb == 21 : raise
            par.detectedModif()
        except :
            PRT("The transmission of information failed.")
        return 
    #--------------------------------------------------------------------
    @property
    def name(self) : return self.__name
    #--------------------------------------------------------------------
    def setText(self,texte) :
        if texte == "" or texte == None :
            self.__edit.clear()
        else :
            self.__edit.setText(texte)
    #--------------------------------------------------------------------
    def setLabel(self,new_label,size=LSZ) :
        self.__label.clear()
        self.__label.suptitle(new_label,fontsize=size,
                      x=0.0, y=0.0, 
                      horizontalalignment='left',
                      verticalalignment='bottom')
        self.__canvas.draw()
    #--------------------------------------------------------------------
    @property
    def text(self) :
        return self.__edit.text()
    #--------------------------------------------------------------------
    def setReadOnly(self,VF = True ) :
        self.__readOnly = VF
        self.__edit.setReadOnly(self.__readOnly)
        if VF :
            self.__edit.setStyleSheet("color: rgb(0, 0, 0); " + \
                           "background: rgb(240, 240, 240); " + \
                         "selection-background-color: rgb(100,100,100);")
        else :
            self.__edit.setStyleSheet("color: rgb(0, 0, 255); " + \
                             "background: rgb(255, 255, 255); " + \
                           "selection-background-color: rgb(0,255,255);")
    #--------------------------------------------------------------------        
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "StiffnessFrame" : 
            if name in StiffnessFrame.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================
class StiffnessesFrame(QFrame) :
    """Array of stiffnesses for an anisotropic material."""
    # liste des attributs
    __attrNames = tuple("_StiffnessesFrame__"+n for n in\
                  ("stiffnessFrames", "mw", "cplx", "imlab", "imedi", \
                   "imuni", "imlay", "rotedi") )
    GBAT = "Set global attenuation: "
    DEFH = 28
    DEFW = 50
    #--------------------------------------------------------------------
    def __init__(self, parent) :
        QFrame.__init__(self, parent)
        self.__mw = parent.parent()  # Main Window
        self.__cplx = False  # Real stiffnesses by default
        self.__stiffnessFrames = [ [ \
                            StiffnessFrame(self,"C{}{}".format(i,j)) \
                              for j in range(i,7) ] for i in range(1,7) ]
        # Layout and widgets for setting the imaginary part with respect
        #     to the real part.        
        self.__imlab = QLabel(self)
        self.__imedi = None
        self.__imuni = QLabel(self)
        for wdg,w in ((self.__imlab,150), (self.__imuni,20)) :
            self.__design(wdg, h=self.DEFH, w=w)
        self.__imlay = QGridLayout()
        self.__imlay.addWidget(self.__imlab, 0, 0)
        self.__imlay.addWidget(self.__imuni, 0, 2)
        right = QHBoxLayout()
        right.addStretch()
        self.__imlay.addLayout(right, 0, 3)
        self.__update_imag_percent(forced=True)
        # Layout and widgets for rotate with respect to axis #3
        rot_lab = QLabel("Rotate with respect to axis #3: ", self)
        self.__rotedi = QLineEdit(self)
        self.__rotedi.textEdited.connect(
                     lambda text,wdg=self.__rotedi: \
                            Material_App.not_yet_available(wdg) )
        rot_uni = QLabel("°", self)
        for wdg,w in ((rot_lab,None), (self.__rotedi,40), \
                      (rot_uni,None)) :
            self.__design(wdg, h=self.DEFH, w=w)
        rot_lay = QGridLayout()
        rot_lay.addWidget(rot_lab, 0, 0)
        rot_lay.addWidget(self.__rotedi, 0, 1)
        rot_lay.addWidget(rot_uni, 0, 2)
        right = QHBoxLayout()
        right.addStretch()
        rot_lay.addLayout(right, 0, 3)
            
        # Global Layout
        sf = self.__stiffnessFrames
        boxes = QGridLayout()
        for i,row in enumerate(sf,1) :         
            for j,f in enumerate(row,i) :
                boxes.addWidget(f,i,j,1,1)
        boxes.addLayout(self.__imlay,5,1,1,2)
        boxes.addLayout(rot_lay,6,1,1,3)

        # Stretches
        vb = QVBoxLayout()
        hb = QHBoxLayout()
        hb.addLayout(boxes)
        hb.addStretch()
        vb.addLayout(hb)
        vb.addStretch()
        self.setLayout(vb)
    #--------------------------------------------------------------------
    @staticmethod
    def __design(widget, w=None, h=None) :
        try :
            widget.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
            widget.setFont(QFont("Times New Roman",12, QFont.Bold))
            widget.setStyleSheet("color: rgb(0, 0, 0);")
            if w is None :
                if h is None :
                    pass
                else :
                    widget.setFixedHeight(h)
            elif h is None :
                widget.setFixedWidth(w)
            else :
                widget.setFixedSize(w, h)
        except Exception as err :
            print(f"Error in StiffnessesFrame.__design:\n\t{err}.")
    #--------------------------------------------------------------------
    def update_complex(self) :
        self.__update_imag_percent()       
    #--------------------------------------------------------------------
    def __update_imag_percent(self, forced=False) :
        if not forced and self.__cplx == self.__mw.iscomplex :
            # no change
            return
        self.__cplx = self.__mw.iscomplex
        # Removing previous widget
        self.__imedi = None
        lay_item = self.__imlay.itemAtPosition(0, 1)
        if lay_item is not None :
            if lay_item.widget() is not None: # widget to remove
                widgetToRemove = lay_item.widget()
                widgetToRemove.setParent(None)
                self.__imlay.removeWidget(widgetToRemove)
        if self.__cplx : # Complex values            
            self.__imlab.setText(self.GBAT)         
            self.__imedi = QLineEdit(self)
            self.__imedi.textEdited.connect( \
                lambda text, wdg=self.__imedi: \
                       Material_App.not_yet_available(wdg) )          
            self.__imuni.setText(" %")
        else : # Real values          
            self.__imlab.setText(" "*len(self.GBAT))
            self.__imedi = QLabel(self)            
            self.__imuni.setText("  ")
        self.__imlay.addWidget(self.__imedi, 0, 1)
        self.__design(self.__imedi, h=self.DEFH, w=self.DEFW)
    #--------------------------------------------------------------------        
    def update(self, material) :
        sf = self.__stiffnessFrames
        if isinstance(material, AESolid) :
            if material.iscomplex :
                if not self.__cplx :
                    self.__mw.set_complex_on_off()
                for i,row in enumerate(sf,1) :         
                    for j,f in enumerate(row,i) :
                        cij = material.c(i,j)
                        if isinstance(cij,float) and \
                           ( i!=j or cij >= 0.0) :
                            f.setText("{:.3f}".format(1e-9*cij))
                        elif isinstance(cij,complex) and \
                           ( i!=j or cij.real >= 0.0) :
                            f.setText("{:.3f}".format(1e-9*cij))
                        else :
                            f.setText("")
            else : # Real stiffnesses
                if self.__cplx :
                    self.__mw.set_complex_on_off()                
                for i,row in enumerate(sf,1) :         
                    for j,f in enumerate(row,i) :
                        cij = material.c(i,j)
                        if isinstance(cij,float) and \
                           ( i!=j or cij >= 0.0) :
                            f.setText("{:.3f}".format(1e-9*cij))
                        else :
                            f.setText("")
        else : # undefined material
            for i,row in enumerate(sf,1) :         
                for j,f in enumerate(row,i) :
                    if i != j and ( i >= 4 or j >= 4 ) :
                        f.setText("0.000")
                    else :
                        f.setText("")
    #--------------------------------------------------------------------
    def readStiffnesses(self) :
        """catches the entered stiffnesses."""
        sf = self.__stiffnessFrames
        param = dict()
        for i,row in enumerate(sf,1) :         
            for j,f in enumerate(row,i) :
                nom,cij = f.name, f.text
                new_complex = False
                try :
                    cij = float(cij)
                    param[nom] = 1e9*cij
                except :
                    try :
                        cij = complex(cij)
                        param[nom] = 1e9*cij
                        new_complex = True
                    except :
                        pass
                if new_complex and not self.__cplx :
                    msg = "Do you really want to switch to " + \
                          "complex stiffnesses?"
                    rep = self.__mw.approve(msg)
                    if rep == 1 : # Yes
                        self.__mw.set_complex_on_off()
                    else : # -1 [No] or 0 [Cancel]
                        pass
        return param
    #--------------------------------------------------------------------
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "StiffnessesFrame" :
            if name in StiffnessesFrame.__attrNames :
                object.__setattr__(self,name,value) 
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================      
class IsotropicFrame(QFrame) :
    __attrNames = tuple("_IsotropicFrame__"+n for n in\
                  ("choice", "par1", "par2", "a11", "a12", "a21", "a22", \
                   "a31", "a32", "a41", "a42", "t1", "t2", "t3", "t4", \
                   "frm0", "frm1", "frm2", "frm3", "frm4", \
                   "previousIndex", "cplx", "mw") )
    __titles = ( ("Lamé Coefficients (c12,c44)", "Stiffnesses", \
                  "Stiffnesses", "Young's Modulus & Poisson's Ratio", \
                  "Velocities"), \
                 ("Stiffnesses c11 & c12", "Lamé Coefficients", \
                  "Stiffnesses", "Young's Modulus & Poisson's Ratio", \
                  "Velocities"),\
                 ("Stiffnesses c11 & c44", "Lamé Coefficients", \
                  "Stiffnesses", "Young's Modulus & Poisson's Ratio", \
                  "Velocities"),\
                 ("Young's Modulus & Poisson's Ratio", \
                  "Lamé Coefficients", "Stiffnesses", "Stiffnesses", \
                  "Velocities"),\
                 ("Velocities ", "Lamé Coefficients", "Stiffnesses", \
                  "Stiffnesses", "Young's Modulus & Poisson's Ratio" ) )
    __names1 = ( (r"$c_{12}\,|\,\lambda$", r"$c_{11}{=}\lambda{+}2\mu$", \
                  r"$c_{11}{=}\lambda+2\,\mu$", \
                  r"$E{=}2\mu{+}\frac{\lambda\,\mu}{\lambda{+}\mu}$", \
                  r"$c_{L}{=}\sqrt{\frac{\lambda+2\,\mu}{\rho}}$"), \
                (r"$c_{11}$",r"$\lambda=c_{12}$",r"$c_{11}$", \
                  r"$E{=}\frac{c_{11}^2{+}c_{11}c_{12}{-}" + \
                         r"2c_{12}^2}{c_{11}{+}c_{12}}$", \
                  r"$c_{L}{=}\sqrt{\frac{c_{11}}{\rho}}$"), \
                (r"$c_{11}$",r"$\lambda=c_{11}-2\,c_{44}$",r"$c_{11}$", \
                  r"$E{=}\frac{c_{44}(3\,c_{11}-4\,c_{44})}" + \
                         r"{c_{11}-c_{44}}$", \
                  r"$c_{L}{=}\sqrt{\frac{c_{11}}{\rho}}$"), \
                (r"$E\,|\,Y$", \
                 r"$\lambda=\frac{\nu\,E}{(1+\nu)(1-2\,\nu)}$", \
                  r"$c_{11}{=}\frac{(1-\nu)\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                  r"$c_{11}{=}\frac{(1-\nu)\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                  r"$c_{L}{=}\sqrt{\frac{(1-\nu)\,(E/\rho)}" + \
                         r"{(1{+}\nu)(1{-}2\nu)}}$"), \
                (r"$c_L$",r"$\lambda=\rho\,(c_L^2{-}2\,c_T^2)$", \
                  r"$c_{11}=\rho\,c_L^2$",r"$c_{11}=\rho\,c_L^2$", \
                  r"$E{=}\frac{\rho\,c_T^2\,(3\,c_L^2-4\,c_T^2)}" + \
                         r"{c_L^2-c_T^2}$") )
    __names1_complex = \
              ( (r"$c_{12}\,|\,\lambda$", r"$c_{11}{=}\lambda{+}2\mu$", \
                  r"$c_{11}{=}\lambda+2\,\mu$", \
                  r"$E{=}2\mu{+}\frac{\lambda\,\mu}{\lambda{+}\mu}$", \
                  r"$c_{L}{=}\sqrt{\frac{\mathrm{Re}(" + \
                                r"\lambda+2\,\mu)}{\rho}}$"), \
                (r"$c_{11}$",r"$\lambda=c_{12}$",r"$c_{11}$", \
                  r"$E{=}\frac{c_{11}^2{+}c_{11}c_{12}{-}" + \
                              r"2c_{12}^2}{c_{11}{+}c_{12}}$", \
                  r"$c_{L}{=}\sqrt{\frac{\mathrm{Re}(c_{11})}{\rho}}$"), \
                (r"$c_{11}$",r"$\lambda=c_{11}-2\,c_{44}$",r"$c_{11}$", \
                  r"$E{=}\frac{c_{44}(3\,c_{11}-4\,c_{44})}" + \
                              r"{c_{11}-c_{44}}$", \
                  r"$c_{L}{=}\sqrt{\frac{\mathrm{Re}(c_{11})}{\rho}}$"), \
                (r"$E\,|\,Y$", \
                 r"$\lambda=\frac{\nu\,E}{(1+\nu)(1-2\,\nu)}$", \
                  r"$c_{11}{=}\frac{(1-\nu)\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                  r"$c_{11}{=}\frac{(1-\nu)\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                  r"$c_{L}{=}\sqrt{\mathrm{Re}\!\left(" + \
                         r"\frac{(1-\nu)\,(E/\rho)}" + \
                         r"{(1{+}\nu)(1{-}2\nu)}\right)}$"), \
                (r"$c_L$",r"$\lambda=\rho\,(c_L^2{-}2\,c_T^2)$", \
                  r"$c_{11}=\rho\,c_L^2$",r"$c_{11}=\rho\,c_L^2$", \
                  r"$E{=}\frac{\rho\,c_T^2\,(3\,c_L^2-4\,c_T^2)}" + \
                         r"{c_L^2-c_T^2}$") )
    __units1 = ( (r"GPa",r"GPa",r"GPa",r"GPa",r"mm/µs"),\
                 (r"GPa",r"GPa",r"GPa",r"GPa",r"mm/µs"),\
                 (r"GPa",r"GPa",r"GPa",r"GPa",r"mm/µs"),\
                 (r"GPa",r"GPa",r"GPa",r"GPa",r"mm/µs"),\
                 (r"mm/µs",r"GPa",r"GPa",r"GPa",r"GPa") )
    __names2 = ( (r"$c_{44}\,|\,\mu$", r"$c_{12}=\lambda$", \
                 r"$c_{44}=\mu$",\
                 r"$\nu=\frac{\lambda}{2(\lambda{+}\mu)}$", \
                 r"$c_{T}{=}\sqrt{\frac{\mu}{\rho}}$"), \
                (r"$c_{12}$",r"$\mu{=}\frac{c_{11}-c_{12}}{2}$", \
                 r"$c_{44}=\frac{c_{11}-c_{12}}{2}$", \
                 r"$\nu=\frac{c_{12}}{c_{11}{+}c_{12}}$", \
                 r"$c_{T}{=}\sqrt{\frac{c_{11}-c_{12}}{2\,\rho}}$"), \
                (r"$c_{44}$", r"$\mu=c_{44}$", \
                 r"$c_{12}=c_{11}-2\,c_{44}$", \
                 r"$\nu=\frac{c_{11}-2\,c_{44}}{2\,(c_{11}{-}c_{44})}$",\
                 r"$c_{T}{=}\sqrt{\frac{c_{44}}{\rho}}$"), \
                (r"$\nu$",r"$\mu=\frac{E}{2\,(1+\nu)}$", \
                 r"$c_{12}{=}\frac{\nu\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                 r"$c_{44}=\frac{E}{2\,(1+\nu)}$", \
                 r"$c_{T}{=}\sqrt{\frac{E}{2\,\rho\,(1+\nu)}}$"), \
                (r"$c_T$",r"$\mu=\rho\,c_T^2$", \
                 r"$c_{12}=\rho\,(c_L^2{-}2\,c_T^2)$", \
                 r"$c_{44}=\rho\,c_T^2$", \
                 r"$\nu=\frac{c_L^2-2\,c_T^2}{2\,(c_L^2-c_T^2)}$") )
    __names2_complex = \
              ( (r"$c_{44}\,|\,\mu$", r"$c_{12}=\lambda$", \
                 r"$c_{44}=\mu$",\
                 r"$\nu=\frac{\lambda}{2(\lambda{+}\mu)}$", \
                 r"$c_{T}{=}\sqrt{\frac{\mathrm{Re}(\mu)}{\rho}}$"), \
                (r"$c_{12}$",r"$\mu{=}\frac{c_{11}-c_{12}}{2}$", \
                 r"$c_{44}=\frac{c_{11}-c_{12}}{2}$", \
                 r"$\nu=\frac{c_{12}}{c_{11}{+}c_{12}}$", \
                 r"$c_{T}{=}\sqrt{\frac{\mathrm{Re}(" + \
                                  r"c_{11}-c_{12})}{2\,\rho}}$"), \
                (r"$c_{44}$", r"$\mu=c_{44}$", \
                 r"$c_{12}=c_{11}-2\,c_{44}$", \
                 r"$\nu=\frac{c_{11}-2\,c_{44}}{2\,(c_{11}{-}c_{44})}$",\
                 r"$c_{T}{=}\sqrt{\frac{\mathrm{Re}(c_{44})}{\rho}}$"), \
                (r"$\nu$",r"$\mu=\frac{E}{2\,(1+\nu)}$", \
                 r"$c_{12}{=}\frac{\nu\,E}{(1{+}\nu)(1{-}2\nu)}$", \
                 r"$c_{44}=\frac{E}{2\,(1+\nu)}$", \
                 r"$c_{T}{=}\sqrt{\mathrm{Re}\!\left(" + \
                            r"\frac{E}{2\,\rho\,(1+\nu)}\right)}$"), \
                (r"$c_T$",r"$\mu=\rho\,c_T^2$", \
                 r"$c_{12}=\rho\,(c_L^2{-}2\,c_T^2)$", \
                 r"$c_{44}=\rho\,c_T^2$", \
                 r"$\nu=\frac{c_L^2-2\,c_T^2}{2\,(c_L^2-c_T^2)}$") )
    __units2 = ( (r"GPa",r"GPa",r"GPa","",r"mm/µs"),\
                 (r"GPa",r"GPa",r"GPa","",r"mm/µs"),\
                 (r"GPa",r"GPa",r"GPa","",r"mm/µs"),\
                 ("",r"GPa",r"GPa",r"GPa",r"mm/µs"),\
                 (r"mm/µs",r"GPa",r"GPa",r"GPa","") )
    #--------------------------------------------------------------------
    def __init__(self, parent, mainwin, complex_stiffnesses=False) :
        self.__mw = mainwin
        QFrame.__init__(self, parent)
        self.__cplx = complex_stiffnesses
        
        # First column vb1 :
        vb1 = QVBoxLayout()
        #vb1.addStretch()
        title = QLabel("Entry of the parameters:")
        vb1.addWidget(title)
            # Parameters to enter
        self.__choice = QComboBox(self)
        self.__choice.setFixedSize(220,22)
        titles = IsotropicFrame.__titles
        for tt in titles :
            self.__choice.addItem(tt[0])
        self.__choice.currentIndexChanged.connect(self.__changeEntry)
        choice = self.__choice.currentIndex()
        self.__previousIndex = choice
        vb1.addWidget(self.__choice)
            # editing frame 
        title = IsotropicFrame.__titles[choice][0]
        name1 = self.names1[choice][0]
        unit1 = IsotropicFrame.__units1[choice][0]
        name2 = self.names2[choice][0]
        unit2 = IsotropicFrame.__units2[choice][0]
        self.__frm0 = Frame(self, title, name1, unit1, name2, unit2, \
                            readOnly=False, height=180, lbl_w=60)
        vb1.addWidget(self.__frm0)
        vb1.addStretch()

        # Second column vb2 :
        vb2 = QVBoxLayout()
        title = IsotropicFrame.__titles[choice][1]
        name1 = self.names1[choice][1]
        unit1 = IsotropicFrame.__units1[choice][1]
        name2 = self.names2[choice][1]
        unit2 = IsotropicFrame.__units2[choice][1]
        self.__frm1 = Frame(self, title, name1, unit1, name2, unit2, \
                            readOnly=True)
        vb2.addWidget(self.__frm1)
        
        title = IsotropicFrame.__titles[choice][2]
        name1 = self.names1[choice][2]
        unit1 = IsotropicFrame.__units1[choice][2]
        name2 = self.names2[choice][2]
        unit2 = IsotropicFrame.__units2[choice][2]
        self.__frm2 = Frame(self, title, name1, unit1, name2, unit2, \
                            readOnly=True)
        vb2.addWidget(self.__frm2)
        vb2.addStretch()

        # Third column vb3 :
        vb3 = QVBoxLayout()
        title = IsotropicFrame.__titles[choice][3]
        name1 = self.names1[choice][3]
        unit1 = IsotropicFrame.__units1[choice][3]
        name2 = self.names2[choice][3]
        unit2 = IsotropicFrame.__units2[choice][3]
        self.__frm3 = Frame(self, title, name1, unit1, name2, unit2, \
                            readOnly=True)
        vb3.addWidget(self.__frm3)
        title = IsotropicFrame.__titles[choice][4]
        name1 = self.names1[choice][4]
        unit1 = IsotropicFrame.__units1[choice][4]
        name2 = self.names2[choice][4]
        unit2 = IsotropicFrame.__units2[choice][4]
        self.__frm4 = Frame(self, title, name1, unit1, name2, unit2, \
                            readOnly=True)
        vb3.addWidget(self.__frm4)
        vb3.addStretch()
        
        # Global layout       
        hb = QHBoxLayout()
        #hb.addStretch()
        hb.addLayout(vb1)
        #hb.addStretch()
        hb.addLayout(vb2)
        #hb.addStretch()
        hb.addLayout(vb3)
        hb.addStretch()

        self.setLayout(hb)
        
        self.__changeEntry()    
    #--------------------------------------------------------------------
    @property
    def names1(self) :
        if self.__cplx : return IsotropicFrame.__names1_complex
        else : return IsotropicFrame.__names1   
    #--------------------------------------------------------------------
    @property
    def names2(self) :
        if self.__cplx : return IsotropicFrame.__names2_complex
        else : return IsotropicFrame.__names2        
    #--------------------------------------------------------------------
    def __changeEntry(self) :
        choice = self.__choice.currentIndex()
        PRT(f"__changeEntry : current {choice}, " + \
            f"previous {self.__previousIndex}")
        if self.__choice.currentText() == \
              IsotropicFrame.__titles[-1][0] : # Velocities (real)
            if self.__cplx : # Complex values
                msg = "Do you really want to switch to real stiffnesses?"
                rep = self.__mw.approve(msg)
                if rep == 1 :
                    p1 = self.__frm0.frame1.value
                    p2 = self.__frm0.frame2.value
                    if isinstance(p1,complex) :
                        self.__frm0.frame1.value = p1.real
                    if isinstance(p2,complex) :
                        self.__frm0.frame2.value = p2.real
                    self.__mw.set_complex_on_off()
                else : # Undo
                    self.__choice.setCurrentIndex(self.__previousIndex)
                    return
        if choice != self.__previousIndex : 
            self.__mw.updateMaterial(False,True)
            self.__previousIndex = choice
        self.update_frame_labels()
        self.update(self.__mw.isomat)      
    #--------------------------------------------------------------------
    def update_frame_labels(self) :
        choice = self.__choice.currentIndex()
        titles = IsotropicFrame.__titles[choice]
        names1 = self.names1[choice]
        units1 = IsotropicFrame.__units1[choice]
        names2 = self.names2[choice]
        units2 = IsotropicFrame.__units2[choice]
        frms = (self.__frm0, self.__frm1, self.__frm2, self.__frm3, \
                self.__frm4)
        for frm,tit,nm1,un1,nm2,un2 in zip(frms, titles, names1, \
                                           units1, names2, units2) :
            frm.redraw(tit, nm1, un1, nm2, un2)        
    #--------------------------------------------------------------------
    def update_complex(self) :
        if self.__cplx == self.__mw.iscomplex : # No change
            return
        self.__cplx = self.__mw.iscomplex
        if self.__cplx : # Complex stiffnesses
            if self.__choice.currentText() == \
                     IsotropicFrame.__titles[-1][0] : # Velocities (real)
                self.__choice.setCurrentIndex(0) # Automatic Change
        else : # Real stiffnesses
            pass
        self.update_frame_labels()
    #--------------------------------------------------------------------
    def readAll(self) :
        PRT("+++ IsotropicFrame.readAll +++")
        p1 = self.__frm0.frame1.value
        p2 = self.__frm0.frame2.value
        choice = self.__previousIndex
        prm = dict()
        units = [1.0e9, 1.0e9] # Default values             
        if choice == 0 : # Lamé Coefficients (c12,c44)
            names = ("c12","c44")
        elif choice == 1 : # Stiffnesses c11 & c12
            names = ("c11","c12")
        elif choice == 2 : # Stiffnesses c11 & c44
            names = ("c11","c44")
        elif choice == 3 : # Young's Modulus & Poisson's Ratio
            names = ("young modulus","poisson ratio")
            units[1] = 1.0
        elif choice == 4 : # Velocities
            names = ("cL","cT")
            units = [1.0e3, 1.0e3]
        for (k,v,u) in zip(names,(p1,p2),units) :
            try :
                val = v*u
                prm[k] = val
            except :
                pass # prm[k] = None
        return prm
    #--------------------------------------------------------------------
    def update(self, material) :
        PRT("+++ IsotropicFrame.update +++")
        if isinstance(material,IESolid) :
            if material.iscomplex :
                if not self.__cplx :
                    self.__mw.set_complex_on_off()
            else :
                if self.__cplx :
                    self.__mw.set_complex_on_off()
            choice = self.__choice.currentIndex()
            # Frames
            C01 = self.__frm0.frame1
            C02 = self.__frm0.frame2
            C11 = self.__frm1.frame1
            C12 = self.__frm1.frame2
            C21 = self.__frm2.frame1
            C22 = self.__frm2.frame2
            C31 = self.__frm3.frame1
            C32 = self.__frm3.frame2
            C41 = self.__frm4.frame1
            C42 = self.__frm4.frame2
            # Parameters
            if material.c11 is not None : c11 = material.c11*1e-9
            else : c11 = None
            if material.c12 is not None : c12 = material.c12*1e-9
            else : c12 = None
            if material.c44 is not None : c44 = material.c44*1e-9
            else : c44 = None
            if material.E is not None : E = material.E*1e-9
            else : E = None
            nu = material.nu
            if material.real_cL is not None :
                cL = material.real_cL*1e-3
            else : cL = None
            if material.real_cT is not None :
                cT = material.real_cT*1e-3
            else : cT = None
            if choice == 0 : # Lamé Coefficients (c12,c44)
                C01.value = c12
                C02.value = c44
                C11.value = c11
                C12.value = c12
                C21.value = c11
                C22.value = c44
                C31.value = E
                C32.value = nu
                C41.value = cL
                C42.value = cT
            elif choice == 1 : # Stiffnesses c11 & c12
                C01.value = c11
                C02.value = c12
                C11.value = c12
                C12.value = c44
                C21.value = c11
                C22.value = c44
                C31.value = E
                C32.value = nu
                C41.value = cL
                C42.value = cT
            elif choice == 2 : # Stiffnesses c11 & c44
                C01.value = c11
                C02.value = c44
                C11.value = c12
                C12.value = c44
                C21.value = c11
                C22.value = c12
                C31.value = E
                C32.value = nu
                C41.value = cL
                C42.value = cT
            elif choice == 3 : # Young's Modulus & Poisson's Ratio 
                C01.value = E
                C02.value = nu
                C11.value = c12
                C12.value = c44
                C21.value = c11
                C22.value = c12
                C31.value = c11
                C32.value = c44
                C41.value = cL
                C42.value = cT
            elif choice == 4 : # Velocities  
                C01.value = cL
                C02.value = cT
                C11.value = c12
                C12.value = c44
                C21.value = c11
                C22.value = c12
                C31.value = c11
                C32.value = c44
                C41.value = E
                C42.value = nu
        else : # Undefined material
            PRT("Undefined isotropic material...")
    #--------------------------------------------------------------------
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "IsotropicFrame" : 
            if name in IsotropicFrame.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================
class Frame(QFrame) :
    """ Frame with one title and two areas of type 'Entry' or 'Display'.
    """
    #--------------------------------------------------------------------
    def  __init__(self, parent, title, name1, unit1, name2, unit2, \
                  readOnly=True, height=165, lbl_w=200, edi_w=110, \
                  uni_w=60) :
        dim = ( 20 + lbl_w + edi_w + uni_w, height)
        QFrame.__init__(self,parent)
        self.__RO = readOnly
        self.setFrameStyle(QFrame.Box|QFrame.Plain)
        self.setStyleSheet("background:transparent")
        self.setLineWidth(2)
        if readOnly :
            self.__title = QLabel("  "+title)
            self.__frm1 = Display(self, name1 ,unit1 ,fmt=".3f", \
                                 labelW=lbl_w, editW=edi_w, unitW=uni_w)
            self.__frm2 = Display(self,name2,unit2,fmt=".3f", \
                                 labelW=lbl_w, editW=edi_w, unitW=uni_w)
        else :
            self.__title = QLabel("  Enter "+title+":")
            self.__frm1 = Entry(self,name1,unit1,fmt=".3f", \
                                 labelW=lbl_w, editW=edi_w, unitW=uni_w)
            self.__frm2 = Entry(self,name2,unit2,fmt=".3f", \
                                 labelW=lbl_w, editW=edi_w, unitW=uni_w)
        self.__title.setFixedSize(220,30)
        self.__VBL = QVBoxLayout()
        self.__VBL.setSpacing(1)
        try : # PyQt5
            self.__VBL.setContentsMargins(1,1,1,1)
        except : # PyQt4
            self.__VBL.setMargin(1)
        self.__VBL.addWidget(self.__title)
        self.__VBL.addWidget(self.__frm1)
        self.__VBL.addWidget(self.__frm2)
        self.setLayout(self.__VBL)
        self.setFixedSize(*dim)
    #--------------------------------------------------------------------        
    @property
    def frame1(self) : return self.__frm1
    #--------------------------------------------------------------------
    @property
    def frame2(self) : return self.__frm2
    #--------------------------------------------------------------------
    def redraw(self,title,name1,unit1,name2,unit2) :
        readOnly = self.__RO
        if readOnly :
            newtitle = "  "+title
        else :
            newtitle = "  Enter "+title+":"
        if newtitle != self.__title.text() :
            self.__title.setText(newtitle)
        if self.frame1.name != name1 :
            self.frame1.name = name1
        if self.frame1.unit != unit1 :
            self.frame1.unit = unit1
        if self.frame2.name != name2 :
            self.frame2.name = name2
        if self.frame2.unit != unit2 :
            self.frame2.unit = unit2            
#========================================================================
#========================================================================   
class Entry(QFrame) :
    """QFrame with a name, an entry window, a unit."""
    editingFinished = pyqtSignal(int)
    #--------------------------------------------------------------------
    def __init__(self, parent, name, unit, fmt=".3f", labelW=80, \
                 editW=60, unitW=40) :
        """ 'name' is the parameter name to enter, 'unit' its unit,
            and 'fmt' the format of the value to enter."""
        QFrame.__init__(self,parent)
        self.__name = name
        self.__unit = unit
        self.__str = "{:"+fmt+"}"
        self.__integer_values = fmt.endswith("d")
        if self.__integer_values : # Integer values
            try :
                self.__str.format(0)
            except :
                PRT(f"Incorrect format '{self.__str}'")
                self.__str = "{:d}"
        else : # Float or Complex
            try :
                self.__str.format(0.0)
            except :
                PRT(f"Incorrect format '{self.__str}'")
                self.__str = "{:.3e}"
        HL = QHBoxLayout()
        HL.setSpacing(1)
        try : # PyQt5
            HL.setContentsMargins(1,1,1,1)
        except : # PyQt4
            HL.setMargin(1)
        self.__edit = QLineEdit()
        self.__value = None
        # Equation with matplotlib
        HL.addStretch()
        self.__fig1 = Figure(facecolor = "none")
        self.__label1 = FigureCanvas(self.__fig1)
        self.__label1.setParent(parent)
        self.__label1.setFixedSize(labelW,30)
        self.__updateName()
        HL.addWidget(self.__label1)
        # Entry area
        self.__edit.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        self.__edit.setFont(QFont("Times New Roman",10, QFont.Bold))
        self.__edit.setStyleSheet("color: rgb(0, 0, 200);")
        self.__edit.setFixedSize(editW,20)
        HL.addWidget(self.__edit)
        self.__edit.editingFinished.connect(self.__changeValue)
        # Unit
        self.__label2 = QLabel(self)
        self.__label2.setFont( QFont("Times New Roman",USZ) )
        self.__label2.setFixedSize(unitW,30)
        self.__label2.setAlignment(Qt.AlignLeft|Qt.AlignVCenter)
        self.__updateUnit()
        HL.addWidget(self.__label2)
        HL.addStretch()
        self.setLayout(HL)
        self.adjustSize()
    #--------------------------------------------------------------------
    def setEnabled(self, true_false) :
        if true_false :
            self.__edit.setStyleSheet("color: rgb(0, 0, 200);")
        else :
            self.__edit.setStyleSheet("color: rgb(0, 0, 0); " + \
                            "background-color: rgb(230, 230, 230);")
        QFrame.setEnabled(self, true_false)
    #--------------------------------------------------------------------    
    def __updateName(self) :
        self.__fig1.clear()
        self.__fig1.suptitle(self.__name,fontsize=LSZ,\
                        x = 1.0, y = 0.5,\
                        horizontalalignment="right",\
                        verticalalignment="center")
        self.__label1.draw()
    #--------------------------------------------------------------------
    def __updateUnit(self) : 
        self.__label2.setText(self.__unit)
    #--------------------------------------------------------------------   
    @property
    def name(self) : return self.__name
    @name.setter
    def name(self,newname) :
        try :
            self.__name = newname
            self.__updateName()
        except :
            PRT("Entry :: error in the name change...")
    #--------------------------------------------------------------------
    @property
    def unit(self) : return self.__unit
    @unit.setter
    def unit(self,newunit) :
        try :
            self.__unit = newunit
            self.__updateUnit()
        except :
            PRT("Entry :: error in the unit change...")
    #--------------------------------------------------------------------
    @property
    def value(self) : return self.__value
    @value.setter
    def value(self,text) :
        # Integer values
        if self.__integer_values :
            try :
                value = int(text)
            except :
                if text != "" and text is not None:
                    PRT("Incorrect entered value '{text}'")
                self.__value = None
                self.__edit.clear()
                return
        # Float or Complex
        else :
            try :
                value = complex(text)
                if abs(value.imag) <= ZERONUM*abs(value.real) :
                    value = value.real
            except :
                if text != "" and text is not None:
                    PRT("Incorrect entered value '{text}'")
                self.__value = None
                self.__edit.clear()
                return
        self.__value = value
        self.__edit.setText(self.__str.format(self.__value))
    #--------------------------------------------------------------------    
    def __changeValue(self) :
         textelu = self.__edit.text()
         PRT("texte lu :",textelu)
         self.value = textelu
         self.editingFinished.emit(1)
#========================================================================
#========================================================================
class Display(QFrame) :
    """QFrame with an equation, a displayed value, a unit."""
    def __init__(self, parent, name, unit, value="", fmt=".3f", \
                 size=None, labelW=200, labelH=60, editW=60, unitW=60) :
    #--------------------------------------------------------------------
        """ 'name' is the name of the parameter, 'unit' its unit,
            and 'fmt' its format """
        if size is None : size = (labelW,labelH)
        QFrame.__init__(self,parent)
        self.__name = name 
        self.__value = value
        self.__str = "{:"+fmt+"}"
        self.__unit = unit
        try :
            self.__str.format(0.0)
        except :
            PRT(f"Incorrect format '{self.__str}'")
            self.__str = "{:.3e}"
        HL = QHBoxLayout()
        HL.setSpacing(0)
        try : # PyQt5
            HL.setContentsMargins(1,1,1,1)
        except : # PyQt4
            HL.setMargin(1)
        # Equation with matplotlib
        self.__fig1 = Figure(facecolor = "none")
        self.__label1 = FigureCanvas(self.__fig1)
        self.__label1.setParent(parent)
        self.__label1.setFixedSize(*size)
        self.__updateName()
        HL.addWidget(self.__label1)
        # Display window
        self.__edit = QLineEdit()
        self.__edit.setReadOnly(True)
        self.__edit.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        self.__edit.setFont(QFont("Times New Roman",12, QFont.Bold))
        self.__edit.setStyleSheet("color: rgb(0, 0, 0); " + \
                            "background-color: rgb(230, 230, 230);")
        self.__edit.setFixedSize(editW,20)
        HL.addWidget(self.__edit)
        # Equation with matplotlib
        self.__fig2 = Figure(facecolor = "none")
        self.__label2 = FigureCanvas(self.__fig2)
        self.__label2.setParent(parent)
        self.__label2.setFixedSize(unitW,25)
        self.__updateUnit()
        HL.addWidget(self.__label2)
        self.setLayout(HL)
        self.adjustSize()
    #--------------------------------------------------------------------        
    def __updateName(self) :
        self.__fig1.clear()
        self.__fig1.suptitle(self.__name,fontsize=LSZ,\
                        x = 1.0, y = 0.5,\
                        horizontalalignment="right",\
                        verticalalignment="center")
        self.__label1.draw()
    #--------------------------------------------------------------------
    def __updateUnit(self) : 
        self.__fig2.clear()
        self.__fig2.suptitle(self.__unit,fontsize=USZ,\
                        x = 0.0, y = 0.5,\
                        horizontalalignment="left",\
                        verticalalignment="center")
        self.__label2.draw()
    #--------------------------------------------------------------------
    @property
    def name(self) : return self.__name
    @name.setter
    def name(self,newfm) :
        try :
            self.__name = newfm
            self.__updateName()
        except :
            PRT("Display :: error in the equation change...")
    #--------------------------------------------------------------------            
    @property
    def unit(self) : return self.__unit
    @unit.setter
    def unit(self,newunit) :
        try :
            self.__unit = newunit
            self.__updateUnit()
        except :
            PRT("Display :: error in the unit change...")
    #--------------------------------------------------------------------  
    @property
    def value(self) : return self.__value
    @value.setter
    def value(self,val) :
        if isinstance(val,float) or isinstance(val,complex) :
            self.__value = val
            self.__edit.setText(self.__str.format(val))
        elif val == None :
            self.__value = None
            self.__edit.clear()
        else :
            PRT("Display.value : Valeur '{}' de type {}".format(\
                val,type(val).__name__))
#========================================================================
#========================================================================
class TransIsoFrame(QFrame) :
    __attrNames = tuple("_TransIsoFrame__"+n for n in\
                          ("stiff13", "stiff46", "choice", "prec", \
                           "velPV", "velSV", "velPH", "velSH", "cplx", \
                           "mw") )
    __titles = ("c66 = (c11-c12)/2", "c12 = c11 - 2*c66", \
                "c11 = c12 + 2*c66")
    #--------------------------------------------------------------------
    def __init__(self, parent, complex_stiffnesses=False) :        
        QFrame.__init__(self, parent)
        TwoByTwo = QGridLayout()
        self.__cplx = complex_stiffnesses
        self.__mw = parent.parent()
        self.__stiff13 = [[StiffnessFrame(self,"C{}{}".format(i,j))\
                           for j in range(i,4)] for i in [1,2,3]]
        self.__stiff13[1][0].setLabel(r"$c_{22}=c_{11}$")
        self.__stiff13[1][0].setReadOnly()
        self.__stiff13[1][1].setLabel(r"$c_{23}=c_{13}$")
        self.__stiff13[1][1].setReadOnly() 
        bx13 = QGridLayout()
        for i,row in enumerate(self.__stiff13,1) :         
            for j,f in enumerate(row,i) :
                bx13.addWidget(f,i,j)
        TwoByTwo.addLayout(bx13,1,1)
        self.__stiff46 = [ StiffnessFrame(self,"C{}{}".format(i,i))\
                           for i in [4,5,6] ]
        self.__stiff46[1].setLabel(r"$c_{55}=c_{44}$")
        self.__stiff46[1].setReadOnly()
        bx46 = QGridLayout()
        for i,f in enumerate(self.__stiff46,1) :
                bx46.addWidget(f,i,i)
        TwoByTwo.addLayout(bx46,2,2)
        # Values to enter
        self.__choice = QComboBox(self)
        self.__choice.setFont( \
            QFont("Arial",12, QFont.Bold) )
        self.__choice.setStyleSheet("color: rgb(0, 0, 0);")
        self.__choice.setFixedSize(180,28)
        titles = TransIsoFrame.__titles
        for tt in titles :
            self.__choice.addItem(tt)
        self.__choice.setCurrentIndex(1)
        self.__choice.currentIndexChanged.connect(self.__changeEntry)
        self.__prec = None
        hb12 = QHBoxLayout()
        vb12 = QVBoxLayout()
        vb12.addStretch()
        vb12.addWidget(self.__choice)
        vb12.addStretch()
        hb12.addLayout(vb12)
        hb12.addStretch()
        TwoByTwo.addLayout(hb12,1,2)
        # Display of speeds
        cadre = QFrame(self)
        cadre.setFrameStyle(QFrame.Box|QFrame.Plain)
        cadre.setStyleSheet("background:transparent")
        cadre.setLineWidth(2)
        ly = QVBoxLayout()
        self.__velPV = Display(self,r"$c_{PV}$",r" mm/µs",\
                              size = (40,25))
        ly.addWidget(self.__velPV)
        self.__velSV = Display(self,r"$c_{SV}$",r" mm/µs",\
                              size = (40,25))
        ly.addWidget(self.__velSV)
        self.__velPH = Display(self,r"$c_{PH}$",r" mm/µs",\
                              size = (40,25))
        ly.addWidget(self.__velPH)
        self.__velSH = Display(self,r"$c_{SH}$",r" mm/µs",\
                              size = (40,25))
        ly.addWidget(self.__velSH)
        cadre.setLayout(ly)
        velText = QLabel("Wave Speeds:")
        velText.setFont(QFont("Times New Roman",12,QFont.Bold))
        hb21 = QHBoxLayout()
        vb21 = QVBoxLayout()
        vb21.addStretch()
        vb21.addWidget(velText)
        vb21.addWidget(cadre)
        hb21.addLayout(vb21)
        hb21.addStretch()
        TwoByTwo.addLayout(hb21,2,1)
        
        # Stretches
        vb = QVBoxLayout()
        hb = QHBoxLayout()
        hb.addLayout(TwoByTwo)
        hb.addStretch()
        vb.addLayout(hb)
        vb.addStretch()
        self.setLayout(vb)

        # Update
        self.__changeEntry()
    #--------------------------------------------------------------------
    def update_complex(self) :
        if self.__cplx == self.__mw.iscomplex : # No change
            return
        self.__cplx = self.__mw.iscomplex
    #--------------------------------------------------------------------
    def readStiffnesses(self) :
        prm = dict()
        #+++++++++++
        def transfer(str_cij, edit_cij, prm=prm, mw=self.__mw) :
            try :
                cij = complex(edit_cij.text)
                if abs(cij.imag) < ZERONUM*abs(cij.real) :
                    cij = cij.real                    
                    prm[str_cij] = 1e9*cij
                elif not mw.iscomplex :
                    msg = "Do you really want to switch to " + \
                          "complex stiffnesses?"
                    rep = mw.approve(msg)
                    if rep == 1 : # Yes
                        mw.set_complex_on_off()                   
                        prm[str_cij] = 1e9*cij
                    else :
                        pass
                else :                   
                    prm[str_cij] = 1e9*cij
            except :
                pass
            return
        #+++++++++++
        for sij,stij in ( ("c13",self.__stiff13[0][2]), \
                          ("c33",self.__stiff13[2][0]), \
                          ("c44",self.__stiff46[0]) ) :
            transfer(sij, stij)
        #+++++++++++
        choice = self.__choice.currentIndex()
        if choice == 0 : 
            for sij,stij in ( ("c11",self.__stiff13[0][0]), \
                              ("c12",self.__stiff13[0][1]) ) :
                transfer(sij, stij) 
        elif choice == 1 :
            for sij,stij in ( ("c11",self.__stiff13[0][0]), \
                              ("c66",self.__stiff46[2] ) ) :
                transfer(sij, stij) 
        elif choice == 2 : 
            for sij,stij in ( ("c12",self.__stiff13[0][1]), \
                              ("c66",self.__stiff46[2] ) ) :
                transfer(sij, stij)
        return prm
    #--------------------------------------------------------------------
    def update(self, material) :
        PRT("+++ TransIsoFrame.update +++")
        if isinstance(material, TIESolid) :
            # Stiffnesses
            if material.iscomplex : # complex stiffnesses
                if not self.__cplx :
                    self.__mw.set_complex_on_off()
                if isinstance(material.c11,float) or \
                   isinstance(material.c11,complex) :
                    c11 = material.c11*1e-9
                    self.__stiff13[0][0].setText(f"{c11:.2f}")
                    self.__stiff13[1][0].setText(f"{c11:.2f}")
                else :
                    self.__stiff13[0][0].setText("")
                    self.__stiff13[1][0].setText("")
                if isinstance(material.c12,float) or \
                   isinstance(material.c12,complex) :
                    c12 = material.c12*1e-9
                    self.__stiff13[0][1].setText(f"{c12:.2f}")
                else :
                    self.__stiff13[0][1].setText("")
                if isinstance(material.c13,float) or \
                   isinstance(material.c13,complex) :
                    c13 = material.c13*1e-9
                    self.__stiff13[0][2].setText(f"{c13:.2f}")
                    self.__stiff13[1][1].setText(f"{c13:.2f}")
                else :
                    self.__stiff13[0][2].setText("")
                    self.__stiff13[1][1].setText("")
                if isinstance(material.c33,float) or \
                   isinstance(material.c33,complex) :
                    c33 = material.c33*1e-9
                    self.__stiff13[2][0].setText(f"{c33:.2f}")
                else :
                    self.__stiff13[2][0].setText("")
                if isinstance(material.c44,float) or \
                   isinstance(material.c44,complex) :
                    c44 = material.c44*1e-9
                    self.__stiff46[0].setText(f"{c44:.2f}")
                    self.__stiff46[1].setText(f"{c44:.2f}")
                else :
                    self.__stiff46[0].setText("")
                    self.__stiff46[1].setText("")
                if isinstance(material.c66,float) or \
                   isinstance(material.c66,complex) :
                    c66 = material.c66*1e-9
                    self.__stiff46[2].setText(f"{c66:.2f}")
                else :
                    self.__stiff46[2].setText("")
            else : # Real stiffnesses
                if self.__cplx :
                    self.__mw.set_complex_on_off()
                if isinstance(material.c11,float) :
                    c11 = material.c11*1e-9
                    self.__stiff13[0][0].setText(f"{c11:.2f}")
                    self.__stiff13[1][0].setText(f"{c11:.2f}")
                else :
                    self.__stiff13[0][0].setText("")
                    self.__stiff13[1][0].setText("")
                if isinstance(material.c12,float) :
                    c12 = material.c12*1e-9
                    self.__stiff13[0][1].setText(f"{c12:.2f}")
                else :
                    self.__stiff13[0][1].setText("")
                if isinstance(material.c13,float) :
                    c13 = material.c13*1e-9
                    self.__stiff13[0][2].setText(f"{c13:.2f}")
                    self.__stiff13[1][1].setText(f"{c13:.2f}")
                else :
                    self.__stiff13[0][2].setText("")
                    self.__stiff13[1][1].setText("")
                if isinstance(material.c33,float) :
                    c33 = material.c33*1e-9
                    self.__stiff13[2][0].setText(f"{c33:.2f}")
                else :
                    self.__stiff13[2][0].setText("")
                if isinstance(material.c44,float) :
                    c44 = material.c44*1e-9
                    self.__stiff46[0].setText(f"{c44:.2f}")
                    self.__stiff46[1].setText(f"{c44:.2f}")
                else :
                    self.__stiff46[0].setText("")
                    self.__stiff46[1].setText("")
                if isinstance(material.c66,float) :
                    c66 = material.c66*1e-9
                    self.__stiff46[2].setText(f"{c66:.2f}")
                else :
                    self.__stiff46[2].setText("")
            # Speeds
            if isinstance(material.cPV,float) :
                cPV = material.cPV*1e-3
                self.__velPV.value = cPV
            else :
                self.__velPV.value = ""
            if isinstance(material.cSV,float) :
                cSV = material.cSV*1e-3
                self.__velSV.value = cSV
            else :
                self.__velSV.value = ""
            if isinstance(material.cPH,float) :
                cPH = material.cPH*1e-3
                self.__velPH.value = cPH
            else :
                self.__velPH.value = ""
            if isinstance(material.cSH,float) :
                cSH = material.cSH*1e-3
                self.__velSH.value = cSH
            else :
                self.__velSH.value = ""
        else :
            PRT("Undefined transversely isotropic material...")
    #--------------------------------------------------------------------
    def __changeEntry(self) :
        choice = self.__choice.currentIndex()
        if choice != self.__prec :
            self.__prec = choice
            if choice == 0 : # c66 = (c11-c12)/2                 
                self.__stiff13[0][0].setLabel(r"$c_{11}$")
                self.__stiff13[0][0].setReadOnly(False)                
                self.__stiff13[0][1].setLabel(r"$c_{12}$")
                self.__stiff13[0][1].setReadOnly(False)              
                self.__stiff46[2].setLabel(\
                    r"$c_{66}=\frac{c_{11}-c_{12}}{2}$",10)
                self.__stiff46[2].setReadOnly()
            elif choice == 1 : # c12 = c11 - 2*c66                
                self.__stiff13[0][0].setLabel(r"$c_{11}$")
                self.__stiff13[0][0].setReadOnly(False)                
                self.__stiff13[0][1].setLabel(\
                    r"$c_{12}{=}c_{11}{-}2c_{66}$",10)
                self.__stiff13[0][1].setReadOnly()              
                self.__stiff46[2].setLabel(r"$c_{66}$")
                self.__stiff46[2].setReadOnly(False)
            else : # c11 = c12 + 2*c66                
                self.__stiff13[0][0].setLabel(\
                    r"$c_{11}{=}c_{12}{+}2c_{66}$",10)
                self.__stiff13[0][0].setReadOnly()                
                self.__stiff13[0][1].setLabel(r"$c_{12}$")
                self.__stiff13[0][1].setReadOnly(False)              
                self.__stiff46[2].setLabel(r"$c_{66}$")
                self.__stiff46[2].setReadOnly(False)
    #--------------------------------------------------------------------
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "TransIsoFrame" : 
            if name in TransIsoFrame.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================       
class FluidFrame(QFrame) :
    __attrNames = tuple("_FluidFrame__"+n for n in \
                          ("sel", "spdFrame", "attFrame", "bmFrame", \
                           "cplx", "mw", "unit") )
    FONT = QFont("Arial", 10, QFont.Bold)
    #--------------------------------------------------------------------
    def __init__(self, parent, complex_bulk_modulus=False) :        
        QFrame.__init__(self, parent)
        self.__cplx = complex_bulk_modulus
        self.__mw = parent.parent()
        self.__unit = "GPa"
        # Combo Box
        self.__sel = QComboBox(self)
        self.__sel.setFont(self.FONT)
        self.__sel.clear()
        self.__sel.addItems(["Sound Speed and Attenuation (reals)", \
                             "Bulk Modulus (real or complex)"])
        self.__sel.currentIndexChanged.connect(self.change_input)
        # Frame 1
        frm1 = QFrame(self)
        frm1.setFrameStyle(QFrame.Box|QFrame.Plain)
        frm1.setStyleSheet("background:transparent")
        frm1.setLineWidth(2)
        lay1 = QVBoxLayout()
        self.__spdFrame = Entry(self, "Sound speed $c$ ", "mm/µs", \
                                fmt=".4f", editW=80, labelW=155, \
                                unitW=80)
        self.__attFrame = Entry(self, "Attenuation ", "%", \
                                fmt=".2f", editW=80, labelW=155, \
                                unitW=80)
        lay1.addWidget(self.__spdFrame)
        lay1.addWidget(self.__attFrame)
        frm1.setLayout(lay1)
        # Frame 2
        frm2 = QFrame(self)
        frm2.setFrameStyle(QFrame.Box|QFrame.Plain)
        frm2.setStyleSheet("background:transparent")
        frm2.setLineWidth(2)
        lay2 = QVBoxLayout()
        self.__bmFrame = Entry(self, "Bulk Modulus ", " GPa", \
                               fmt=".4f", editW=120, labelW=155, \
                               unitW=80)
        lay2.addWidget(self.__bmFrame)
        frm2.setLayout(lay2)
        # Global Layout
        hb = QHBoxLayout()
        hb.addStretch()
        vb = QVBoxLayout()
        vb.addStretch()        
        vb.addWidget(self.__sel)
        vb.addWidget(frm1)
        vb.addWidget(frm2)
        vb.addStretch()
        hb.addLayout(vb)
        hb.addStretch()
        self.setLayout(hb)
        
        self.change_input() # Initialization
    #--------------------------------------------------------------------
    def update_complex(self) :
        if self.__cplx == self.__mw.iscomplex : # No change
            return
        self.__cplx = self.__mw.iscomplex
        self.change_input() # Updates input frames
    #--------------------------------------------------------------------
    def change_input(self) :
        PRT("FluidFrame.change_input")
        if self.__sel.currentIndex() == 0 : # Speed and attenuation
            self.__spdFrame.setEnabled(True)
            if self.__cplx : self.__attFrame.setEnabled(True)
            else :
                self.__attFrame.setEnabled(False)
                self.__attFrame.value = ""
            self.__bmFrame.setEnabled(False)
        else : # Bulk Modulus
            self.__spdFrame.setEnabled(False)
            self.__attFrame.setEnabled(False)
            self.__bmFrame.setEnabled(True)
    #--------------------------------------------------------------------
    def readParameters(self) :
        """returns a dictionary of keys 'c' (sound speed in m/s),
           'a' (attenuation) and 'K' (bulk modulus in Pa)."""
        dico_data = dict()
        if self.__sel.currentIndex() == 0 : # Speed and attenuation
            try :
                c = float(self.__spdFrame.value)
                dico_data["c"] = 1e3*c
            except :
                pass
            try :
                a = float(self.__attFrame.value)
                dico_data["a"] = 0.01*a
            except :
                pass
        else : # Bulk Modulus
            new_complex = False
            try :
                K = float(self.__bmFrame.value)
                if self.__unit == "GPa" :
                    dico_data["K"] = 1e9*K
                else : # "MPa"
                    dico_data["K"] = 1e6*K
            except :
                try :
                    K = complex(self.__bmFrame.value)
                    if self.__unit == "GPa" :
                        dico_data["K"] = 1e9*K
                    else : # "MPa"
                        dico_data["K"] = 1e6*K
                    new_complex = True
                except :
                    pass
            if new_complex and not self.__cplx :
                msg = "Do you really want to switch to " + \
                      "complex bulk modulus?"
                rep = self.__mw.approve(msg)
                if rep == 1 : # Yes
                    self.__mw.set_complex_on_off()
                else : # -1 [No] or 0 [Cancel]
                    dico_data = dict()
        return dico_data
    #--------------------------------------------------------------------
    def update(self, material) :
        PRT("+++ FluidFrame.update +++")
        if isinstance(material,Fluid) :
            if material.iscomplex : # Complex Bulk Modulus
                if not self.__cplx :
                    self.__mw.set_complex_on_off()
                if isinstance(material.c, float) : # No attenuation!
                    c = material.c*1e-3
                    self.__spdFrame.value = c 
                    self.__attFrame.value = 0.0           
                elif isinstance(material.c, complex) : # Attenuation
                    c = material.c.real*1e-3
                    self.__spdFrame.value = c
                    c2 = material.c**2
                    a = 100*(c2.imag/c2.real)
                    self.__attFrame.value = a
                else :
                    self.__spdFrame.value = ""
                if isinstance(material.K, float) : # No attenuation!
                    if self.__unit == "GPa" :
                        K = material.K*1e-9
                        if K < 0.01 :
                            K *= 1e3
                            self.__unit = "MPa"
                            self.__bmFrame.unit = " MPa"
                    else : # self.__unit == "MPa"
                        K = material.K*1e-6
                        if K > 10.0 :
                            K *= 1e-3
                            self.__unit = "GPa"
                            self.__bmFrame.unit = " GPa"                            
                    self.__bmFrame.value = K
                    self.__attFrame.value = 0.0
                elif isinstance(material.K, complex) : # Attenuation
                    if self.__unit == "GPa" :
                        K = material.K*1e-9
                        if K.real < 0.01 :
                            K *= 1e3
                            self.__unit = "MPa"
                            self.__bmFrame.unit = " MPa"
                    else : # self.__unit == "MPa"
                        K = material.K*1e-6
                        if K.real > 10.0 :
                            K *= 1e-3
                            self.__unit = "GPa"
                            self.__bmFrame.unit = " GPa"
                    self.__bmFrame.value = K
                    a = 100*(K.imag/K.real)
                    self.__attFrame.value = a
            else :  # Real Bulk Modulus
                self.__attFrame.value = "" 
                if self.__cplx :
                    self.__mw.set_complex_on_off()
                if isinstance(material.c, float) :
                    c = material.c*1e-3
                    self.__spdFrame.value = c 
                else :
                    self.__spdFrame.value = ""
                if isinstance(material.K, float) :
                    if self.__unit == "GPa" :
                        K = material.K*1e-9
                        if K < 0.01 :
                            K *= 1e3
                            self.__unit = "MPa"
                            self.__bmFrame.unit = " MPa"
                    else : # self.__unit == "MPa"
                        K = material.K*1e-6
                        if K > 10.0 :
                            K *= 1e-3
                            self.__unit = "GPa"
                            self.__bmFrame.unit = " GPa"                            
                    self.__bmFrame.value = K
                else :
                    self.__bmFrame.value = ""
        else : 
            PRT("undefined fluid material...")     
    #--------------------------------------------------------------------
    def __setattr__(self,name,value) :
        cls = str(self.__class__)[8:-2].split(".")[-1]
        if cls == "FluidFrame" : 
            if name in FluidFrame.__attrNames :
                object.__setattr__(self,name,value)
            else :
                msg = "{} : unauthorized creation of attribut '{}'"
                PRT(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#========================================================================
#========================================================================
if __name__ == '__main__':
    def main():
        import sys  
        app = QApplication.instance() 
        if not app :
            app = QApplication(sys.argv)            
        win = Material_App()
        win.show()
        sys.exit(app.exec_())        
    main()
