# Version 1.0 - 2021, February 25
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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
from MaterialClasses import *
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Source_Frame(QWidget):
    #--------------------------------------------------------------------
    def __init__(self, mainWindow):
        QWidget.__init__(self, mainWindow)
        self.mw = mainWindow
        # User Interface Initialization 
        self.__initUI()       
    #--------------------------------------------------------------------
    def __initUI(self):
        pass
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Field_Computation_Frame(QWidget):
    #--------------------------------------------------------------------
    def __init__(self, mainWindow):
        QWidget.__init__(self, mainWindow)
        self.mw = mainWindow
        # User Interface Initialization 
        self.__initUI()       
    #--------------------------------------------------------------------
    def __initUI(self):
        pass
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
