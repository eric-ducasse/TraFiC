# Version 0.81 - 2021, March 15
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import TraFiC_init
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
from PyQt5.QtWidgets import QApplication
from TraFiC_App import TraFiC_Application
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
app    = QApplication(sys.argv)
my_app = TraFiC_Application()
app.exec_() 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

