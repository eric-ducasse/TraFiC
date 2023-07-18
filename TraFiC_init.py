# Version 1.63 - 2023, July 18
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import sys
root = "H:/Recherche/TraFiC"
dirs = [root]
dirs.append( root+"/MaterialClasses" )
dirs.append( root+"/StructureClasses" )
dirs.append( root+"/GUI_classes" )
dirs.append( root+"/Modes_Computation" )
dirs.append( root+"/Grids" )
dirs.append( root+"/Basic_Tools" )
sys.path.extend(dirs)
# Installed: 2023-07-18 08:46:02
