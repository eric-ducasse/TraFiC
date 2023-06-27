# Version 1.61 - 2022, August 25
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Configuration file for CURTA
import sys, platform
computer_name = platform.node()
if computer_name == "I2M198081" : # nouveau portable É.D.
    if platform.system() == "Windows" :
        root = "H:"
    elif platform.system() == "Linux" :
        root = "/home/duke/disque_H"
    else :
        raise ValueError(f"Système '{platform.system()}' inconnu")
    root += "/Recherche/TraFiC"
elif computer_name == "ampt-bx-ported" : # ancien portable É.D. 
    root = "H:/Recherche/TraFiC"
elif computer_name == "PCKRISHNA" : # fixe É.D.
    root = "D:/Eric_Ducasse/TraFiC"
elif computer_name.startswith("login") : # Curta
    root = "/gpfs/home/deric/TraFiC_seul"
else : # srvcoffa ou srv-ebsatt
    root = "/home/educasse/TraFiC_20_06_26"
print(f"TraFiC_init :: root: '{root}'\n      " + \
      f"computer_name: '{computer_name}'")
dirs = [root]
dirs.append( root+"/GUI_classes" )
dirs.append( root+"/Sources" )
dirs.append( root+"/Grids" )
dirs.append( root+"/Defect" )
dirs.append( root+"/Modes_Computation" )
dirs.append( root+"/Basic_Tools" )
sys.path.extend(dirs)
