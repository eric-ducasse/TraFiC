# Version 1.65 - 2025, July 13
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from TraFiC.Basic_Tools.TraFiC_utilities import now 
with open("./TraFiC_init_template.txt","r",encoding="utf-8") as f :
    incomplete_code = f.read()
complete_code = incomplete_code.format(now(True))
with open("./TraFiC/TraFiC_init.py","w",encoding="utf-8") as f :
    f.write(complete_code)
from TraFiC.TraFiC_init import root
print( "TraFiC_installer :: TraFiC_init.py created\n"
      + f"in root: '{root}'.")
