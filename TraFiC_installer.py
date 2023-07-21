# Version 1.64 - 2023, July 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
from TraFiC.Basic_Tools.TraFiC_utilities import now 
import os
root = os.path.dirname(__file__).replace("\\","/")+"/TraFiC"
print(f"TraFiC_installer :: root : '{root}'")
with open("./TraFiC_init_template.txt","r",encoding="utf-8") as f :
    incomplete_code = f.read()
complete_code = incomplete_code.format(root, now(True))
with open("./TraFiC/TraFiC_init.py","w",encoding="utf-8") as f :
    f.write(complete_code)
print("TraFiC_installer :: TraFiC_init.py created")
