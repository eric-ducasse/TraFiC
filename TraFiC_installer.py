# Version 1.62 - 2023, June 23
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
root = os.getcwd().replace("\\","/")
with open("./TraFiC_init_template.txt","r",encoding="utf-8") as f :
    incomplete_code = f.read()
complete_code = incomplete_code.format(root)
with open("./TraFiC_init.py","w",encoding="utf-8") as f :
    f.write(complete_code)
