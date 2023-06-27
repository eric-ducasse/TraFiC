# Version 1.00 - 2023, June 19
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import os, psutil, time
# Available RAM :
MAXMEM = 16.0 # Fixed maximum
coef_mem = 0.5 # 50%
MAXMEM = min(MAXMEM, \
             round(coef_mem*psutil.virtual_memory().free*2**-30,2))
#=========================================================================
# Time stamping
def now() :
    "Donne l'heure..."
    tm = time.localtime(time.time())
    return "{:02d}:{:02d}:{:02d}".format(*tm[3:6])
#=========================================================================
if __name__ == "__main__" :
    print(f"MAXMEM : {MAXMEM:.2f} GB\n" + \
          f"now() : '{now()}'\n")
#=========================================================================
