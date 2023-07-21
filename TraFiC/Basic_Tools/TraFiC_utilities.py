# Version 1.00 - 2023, July 3rd
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import time
#=========================================================================
# Time stamping
def now(with_date=False) :
    "Gives time... and date"
    tm = time.localtime(time.time())
    if with_date :
        txt = "{:04d}-{:02d}-{:02d} ".format(*tm[:3])
    else :
        txt = ""
    return txt + "{:02d}:{:02d}:{:02d}".format(*tm[3:6])
#=========================================================================
if __name__ == "__main__" :
    print(f"now() : '{now()}'\n" + \
          f"now(True) : '{now(True)}'")
#=========================================================================
