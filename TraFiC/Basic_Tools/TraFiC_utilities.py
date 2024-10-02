# Version 1.02 - 2024, March 4th
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#=========================================================================
import time
import numpy as np
#=========================================================================
# Time stamping
def today() :
    tm = time.localtime(time.time())
    return "{:04d}-{:02d}-{:02d} ".format(*tm[:3])
def now(with_date=False) :
    "Gives time... and date"
    tm = time.localtime(time.time())
    if with_date :
        txt = "{:04d}-{:02d}-{:02d} ".format(*tm[:3])
    else :
        txt = ""
    return txt + "{:02d}:{:02d}:{:02d}".format(*tm[3:6])
#=========================================================================
class Quadrisection :
    """For find minimum of a function by quadrisection method."""
    #---------------------------------------------------------------------
    def __init__(self, min_value, max_value) :
        # x values
        self.__x = np.linspace(min_value, max_value, 5)
        # x step
        self.__dx = self.__x[1] - self.__x[0]
        # indexes of the sorted x values
        self.__idx = np.arange(5)
        # values of x for which the function must be evaluated
        self.__to_eval = np.ones(5, dtype=bool)
    #---------------------------------------------------------------------
    def next_iter(self, number) :
        if number == 0 : # Downward translation
                         # 0,1,2 kept |-> 2,3,4; new: 0,1
            idx = np.append( self.__idx[-2:], self.__idx[:3] )
            self.__idx = idx
            for i in (0,1) :
                self.__to_eval[i] = True
            for i in (2,3,4) :
                self.__to_eval[i] = False
            self.__x[ idx[1] ] = self.__x[ idx[2] ] - self.__dx
            self.__x[ idx[0] ] = self.__x[ idx[1] ] - self.__dx
            return
        elif number == 4 : # Upward translation
                           # 2,3,4 kept |-> 0,1,2; new: 3,4
            idx = np.append( self.__idx[-3:], self.__idx[:2] )
            self.__idx = idx
            for i in (3,4) :
                self.__to_eval[i] = True
            for i in (0,1,2) :
                self.__to_eval[i] = False
            self.__x[ idx[3] ] = self.__x[ idx[2] ] + self.__dx
            self.__x[ idx[4] ] = self.__x[ idx[3] ] + self.__dx
            return
        elif number not in (1,2,3) :
            msg = "Quadrisection.next_iter :: Error:\n\t" + \
                  f"{number} not in (0,1,2,3,4)"
            raise ValueError(msg)
        # number in (1,2,3) : width divided by 2
        self.__dx *= 0.5
        if number == 1 : # 0,1,2 kept |-> 0,2,4; new: 1,3
            idx = self.__idx[(0,3,1,4,2),]
        elif number == 2 : # 1,2,3 kept |-> 0,2,4; new: 1,3
            idx = self.__idx[(1,0,2,4,3),]
        elif number == 3 : # 2,3,4 kept |-> 0,2,4; new: 1,3
            idx = self.__idx[(2,0,3,1,4),]
        self.__idx = idx
        for i in (1,3) :
            self.__to_eval[i] = True
        for i in (0,2,4) :
            self.__to_eval[i] = False
        self.__x[ idx[1] ] = self.__x[ idx[2] ] - self.__dx
        self.__x[ idx[3] ] = self.__x[ idx[2] ] + self.__dx
        return
    #---------------------------------------------------------------------
    def __call__(self, i) :
        return self.__x[ self.__idx[i] ]
    @property
    def step(self) : return  self.__dx
    @property
    def values(self) : return  self.__x[self.__idx,]
    @property
    def indexes(self) : return  self.__idx.copy()
    def reverse_index(self, n) :
        for i,j in enumerate(self.__idx) :
            if n == j : return  i
    @property
    def changed(self) : return  self.__to_eval.copy()
    @property
    def raw_data(self) : return  (self.__x.copy(), self.indexes)
    #---------------------------------------------------------------------
    @staticmethod
    def minimize( function, xranges, iter_number = 30 ) :
        """Returns the vector of parameters minimizing the function and
           the minimum value, after 'iter_number' iteration of the
           quadrisection method. 'xranges' denotes the beginning
           interval(s)."""
        try :
            LQS = []
            for vmin, vmax in xranges :
                LQS.append( Quadrisection( vmin, vmax ) )
        except : # xranges is not an iterable of pairs (vmin, vmax)
            try :
                vmin, vmax = xranges
                LQS = [ Quadrisection( vmin, vmax ) ]
            except :
                msg = "Quadrisection.minimize :: Error:\n\t" + \
                      "xranges must be either a pair of numbers or\n" + \
                      "\ta list of pairs of numbers. Not " + \
                      f"'{xranges}'"
                raise ValueError(msg)
        # Initialization
        nb_args = len(LQS)
        shp = nb_args*(5,)
        size = 5**nb_args
        values = np.empty( shp, dtype = np.float64)
        for n_lin in range(size) :
            indexes = np.unravel_index(n_lin, shp)
            Vx = [ qs(i) for i,qs in zip(indexes, LQS) ]
            values[indexes] = function( *Vx )
        idxmin = np.unravel_index(values.argmin(), shp)
        # Iterations
        for _ in range(iter_number) :
            for i,qs in zip(idxmin, LQS) : qs.next_iter(i)
            for n_lin in range(size) :
                indexes = np.unravel_index(n_lin, shp)
                changed = False
                for i,qs in zip(indexes, LQS) :
                    changed = changed or qs.changed[i]
                if changed :
                    Vx, tab_idx = [], []
                    for i,qs in zip(indexes, LQS) :
                        Vx.append( qs(i) )
                        tab_idx.append( qs.indexes[i] )
                    tab_idx = tuple(tab_idx) # Necessary!
                    values[tab_idx] = function( *Vx )
            idxtab = np.unravel_index(values.argmin(), shp)
            idxmin = [qs.reverse_index(i) for i,qs in zip(idxtab, LQS)]
        Vx = np.array([ qs(i) for i,qs in zip(idxmin, LQS) ])
        return Vx, values[idxtab]
#=========================================================================
if __name__ == "__main__" :
    print(f"now() : '{now()}'\n" + \
          f"now(True) : '{now(True)}'")
    if False : # Visualization of the evolution of Quadrisection instance
        import matplotlib.pyplot as plt
        from random import randint
        plt.figure("Quadrisection example", figsize=(15,6))
        plt.subplots_adjust(0.05,0.06,0.78,0.94)
        plt.title("Quadrisection(-4, 4) with successive next_item " + \
                  "callings")
        items = np.arange(11)
        qsect = Quadrisection(-4, 4)
        plt.plot( qsect.values, 5*[0], "or" )
        lbl = ["Start"]
        Vx, Vidx = qsect.raw_data
        lbl2 = [ f"{Vx.round(2)} {Vidx}"] 
        for i in items[1:] :
            nb = randint(0, 4)
            lbl.append( f"next({nb})")
            qsect.next_iter(nb)
            Vx, Vidx = qsect.raw_data
            lbl2.append( f"{Vx.round(2)} {Vidx}" )
            idx_kept = np.where( np.logical_not(qsect.changed) )
            x_kept = qsect.values[idx_kept]
            plt.plot( x_kept, i*np.ones_like(x_kept), "dg" )
            idx_new  = np.where( qsect.changed )
            x_new = qsect.values[idx_new]
            plt.plot( x_new, i*np.ones_like(x_new), "or" )
        plt.ylim(items[-1]+1, -1)
        plt.yticks( items, lbl )
        plt.grid()
        plt.twinx()
        plt.ylim(items[-1]+1, -1)
        plt.yticks( items, lbl2 )
        plt.show()
    f_test1 = lambda x : -0.2 + (x-np.pi)**2
    Vx1,val1 = Quadrisection.minimize( f_test1, [-1, 0])
    print( "Function x |-> -0.2 + (x-pi)**2:\n\t" + \
          f"Minimum for x ~ {Vx1.round(8)} : ~ {val1:.8f})")
    f_test2 = lambda x,y : 1.2 - np.sin(x)*np.sin(2*y+1)
    Vx2,val2 = Quadrisection.minimize( f_test2, [[0.0,1.0],[-0.5,0.5]])
    print( "Function (x,y) |-> 1.2 - sin(x)*sin(2*y+1):\n\t" + \
          f"Minimum for x ~ {Vx2.round(8)} : ~ {val2:.8f})")
#=========================================================================
