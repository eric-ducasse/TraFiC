# Version 1.23 - 2024, February, 29
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
############################################################################
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import rfft,irfft,ifft
from SpaceGridClasses import Space1DGrid
############################################################################
class TimeGrid :
    """A 1D grid in time, with the corresponding grid in the frequency 
       domain, and the associated FFT tools."""
    def __init__(self, duration, dt, t0=0.0, attenuation=1e-5, \
                 verbose=False) :
        """'duration' is the duration of the signals
           'dt' is the sampling period. 't0' is the time origin.
           'attenuation' is the attenuation of the exponential
           windows, equal to exp(-gamma*duration)."""
        self.__dt, self.__t0 = dt, t0 # Sampling period, time origin
        nb = 2 * round( 0.5*duration / dt + 0.2 )
        self.__nt = nb # Number of time values
        self.__d = dt*nb # Duration
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        if abs(self.__d-duration) > 0.1*dt :
            self.__prt("TimeGrid builder :: warning: effective " + \
                       f"duration ~ {self.__d:.2e} s")
        self.__gamma = -np.log(attenuation) / self.__d # Gamma value
        self.__ns = nb//2 + 1 # Number of frequency values
        self.__fmax = 0.5 / dt # Maximum frequency
        self.__df = 1.0 / self.__d # Discretization step in frequency
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def nt(self) : 
        """Number of time values."""
        return self.__nt
    @property
    def n(self) : 
        """Number of time values."""
        return self.__nt
    @property
    def ns(self) : 
        """Number of frequency values."""
        return self.__ns
    @property
    def nf(self) : 
        """Number of frequency values."""
        return self.__ns
    @property
    def step(self) : 
        """Sampling Period."""
        return self.__dt
    @property
    def dt(self) : 
        """Sampling Period."""
        return self.__dt
    @property
    def Ts(self) : 
        """Sampling Period."""
        return self.__dt
    @property
    def t0(self) : 
        """Time Origin."""
        return self.__t0
    @property
    def t0(self) : 
        """Time Origin."""
        return self.__t0
    @property
    def n_max(self) : 
        """Maximum index in frequencies."""
        return self.__ns-1
    @property
    def duration(self) :
        """Duration."""
        return self.__d
    @property
    def d(self) :
        """Duration."""
        return self.__d
    @property
    def gamma(self) :
        """Gamma coefficient such that
               exp(-gamma*duration) ~ attenuation."""
        return self.__gamma
    @property
    def attenuation(self) :
        """Attenuation at the end of the exponential window,
               equal to exp(-gamma*duration)."""
        return np.exp(-self.__gamma*self.__d)
    @property
    def time_values(self) : 
        """Time values."""
        return self.__dt*np.arange(self.__nt) + self.__t0
    @property
    def T(self) : 
        """Time values."""
        return self.time_values
    @property
    def frequency_values(self) : 
        """Frequency values."""
        return self.__df * np.arange(self.__ns)
    @property
    def F(self) : 
        """Frequency values."""
        return self.frequency_values
    @property
    def S(self) : 
        """Complex values of the Laplace variable."""
        return self.gamma + 2j*np.pi*self.F
    @property
    def df(self) :
        """Discretization step in frequency domain."""
        return self.__df
    @property
    def f_max(self):
        """Maximum frequency."""
        return self.__fmax
    @property
    def t_range(self) : 
        """Time range for graphics."""
        b = self.__t0 - 0.5*self.__dt
        return np.array( [ b, b + self.__d ] )
    @property
    def f_range(self) : 
        """Frequency range for graphics."""
        b = - 0.5*self.__df
        return np.array( [ b, self.__fmax - b ] )
    #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __check(self, array, axis, frequency_domain=False) :
        array = np.array(array)
        if frequency_domain :
            dom = "frequency domain"
            nb = self.__ns
        else : 
            dom = "time domain"
            nb = self.__nt
        if array.shape[axis] != nb :
            msg = f"TimeGrid :: Incompatible shape! {nb} is required " + \
                  f"instead of {array.shape[axis]} in the {dom}."
            raise ValueError(msg)
        return True
    #-------------------------------------
    def __str__(self) :
        fmt = "Time grid of {} points, duration {:.3e} s (step: {:.3e} s)"
        return fmt.format(self.n,self.duration,self.dt)
    #-------------------------------------
    @staticmethod
    def array_indexes(array,axis) :
        array = np.array(array)
        shp = array.shape
        indexes = ""
        for i,n in enumerate(shp) :
            if i != axis :
                indexes += chr(i+106)
            else :
                indexes += "i"
        return array,indexes+",i->"+indexes
    #-------------------------------------        
    def rfft(self, array, axis=0, t0_into_account=True) :
        """Returns a numerical estimation of the Fourier transform
           (integrate(s(t) exp(-2 i pi f t) dt)"""
        if not self.__check(array,axis) : return None
        array,indexes = self.array_indexes(array,axis)
        if t0_into_account :
            R = self.dt*np.exp(-2j*np.pi*self.t0*self.F)
            rfft_array = np.einsum(indexes,rfft(array,axis=axis),R)
        else :
            rfft_array = self.dt*rfft(array,axis=axis)
        return rfft_array
    #-------------------------------------
    def irfft(self, array, axis=0, t0_into_account=True) :
        """Returns a numerical estimation of the inverse Fourier transform
           (integrate(s(f) exp(2 i pi f t) df)"""
        if not self.__check(array, axis, frequency_domain=True) :
            return None
        array,indexes = self.array_indexes(array, axis)
        if t0_into_account :
            V = np.exp( 2j*np.pi*self.t0*self.F )
            array = np.einsum(indexes, array, V)
        # The last value of the rfft has to be real
        slicing = [ slice(None) for s in array.shape ]
        slicing[axis] = -1
        slicing = tuple(slicing)
        array[slicing] = array[slicing].real
        return irfft(array,axis=axis)/self.dt
    #-------------------------------------
    def LT(self, array, axis=0) :
        """Returns a numerical estimation of the Laplace transform
           (integrate(s(t-t0) exp(-(gamma + 2 i pi f)*(t-t0)) dt)"""
        array = np.array(array)
        if not self.__check(array,axis) : return None
        array,indexes = self.array_indexes(array,axis)
        EG = np.exp(-self.gamma*(self.T-self.t0))
        array = np.einsum(indexes,array,EG)
        return self.rfft(array, axis=axis, t0_into_account=False)
    #-------------------------------------
    def iLT(self, array, axis=0) :
        """Returns a numerical estimation of the signal from its
           Laplace transform, with dt = t - t0 and iw = 2*i*pi*f:
           exp(gamma*dt)*( integrate( S(gamma+iw*dt) exp(iw*dt) df ) ).
        """
        signals = self.irfft(array, axis=axis, t0_into_account=False)
        signals,indexes = self.array_indexes(signals,axis)
        EG = np.exp(self.gamma*(self.T-self.t0))
        return np.einsum(indexes,signals,EG)
    #-------------------------------------        
    def derivative(self,array,i,axis=0) :
        """Returns a numerical estimation of the i-th derivative of
           the field associated to 'array'. axis corresponds to the
           variable position.""" 
        if not self.__check(array,axis) : return None
        # Array index names for einsum
        array = np.array(array)
        shp = array.shape
        indexes = ""
        for i,n in enumerate(shp,106) :
            if i != axis :
                indexes += chr(i)
            else :
                indexes += "i"
        indexes = indexes+",i->"+indexes
        fft_array = self.rfft(array,axis=axis)
        Wi = (2j*np.pi*self.frequency_values)**i
        Wi[-1] = 0.0
        fft_array = np.einsum(indexes,fft_array,Wi)
        return self.irfft(fft_array,axis=axis)
    #-------------------------------------
    def zero_padding(self, array, cz, axis=0, Laplace=False) :
        """Returns (g_zp,array_zp) where g_zp is a TimeGrid with a
           smaller step and array_zp is built from array."""
        if Laplace : TF_array = self.LT(array, axis=axis)
        else : TF_array = self.rfft(array, axis=axis)
        mz = round( np.ceil( cz*self.n_max ) )
        shape_zp = list(TF_array.shape)
        new_nf = self.__ns + mz
        shape_zp[axis] = new_nf
        new_nt = 2*(new_nf-1)
        g_zp = TimeGrid( self.duration, self.dt*self.nt/new_nt, self.t0)
        slices = [slice(None) for i in shape_zp]
        TF_array_zp = np.zeros( shape_zp, dtype=np.complex128 )
        slices[axis] = slice(self.n_max)
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[axis] = slice(self.n_max,self.n_max+1)
        TF_array_zp[tuple(slices)] = 0.5*TF_array[tuple(slices)]
        if Laplace : array_zp = g_zp.iLT(TF_array_zp, axis=axis)
        else : array_zp = g_zp.irfft(TF_array_zp, axis=axis)
        return (g_zp,array_zp)
    #-------------------------------------
    def envelope(self, array, cz, axis=0) :
        """ Returns (g_zp,array_zp) where g_zp is a TimeGrid with a
           smaller step and array_zp is the envelope built from array."""
        TF_array = rfft(array,axis=axis)*self.Ts
        n = self.n
        mz = int(np.ceil(cz*self.n_max))
        shape_env = list(TF_array.shape)
        new_nt = n + 2*mz
        g_zp = TimeGrid(self.duration,self.dt*self.nt/new_nt,self.t0)
        shape_env[axis] = new_nt
        slices = [slice(None) for i in shape_env]
        Y = np.zeros( shape_env, dtype = np.complex128)
        slices[axis] = slice(self.n_max)
        Y[tuple(slices)] = 2.0*TF_array[tuple(slices)]
        slices[axis] = slice(self.n_max,self.n_max+1)
        Y[tuple(slices)] = TF_array[tuple(slices)]
        X = ifft(Y,axis=axis)/g_zp.Ts
        return g_zp,np.abs(X)
    #-------------------------------------
    def enhance_causality(self, array, axis=0, method="least-squares", \
                          alpha=1.0, nb=3, threshold=1e-4,
                          zero_valued=1e-8, verbose = False):
        add_values = TimeGrid.add_values_for_enhance_causality
        shp = array.shape
        if shp[axis] != self.nt :
            msg = "TimeGrid.enhance_causality :: Error:\n\t" + \
                 f"Incompatible sizes {self.nt} <=> {shp[axis]}."
            raise ValueError(msg)
        if len(shp) == 1 : # Vector
            values, ib1, n = TimeGrid.clean_values(array, zero_valued)
            nbmax = min(12, ib1//3)
            values, ib2, rel_err = add_values(values, method, False, \
                                              alpha, nbmax, nb, \
                                              threshold, zero_valued, \
                                              verbose)
            idx_beg = ib1+ib2
            new_array = np.zeros_like( array )
            new_array[idx_beg:idx_beg+values.shape[0]] = values
            return new_array, rel_err
        # ndim > 1
        tab = array.swapaxes( axis, -1 ) 
        orig_shape = tab.shape
        tab.shape = ( np.prod(orig_shape[:-1]), orig_shape[-1] )
        rel_err = 0.0
        for i,row in enumerate(tab) :
            values, ib1, n = TimeGrid.clean_values(row, zero_valued)
            nbmax = min(12, ib1//3)
            values, ib2, re = add_values(values, method, False, alpha, \
                                         nbmax, nb, threshold, \
                                         zero_valued, verbose)
            idx_beg = ib1+ib2
            row = np.zeros_like( row )
            row[idx_beg:idx_beg+values.shape[0]] = values
            tab[i,:] = row
            rel_err = max(rel_err, re)
        tab.shape = orig_shape 
        return tab.swapaxes( axis, -1 ), rel_err
    #-------------------------------------
    @staticmethod
    def clean_values(values, zero_valued=1e-8) :
        values = np.array(values)
        n = values.shape[0]
        norm_val = np.abs(values).max()
        if norm_val == 0.0 :
            return values, 0, 0
        # Cleaning zero values
        values = (np.abs(values) >= zero_valued*norm_val)*values
        zero_values = ( values == 0.0 )
        if zero_values.any() :
            # Index of the first nonzero value :
            idx_beg = zero_values.argmin() 
            # Index of the last nonzero value + 1 :
            idx_end = n - zero_values[::-1].argmin()
            values = values[idx_beg:idx_end]
            n = idx_end - idx_beg
        else :            
            idx_beg = 0
            idx_end = n
        return values, idx_beg, n
    #-------------------------------------
    @staticmethod
    def add_values_for_enhance_causality(values, method="least-squares", \
                                         clean_values=True,
                                         alpha=1.0, nbmax=12, nb=3, \
                                         threshold=1e-4, zero_valued=1e-8,\
                                         verbose = True):
        """add values before the vector 'values' for minimizing the
           oscillations of the corresponding continuous-time signal
           before these values.
           'method' is either "least-squares" or "maximum".
           * least-squares method:
             + 'alpha' is a parameter in (0,1]:
                   1 gives the best convergence with possible oscillatory
                       effect (large added values).
                   Small values gives slower convergence with small added
                       values.
             + 'nbmax' is the maximal number of added values.
             + 'nb' is the size of the block for each iteration
           * maximum method: (NOT YET IMPLEMENTED)
             + minimizes de amplitude of parasitic oscillations
             + time consuming method
           A value is considered zero if its absolute value is less 
           than 'zero_valued'. 'threshold' is the goal ratio of norms
           |parasitic oscillations| / |values|. 
           
        """
        if clean_values :
            values, idx_beg, n = TimeGrid.clean_values(values, zero_valued)
            if n == 0 : return values, 0, 0.0
        else :
            values, idx_beg, n = np.array(values), 0, len(values)
        norm_val = np.sqrt( values@values / n )
        if method == "least-squares" :
            # Matrix for compute the quadratic error
            Inum = Space1DGrid.Inum
            Va = alpha + np.arange(n+nbmax)
            M = Inum( Va, Va )
            err2 = values@M[:n,:n]@values
            rel_err = np.sqrt(err2)/norm_val
            if rel_err <= threshold :
                return values, idx_beg, rel_err
            # Coefficient for compute the additional value
            nb = 3 # Number of values to add at each iteration
            Mc = - np.linalg.inv(M[:nb,:nb])@M[:nb, nb:] 
            for k in range(nbmax//nb) :
                add_values = Mc[:,:n]@values
                values = np.append( add_values, values )
                idx_beg -= nb
                n += nb
                err2 = values@M[:n,:n]@values
                err1 = values[nb:]@M[nb:n,nb:n]@values[nb:]
                print(f"k = {k}, {err1:.2e} -> {err2:.2e}")
                rel_err = np.sqrt(err2)/norm_val 
                if rel_err <= threshold :
                    return values, idx_beg, rel_err
            if verbose :
                print(f"Warning: Relative error <= {threshold:.2e} " + \
                      f"not reached!\n\t{rel_err:.2e} for {nbmax} " + \
                       "additional values.")
            return values, idx_beg, rel_err
        elif method == "maximum" :
            subdiv = 5
            #$$$$$$$$$$$$$$$$$$$
            #$$$$$ TO DO $$$$$$$
            #$$$$$$$$$$$$$$$$$$$
        else :
            msg = "TimeGrid.add_values_for_enhance_causality error:\n" + \
                  f"\tUnknown '{method}'."
            raise ValueError(msg)
############################################################################
if __name__ == "__main__" :
    float_prt = lambda x : f"{x:.3f}"
    def complex_prt(z) :
        a,b = z.real,z.imag
        if b >= 0 : return f"{a:.3f}+{b:.3f}j"
        else : return f"{a:.3f}{b:.3f}j"
    np.set_printoptions(formatter={"complex_kind":complex_prt,\
                                   "float_kind":float_prt})
    if False : # Basic tests
        gt = TimeGrid(4.0,0.5,-1)
        print(gt)
        print("Time vector:",gt.T)
        print("Frequencies:",gt.F)
        M = [[1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0]]
        test = "gt.rfft({},axis=1)".format(M)
        ft = eval(test)
        print(test,"-> ft:")
        print(ft)
        test2 = "gt.irfft(ft,axis=1)"
        print(test2,"->")
        print(eval(test2))
        A = np.array([[0,0,0,1,0,0,0,0],[0,0,0.22,0.78,1,0.78,0.22,0]])
        gzp,Azp = gt.zero_padding(A,10,axis=1)
        _,Bzp = gt.zero_padding(A,10,axis=1,Laplace=True)
        plt.figure("Zero-padding 1D")
        plt.plot(gzp.T,Azp[0].real,".r")
        plt.plot(gzp.T,Azp[1].real,".m")
        plt.plot(gzp.T,Bzp[0].real,"+r",markersize=4.0,markeredgewidth=1.5)
        plt.plot(gzp.T,Bzp[1].real,"+m",markersize=4.0,markeredgewidth=1.5)
        plt.plot(gt.T,A[0].real,"ob")
        plt.plot(gt.T,A[1].real,"sg")
        plt.ylim(-5.7,3.4)
        plt.show()
    if False : # Test of envelope
        tgenv = TimeGrid(10,0.1,-4)
        T = tgenv.T
        signal = np.sin(2*np.pi*T)*np.exp(-T**2)+\
                 0.3*np.sin(2.3*np.pi*T)*np.exp(-1.5*(T-3.1)**2)
        plt.plot(tgenv.T,signal,".r")
        zpg0,sig_zp = tgenv.zero_padding(signal,5)
        plt.plot(zpg0.T,sig_zp,"-g")
        zpg,sig_env = tgenv.envelope(signal,10)
        plt.plot(zpg.T,sig_env,"-b")
        plt.plot(zpg.T,-sig_env,"--b")
        plt.show()
    # Test of add_values_for_enhance_causality (static method)
    if False :
        values = [0.1,0.4,1.0,0.2,-0.9,-0.3,-0.1]
        nb_val = len(values)
        idx0 = 50
        tm_gd = TimeGrid(170,1.0,-idx0)
        all_values = np.zeros( tm_gd.nt )
        all_values[idx0:idx0+nb_val] = values
        tm_zp, zp_values = tm_gd.zero_padding(all_values, 9.0)
        fig = plt.figure("Test of enhance_causality functions", \
                         figsize = (14,7) )
        fig.subplots_adjust(top=0.96, bottom=0.09, left=0.05, \
                            right=0.99, hspace=0.25, wspace=0.2 )
        axes = list(zip(fig.subplots(2,1),(1,1e4)))
        for ax,c in axes : 
            ax.plot(tm_zp.T, c*zp_values, "-b")
            ax.plot(tm_gd.T, c*all_values, "or")
        new_val, idx_beg, err = \
            TimeGrid.add_values_for_enhance_causality(values)    
        new_values = np.zeros( tm_gd.nt )
        new_values[idx0+idx_beg:idx0+idx_beg+new_val.shape[0]] = new_val
        _, zp_new_values = tm_gd.zero_padding(new_values, 9.0)
        for ax,c in axes : 
            ax.plot(tm_zp.T, c*zp_new_values, "--g")
            ax.plot(tm_gd.T, c*new_values, "dc", markersize = 4)
        nv1, err1 = tm_gd.enhance_causality( all_values )
        for ax,c in axes : 
            ax.plot(tm_gd.T, c*nv1, "om", markersize = 1.5)
        nv2, err2 = tm_gd.enhance_causality( \
                       all_values.reshape(all_values.shape+(1,1))  )
        for ax,c in axes :
            ax.plot(tm_gd.T, c*nv2[:,0,0], "+m", markersize = 3.0)
            ax.set_xlim(-idx0-0.5, 10)
            ax.grid()
        axes[0][0].set_title("All values")
        axes[0][0].set_ylabel("Values")
        coef = axes[1][1]
        val_max = 8.5e-4 * coef
        axes[1][0].set_title(f"Zoom x{coef:.1e} on smallest values")
        axes[1][0].set_ylim(-val_max, val_max)
        axes[1][0].set_ylabel(f"Values [x{1/coef:.2e}]")
        plt.show()
