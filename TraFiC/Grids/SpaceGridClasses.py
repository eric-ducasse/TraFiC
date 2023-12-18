# Version 1.6 - 2023, October, 30
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
###############################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
from scipy.special import sici
###############################################################################
class Space1DGrid :
    """A 1D grid in space, with the corresponding grid in the wavenumbers 
       domain, and the associated FFT tools.
            'nb' is the (even) number of discretization points.
            'step' is the discretization step in space."""
    #--------------------------------------------------------------------------
    def __init__(self, nb, step=1.0, verbose =False) :
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        prt = self.__prt
        if nb%2 == 1 :
            nb -= 1
            prt(f"The number of points as to be even: {nb} considered")
        self.__nx = nb
        self.__mx = self.__nx//2
        self.__dx = step
        self.__xmax = self.__dx*self.__mx
        self.__xmin = -self.__xmax + self.__dx
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def nb(self) : 
        """Number of points."""
        return self.__nx
    @property
    def nx(self) : 
        """Number of points."""
        return self.__nx
    @property
    def shape(self) : return (self.__nx,)
    @property
    def step(self) : 
        """Discretization step in space."""
        return self.__dx
    @property
    def dx(self) : 
        """Discretization step in space."""
        return self.__dx
    @property
    def n_max(self) : 
        """Maximum index in the "centered" representation."""
        return self.__mx
    @property
    def n_min(self) : 
        """Minimum index in the "centered" representation."""
        return 1-self.__mx
    @property
    def v_max(self) :
        """Maximum value in space."""
        return self.__xmax
    @property
    def v_min(self) :
        """Minimum value in space."""
        return self.__xmin
    @property
    def xmax(self) :
        """Maximum value in space."""
        return self.__xmax
    @property
    def xmin(self) :
        """Minimum value in space."""
        return self.__xmin
    @property
    def x_max(self) :
        """Maximum value in space."""
        return self.__xmax
    @property
    def x_min(self) :
        """Minimum value in space."""
        return self.__xmin
    @property
    def numbers(self) : 
        """Numbers in the "sorted" representation."""
        return np.append(np.arange(0,self.n_max+1),np.arange(self.n_min,0))
    @property
    def space_values(self) : 
        """Space values in the "sorted" representation."""
        return self.step*self.numbers
    @property
    def X(self) : 
        """Space values in the "sorted" representation."""
        ### Warning: changed at 2023, July, 17. Can create bugs.
        return self.step*self.numbers
    @property
    def Xc(self) : 
        """Space values in the "centered" representation."""
        return self.step*np.arange(self.n_min,self.n_max+1)
    @property
    def g_range(self) : 
        """Range for graphics (imshow)."""
        dxs2 = 0.5*self.step
        return np.array( [self.v_min - dxs2, self.v_max + dxs2 ] )
    @property
    def dk(self) :
        """Discretization step in wavenumber domain."""
        return np.pi/self.v_max
    @property
    def k_max(self):
        """Maximum value in wavenumber domain."""
        return self.dk*self.n_max
    @property
    def k_min(self):
        """Minimum value in wavenumber domain."""
        return self.dk*self.n_min
    @property
    def wavenumber_values(self) :
        """Wavenumber values in the "sorted" representation."""
        return self.dk*self.numbers
    @property
    def K(self) :
        """ = self.wavenumber_values."""
        return self.wavenumber_values
    @property
    def Kc(self) :
        """centered wavenumber_values."""
        return self.sort2cent( self.wavenumber_values )
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __check(self, array, axis) :
        if array.shape[axis] != self.nb :
            msg =  "Space1DGrid :: Incompatible shape!\n\t" + \
                  f"{self.nb} is required  instead of {array.shape[axis]}"
            raise ValueError(msg)
        return True
    def sort2cent(self, array, axis=0) :
        """From "sorted" representation to "centered" representation."""
        if not self.__check(array,axis) : return None
        return np.roll(array,self.n_max-1,axis=axis)        
    def cent2sort(self, array, axis=0) :
        """From "centered" representation to "sorted" representation."""
        if not self.__check(array, axis) : return None
        return np.roll(array, 1-self.n_max, axis=axis)        
    def __str__(self) :
        fmt = "1D grid of {} points, from {:.3e} to {:.3e} (step: {:.3e})"
        return fmt.format(self.nb, self.v_min, self.v_max, self.step)
    def fft(self, array, axis=0, centered=False) :
        """Returns a numerical estimation of the Fourier transform
           (integrate(f(x) exp(i k x) dx). 'centered' indicates
           if coordinates are centered or not in both domains."""
        if not self.__check(array, axis) : return None
        if centered : array = self.cent2sort(array, axis=axis)
        fft_array = ifft(array, axis=axis) * (self.nb*self.step)
        if centered : return self.sort2cent(fft_array, axis=axis)
        return fft_array
    def ifft(self, array, axis=0, centered=False) :
        """Returns a numerical estimation of the Fourier transform
           ((1/2pi)*integrate(f(k) exp(-i k x) dk).
           'centered' indicates if coordinates are centered
           or not in both domains."""
        if not self.__check(array,axis) : return None
        if centered : array = self.cent2sort(array, axis=axis)
        ifft_array = fft(array, axis=axis) / (self.nb*self.step)
        if centered : return self.sort2cent(ifft_array, axis=axis)
        return ifft_array    
    def derivative(self, array, i, axis=0, centered=False) :
        """Returns a numerical estimation of the i-th derivative of
           the field associated to 'array'. axis corresponds to the
           variable position.""" 
        if not self.__check(array, axis) : return None
        if centered : array = self.cent2sort(array, axis=axis)
        # Array index names for einsum
        array_indexes = ["-"]*len(array.shape)
        array_indexes[axis] = "i"            
        array_indexes = "".join(array_indexes)
        no_char = 106 # "j"
        while "-" in array_indexes :
            array_indexes = array_indexes.replace("-",chr(no_char),1)
            no_char += 1  
        fft_array = self.fft(array,axis=axis)
        Ki = (-1j*self.wavenumber_values)**i
        Ki[self.n_max] = 0.0
        fft_array = np.einsum("i,"+array_indexes+"->"+array_indexes,\
                                  Ki,fft_array)
        new_array = self.ifft(fft_array,axis=axis)
        if centered : new_array = self.sort2cent(new_array,axis=axis)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return new_array
        else : # Real-valued array
            Re_max,Im_max = abs(new_array.real).max(),abs(new_array.imag).max()
            if Im_max > 1e-8*Re_max :
                msg = "Space1DGrid.derivative :: Warning: non neglictible "+\
                      "imaginary part of the obtained array."
                print(msg)
            return new_array.real
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def zero_padding(self, array, cz, axis=0, centered=False) :
        """Returns a (g_zp,array_zp) where g_zp is a Space1DGrid with a
           smaller step and array_zp is built from array."""
        TF_array = self.fft(array,axis=axis,centered=centered)
        if centered : TF_array = self.cent2sort(TF_array,axis=axis)
        mz = int(np.ceil(cz*self.__mx))
        shape_zp = list(TF_array.shape)
        new_nb = self.nb+2*mz
        shape_zp[axis] = new_nb
        g_zp = Space1DGrid(new_nb,self.step*self.nb/new_nb)
        slices = [slice(None) for i in shape_zp]
        TF_array_zp = np.zeros( shape_zp, dtype=np.complex128 )
        slices[axis] = slice(self.n_max+1)
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[axis] = slice(self.n_max,self.n_max+1)
        TF_array_zp[tuple(slices)] = 0.5*TF_array_zp[tuple(slices)]
        slices[axis] = slice(self.n_min-1,None)
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[axis] = slice(self.n_min-1,self.n_min)
        TF_array_zp[tuple(slices)] = 0.5*TF_array_zp[tuple(slices)]
        array_zp = g_zp.ifft(TF_array_zp,axis=axis)
        if centered : array_zp = g_zp.sort2cent(array_zp,axis=axis)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return (g_zp,array_zp)
        else : # Real-valued array
            Re_max,Im_max = abs(array_zp.real).max(),abs(array_zp.imag).max()
            if Im_max > 1e-8*Re_max :
                msg = "Space1DGrid.zero_padding :: Warning: non neglictible "+\
                      "imaginary part of the obtained array."
                print(msg)
            return (g_zp,array_zp.real)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def Inum(X, Y) :
        """ufunc which returns a numerical evaluation of the
           integral over (-oo,0) of sinc(x-xi) sinc(x-yj) dx,
           for all pairs (i,j), where sinc(x) = sin(pi x)/(pi x).
           X and Y can be scalars or vectors and all xi and yj
           must be positive."""
        X_is_value = ( np.ndim(X) == 0 )
        if X_is_value : X = [X]
        Y_is_value = ( np.ndim(Y) == 0 )
        if Y_is_value : Y = [Y]
        X,Y = np.array(X), np.array(Y)
        try :
            assert np.all(X>0) and np.all(Y>0)
        except :
            raise ValueError("Error in 'Space1DGrid.Inum' staticmethod:\n" + \
                             "\tall parameters have to be strictly positive.")
        dpiX, dpiY = np.pi*X, np.pi*Y
        # Matrices pi*X, pi*Y
        piX, piY = np.meshgrid(dpiX, dpiY, indexing = "ij") 
        dpiX *= 2 # Vector 2*pi*X
        dpiY *= 2 # Vector 2*pi*Y
        piXmY = piX - piY # Matrix pi*(X-Y)
        SiX, CiX = sici( dpiX ) # Vectors of integral sine and cosine of X
        SiY, CiY = sici( dpiY ) # Idem for Y
        idx_diag = np.where( np.isclose( piX, piY , rtol=1e-7) )
        piXmY[idx_diag] = 1.0 # To avoid division by zero
        # xi != yj
        R = 0.5 * ( ( np.log(piX/piY) - \
                      np.einsum("i,j->ij",CiX,np.ones_like(CiY)) + \
                      np.einsum("i,j->ij",np.ones_like(CiX),CiY) ) \
                    * np.cos(piXmY) - \
                    ( np.einsum("i,j->ij",SiX,np.ones_like(SiY)) + \
                      np.einsum("i,j->ij",np.ones_like(SiX),SiY) - np.pi ) \
                    * np.sin(piXmY) ) / (np.pi*piXmY)
        # xi == yj
        piX_diag = piX[idx_diag]
        SiX_diag = SiX[idx_diag[0]]
        R[idx_diag] = ( np.sin(piX_diag)**2 / piX_diag - SiX_diag ) / np.pi \
                      + 0.5
        if X_is_value :
            if Y_is_value :
                return R[0,0]
            else :
                return R[0,:]
        else :
            if Y_is_value :
                return R[:,0]
            else :
                return R
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def OptimizedBasis(n, m, sharpest_peaks_at_sides=0) :
        """Returns :
           1/ a n-by-(n+2m) matrix B representing a canonical (non-local) basis
              of n discrete-variable functions that are zero-valued outside a
              set S of n+2*m integer contiguous values.
           2/ The 2m-by-n' matrix A2C which gives the first m and last m values
              with respect to the n central values of any function it basis B.
              n' = n - 2*sharpest_peaks_at_sides must be >= 0/
           3/ The sharpest peak values (empty if sharpest_peaks_at_sides=0).
           4/ An estimation of the relative negligibility of the corresponding
              continuous-variable function outside the interval [b-dx, e-dx],
              where dx is the discretization step, b and e are the first and
              last values of S."""
        SG = Space1DGrid
        sps = sharpest_peaks_at_sides
        try :
           assert n > 0 and m > 0
           n = round(n)
           m = round(m)
        except :
            raise ValueError("Error in 'Space1DGrid.OptimizedBasis' " + \
                             "static method:\n\tn and m have to be " + \
                             "positive integers.")
        try :
            assert 0 <= sps <= n//2
            sps = round(sps)
        except :
            raise ValueError("Error in 'Space1DGrid.OptimizedBasis' " + \
                             "static method:\n\tsharpest_peaks_at_sides " + \
                             f"must be an integer in [0,{n//2}].")
        w = n+2*m
        B = np.zeros( (n,w) )
        if sps > 0 :
            peak, _, _, err = SG.OptimizedBasis(1, m)
            peak = peak[0]
            npeak = peak.shape[0]
            for i in range(sps) :
                B[i,i:i+npeak] = peak
                B[n-i-1,w-i-npeak:w-i] = peak
            n -= 2 * sps
        else :
            peak = np.array([])
            err = 0.0
        if n > 0 :
            a = np.arange(n)
            c = np.append( np.arange(-m, 0), np.arange(n, m+n) )
            Maa = SG.Inum(m+1+a,m+1+a) + SG.Inum(n+m-a,n+m-a)
            Mca = SG.Inum(c+m+1,a+m+1) + SG.Inum(n+m-c,n+m-a)
            Mcc = SG.Inum(c+m+1,c+m+1) + SG.Inum(n+m-c,n+m-c)
            A2C = np.linalg.solve(Mcc, -Mca)
            Bsup = np.concatenate( [ A2C[:m,:], np.eye(n), A2C[-m:,:] ] ).T
            err_max = np.linalg.eigvalsh( Maa ).max()
            Merr = Maa +  Mca.T@A2C
            err = max( np.linalg.eigvalsh( Merr ).max()/err_max, err)
            B[sps:sps+n, sps:w-sps] = Bsup
        else :
            A2C = np.array([])
        return B, A2C, peak, err
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @staticmethod
    def OrthonormalBasis(n, m, sharpest_peaks_at_sides=0, verification = False) :
        """Returns an orthonormal basis (L2 scalar product )of size n defined
           on 2n+4m+1 evenly spaced values (step 1/2) for which the support
           of corresponding continuous-variable functions are included in a
           common interval of length (n+2m+1) symmetrically containing the
           2n+4m+1 values."""
        SG = Space1DGrid
        B, _, _, err = SG.OptimizedBasis(n, m, sharpest_peaks_at_sides)
        if n%2 == 0 : # Even n
            nL, nR = 1,1
            size = n + 2*m + 2
        else :# Odd n
            nL, nR = 1,2
            size = n + 2*m + 3
        B = np.concatenate([ np.zeros((n,nL)), B, np.zeros((n,nR)) ], \
                           axis=1)
        local_grid = SG( size )
        _,Bdouble = local_grid.zero_padding(B, 1, axis=1, centered=True)
        S = 0.5 * Bdouble@Bdouble.T # integral of b_i*b_j on their common
                                    # support of length n+2m+1
        L, P = np.linalg.eigh( S )  # Diagonalization in orthonormal basis
        B_ortho = np.diag( 1./np.sqrt(L) )@P.T@Bdouble
        B_ortho = B_ortho[:, 2*nL: -2*nR+1]
        if verification :
            print( "Orthonormality checking:", \
                   np.allclose(0.5*B_ortho@B_ortho.T, np.eye(n)) )
        return B_ortho
###############################################################################
if __name__ == "__main__" :
    # Space1DGrid Example
    if False : # Change True/False to show example
        float_prt = lambda x : f"{x:.3f}"
        def complex_prt(z) :
            a,b = z.real,z.imag
            if b >= 0 : return f"{a:.3f}+{b:.3f}j"
            else : return f"{a:.3f}{b:.3f}j"
        np.set_printoptions(formatter={"complex_kind":complex_prt,\
                                       "float_kind":float_prt})
        gt = Space1DGrid(8,0.5)
        print(gt)
        print("(sorted) space vector:",gt.space_values)
        print("centered space vector:",gt.Xc )
        cases,opt = ["[0,1,0,0,0,0,0,0]","[0,0,0,0,1,0,0,0]"],\
                    ["",",centered=True"]
        for c,o in zip(cases,opt) :
            test = "gt.fft(np.array("+c+")"+o+")"
            tf = eval(test)
            print("tf =",test,"; tf:")
            print(tf)
            test2 = "gt.ifft(tf"+o+")"
            print(test2,"->",eval(test2))
        print("sorted wavenumbers:", gt.wavenumber_values)
        print("centered wavenumbers:", gt.Kc )
        A = np.array([[0,1,0,0,0,0,0,0], [0,0,0.22,0.78,1,0.78,0.22,0] ])
        gzp,Azp = gt.zero_padding(A, 5, axis=1, centered=True)
        plt.figure("Zero-padding 1D")
        plt.plot(gzp.Xc,Azp[0].real,".r")
        plt.plot(gzp.Xc,Azp[1].real,".m")
        plt.plot(gt.Xc,A[0].real,"ob")
        plt.plot(gt.Xc,A[1].real,"sg")
        plt.show()
    # Tools for function bases on an interval [a,b]
    if True :
        examples = [ f"Space1DGrid.Inum({n1},{n2})" for n1,n2 in \
                     ((1,1), (1,2), (1,[1,2]), ([1,2],1), ([1,2,3],[1,2,3])) ]
        for ex in examples :
            result = eval(ex).round(6)
            print(f"***** {ex} *****\n{result}")            
        for n,m,sps in ((1,2,0), (1,3,0), (2,3,0), (8,3,2)) :
            B, A2C, pk, err = Space1DGrid.OptimizedBasis(n, m, sps)
            sz,pts = B.shape
            print(f"***** Space1DGrid.OptimizedBasis({n}, {m}) *****" + \
                  f"\nBasis of size {sz} defined on {pts} points" + \
                  f"\n{A2C.round(3)}\nRelative Error ~ {err:.2e}.")
            if sps > 0 :
                print(f"B matrix:\n{B[:,:7].round(3)}\n{B[:,7:].round(3)}")
        for n,m,sps in ((1,2,0), (1,3,0), (2,3,0), (8,3,2), (20,4,2)) :
            B = Space1DGrid.OrthonormalBasis(n, m, sps, True)
            sz,pts = B.shape
            print(f"***** Space1DGrid.OrthonormalBasis({n}, {m}) *****" + \
                  f"\nBasis of size {sz} defined on {pts} points" + \
                  f"\n{B.round(2)}")
###############################################################################
class Space2DGrid :
    """A 2D grid in space, with the corresponding grid in the wavenumbers 
       domain, and the associated FFT tools."""
    def __init__(self, nx, ny, dx=1.0, dy=None) :
        if dy is None : dy = dx
        self.__grid_x = Space1DGrid(nx, dx)
        self.__grid_y = Space1DGrid(ny, dy)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def x_grid(self) : return self.__grid_x
    @property
    def y_grid(self) : return self.__grid_y
    @property
    def X(self) :
        """x values in the "sorted" representation.""" 
        return self.__grid_x.space_values
    @property
    def Xc(self) :
        """x values in the "centered" representation.""" 
        return self.__grid_x.Xc
    @property
    def nx(self) : return self.__grid_x.nb
    @property
    def nx_max(self) : return self.__grid_x.n_max
    @property
    def nx_min(self) : return self.__grid_x.n_min
    @property
    def dx(self) : return self.__grid_x.step
    @property
    def Y(self) : 
        """y values in the "sorted" representation."""
        return self.__grid_y.space_values
    @property
    def Yc(self) : 
        """y values in the "centered" representation."""
        return self.__grid_y.Xc
    @property
    def ny(self) : return self.__grid_y.nb
    @property
    def ny_max(self) : return self.__grid_y.n_max
    @property
    def ny_min(self) : return self.__grid_y.n_min
    @property
    def dy(self) : return self.__grid_y.step
    @property
    def shape(self) : return (self.ny,self.nx)
    @property
    def MX_MY(self) : return np.meshgrid(self.X,self.Y)
    @property
    def Kx(self) : return self.__grid_x.wavenumber_values
    @property
    def dkx(self) : return self.__grid_x.dk
    @property
    def Ky(self) : return self.__grid_y.wavenumber_values
    @property
    def dky(self) : return self.__grid_y.dk
    @property
    def MKx_MKy(self) : return np.meshgrid(self.Kx,self.Ky)
    @property
    def g_range(self) : 
        """Range for graphics (imshow)."""
        return np.append( self.__grid_x.g_range, self.__grid_y.g_range )
    @property
    def xmin(self) : return self.__grid_x.v_min
    @property
    def x_min(self) : return self.__grid_x.v_min
    @property
    def xmax(self) : return self.__grid_x.v_max
    @property
    def x_max(self) : return self.__grid_x.v_max
    @property
    def ymin(self) : return self.__grid_y.v_min
    @property
    def y_min(self) : return self.__grid_y.v_min
    @property
    def ymax(self) : return self.__grid_y.v_max
    @property
    def y_max(self) : return self.__grid_y.v_max
    @property
    def kx_min(self) : return self.__grid_x.k_min
    @property
    def kx_max(self) : return self.__grid_x.k_max
    @property
    def ky_min(self) : return self.__grid_y.k_min
    @property
    def ky_max(self) : return self.__grid_y.k_max
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++        
    def __check(self, array, axes) :
        array_shape = (array.shape[axes[0]],array.shape[axes[1]])
        if array_shape != self.shape :
            print("Space2DGrid :: Incompatible shape!", self.shape,\
                  "is required instead of",array_shape)
            return False
        return True
    def sort2cent(self, array, axes=[0,1]) :
            """From "sorted" representation to "centered" representation.
               axes[0] corresponds to y. axes[1] corresponds to x."""
            new_array = self.__grid_x.sort2cent(array, axis=axes[1])
            return self.__grid_y.sort2cent(new_array, axis=axes[0])
    def cent2sort(self, array, axes=[0,1]) :
            """From "centered" representation to "sorted" representation.
               axes[0] corresponds to y. axes[1] corresponds to x."""
            new_array = self.__grid_x.cent2sort(array, axis=axes[1])
            return self.__grid_y.cent2sort(new_array, axis=axes[0])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __str__(self) :
        fmt = "2D grid of (ny={})-by-(nx={}) points, with (x,y) in\n"+\
            " [{:.3e},{:.3e}]x[{:.3e},{:.3e}]\n"+\
            " (steps: dx={:.3e} and dy={:.3e})"
        return fmt.format(self.ny, self.nx, self.__grid_x.v_min,\
                          self.__grid_x.v_max, self.__grid_y.v_min,\
                          self.__grid_y.v_max, self.dx, self.dy)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def fft(self,array,axes=[0,1],centered=False) :
        """Returns a numerical estimation of the Fourier transform
           (integrate(f(x,y) exp[i(kx x + ky y)] dx dy).
           axes[0] corresponds to y. axes[1] corresponds to x."""
        if not self.__check(array,axes) : return None
        if centered : array = self.cent2sort(array,axes=axes)
        fft_array = ifft2(array,axes=axes)*(self.nx*self.ny*self.dx*self.dy)
        if centered : return self.sort2cent(fft_array,axes=axes)
        return fft_array
    def ifft(self,array,axes=[0,1],centered=False) :
        """Returns a numerical estimation of the Fourier transform
           ((1/4pi**2)*integrate(f(kx,ky) exp[-i(kx x + ky y)] dkx dky).
           axes[0] corresponds to ky. axes[1] corresponds to kx.""" 
        if not self.__check(array,axes) : return None
        if centered : array = self.cent2sort(array,axes=axes)
        ifft_array = fft2(array,axes=axes)/(self.nx*self.ny*self.dx*self.dy)
        if centered : return self.sort2cent(ifft_array,axes=axes)
        return ifft_array
    def derivative(self,array,ix,iy,axes=(0,1),centered=False) :
        """Returns a numerical estimation of the Partial derivative of
           the field associated to 'array'. axes[0] corresponds to y/ky.
           axes[1] corresponds to x/kx.""" 
        if not self.__check(array,axes) : return None
        if centered : array = self.cent2sort(array,axes=axes)
        # Array index names for einsum
        array_indexes = ["x"]*len(array.shape)
        for p,c in zip(axes,"ij") :
            array_indexes[p] = c
        array_indexes = "".join(array_indexes)
        no_char = 107 # "k"
        while "x" in array_indexes :
            array_indexes = array_indexes.replace("x",chr(no_char),1)
            no_char += 1  
        fft_array = self.fft(array,axes=axes)
        if ix > 0 :
            Cx = (-1j*self.Kx)**ix
            Cx[self.nx_max] = 0.0
            fft_array = np.einsum("j,"+array_indexes+"->"+array_indexes,\
                                  Cx,fft_array)
        if iy > 0 :
            Cy = (-1j*self.Ky)**iy
            Cy[self.ny_max] = 0.0
            fft_array = np.einsum("i,"+array_indexes+"->"+array_indexes,\
                                  Cy,fft_array)
        new_array = self.ifft(fft_array,axes=axes)
        if centered : new_array = self.sort2cent(new_array,axes=axes)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return new_array
        else : # Real-valued array
            Re_max,Im_max = abs(new_array.real).max(),abs(new_array.imag).max()
            if Im_max > 1e-8*Re_max :
                msg = "Space2DGrid.derivative :: Warning: non neglictible "+\
                      "imaginary part of the obtained array."
                print(msg)
            return new_array.real
    def adim_derivative(self,array,ix,iy,axes=(0,1),centered=False) :
        """Returns a numerical estimation of the Adimensional Partial
           derivative (i.e. dx and dy are considered has units) of
           the field associated to 'array'. axes[0] corresponds to y/ky.
           axes[1] corresponds to x/kx.""" 
        if not self.__check(array,axes) : return None
        if centered : array = self.cent2sort(array,axes=axes)
        # Array index names for einsum
        array_indexes = ["x"]*len(array.shape)
        for p,c in zip(axes,"ij") :
            array_indexes[p] = c
        array_indexes = "".join(array_indexes)
        no_char = 107 # "k"
        while "x" in array_indexes :
            array_indexes = array_indexes.replace("x",chr(no_char),1)
            no_char += 1  
        fft_array = self.fft(array,axes=axes)
        if ix > 0 :
            Cx = (-1j*self.dx*self.Kx)**ix
            Cx[self.nx_max] = 0.0
            fft_array = np.einsum("j,"+array_indexes+"->"+array_indexes,\
                                  Cx,fft_array)
        if iy > 0 :
            Cy = (-1j*self.dy*self.Ky)**iy
            Cy[self.ny_max] = 0.0
            fft_array = np.einsum("i,"+array_indexes+"->"+array_indexes,\
                                  Cy,fft_array)
        new_array = self.ifft(fft_array,axes=axes)
        if centered : new_array = self.sort2cent(new_array,axes=axes)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return new_array
        else : # Real-valued array
            Re_max,Im_max = abs(new_array.real).max(),abs(new_array.imag).max()
            if Im_max > 1e-8*Re_max :
                msg = "Space2DGrid.adim_derivative :: Warning: non "+\
                      "neglictible imaginary part of the obtained array."
                print(msg)
            return new_array.real
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def zero_padding(self,array,cx,cy=None,axes=[0,1],centered=False) :
        """Returns a (g_zp,array_zp) where g_zp is a Space2DGrid with a
           smaller step and array_zp is built from array."""
        if cy is None : cy = cx
        TF_array = self.fft(array,axes=axes,centered=centered)
        if centered : TF_array = self.cent2sort(TF_array,axes=axes)
        mx,my = int(np.ceil(cx*self.nx_max)),int(np.ceil(cy*self.ny_max))
        shape_zp = list(TF_array.shape)
        new_nx,new_ny = self.nx+2*mx,self.ny+2*my
        ax0,ax1 = axes
        shape_zp[ax0],shape_zp[ax1] = new_ny,new_nx
        g_zp = Space2DGrid(new_nx,new_ny,self.dx*self.nx/new_nx,\
                           self.dy*self.ny/new_ny)
        slices = [slice(None) for i in shape_zp]
        TF_array_zp = np.zeros( shape_zp, dtype=np.complex128 )
        deby,debx = slice(self.ny_max+1),slice(self.nx_max+1)
        deby_fin,debx_fin = slice(self.ny_max,self.ny_max+1),\
                            slice(self.nx_max,self.nx_max+1)
        finy,finx = slice(self.ny_min-1,None),slice(self.nx_min-1,None)
        finy_deb,finx_deb = slice(self.ny_min-1,self.ny_min),\
                            slice(self.nx_min-1,self.nx_min)
        # Lower left
        slices[ax0],slices[ax1] = deby,debx
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[ax1] = debx_fin
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        slices[ax0],slices[ax1] = deby_fin,debx
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        # Upper left
        slices[ax0] = finy
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[ax0] = finy_deb
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        slices[ax0],slices[ax1] = finy,debx_fin
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        # Upper right
        slices[ax1] = finx
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[ax1] = finx_deb
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        slices[ax0],slices[ax1] = finy_deb,finx
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        # Lower right
        slices[ax0] = deby
        TF_array_zp[tuple(slices)]=TF_array[tuple(slices)]
        slices[ax0] = deby_fin
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        slices[ax0],slices[ax1] = deby,finx_deb
        TF_array_zp[tuple(slices)]=0.5*TF_array_zp[tuple(slices)]
        array_zp = g_zp.ifft(TF_array_zp,axes=axes)
        if centered : array_zp = g_zp.sort2cent(array_zp,axes=axes)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return (g_zp,array_zp)
        else : # Real-valued array
            Re_max,Im_max = abs(array_zp.real).max(),abs(array_zp.imag).max()
            if Im_max > 1e-8*Re_max :
                msg = "Space2DGrid.zero_padding :: Warning: non " + \
                      "neglictible imaginary part of the obtained array."
                print(msg)
            return (g_zp,array_zp.real) 
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++    
    def plot(self, array, centered=False, show=False, height=7,\
             fig_name=None, draw_axis=None, **kwargs) :
        """Plot a 2d array on the 2d space grid. This plot can be drawn
           on an existing figure (real-valued and complex_valued arrays)
           or on an existing axis (real_valued array only).
           If 'show' is False, returns the axes."""
        if not self.__check(array,(0,1)) : return None
        if len(array.shape) > 2 :
            print("2D arrays only!")
            return None
        if not centered : array = self.sort2cent(array)
        if fig_name is None and draw_axis is None :
            fig_name = "2D-field representation"
        xmin,xmax,ymin,ymax = self.g_range
        Lx_over_Ly = (xmax-xmin)/(ymax-ymin)
        dico_opt = {"interpolation":"none","extent":self.g_range,\
                   "cmap":"seismic","origin":"lower"} # Default options
        if "colorbar" in kwargs.keys() : clrbar = kwargs["colorbar"]
        else : clrbar = True
        if str(array.dtype).startswith("complex") : # 2 subplots
            if draw_axis is not None :
                msg = "Space2DGrid.plot :: Warning:\n\t"
                msg += "draw_axis is specified while the field is "
                msg += "complex-valued."
                print(msg)
            fig = plt.figure(fig_name, figsize=(0.6*Lx_over_Ly*height,height))   
            v_max = max(np.abs(array.real).max(), np.abs(array.imag).max())
            dico_opt["vmin"] = -v_max ; dico_opt["vmax"] = v_max
            for k in dico_opt.keys() :
                if k in kwargs.keys() : dico_opt[k]=kwargs[k]
            ax_re, ax_im = fig.subplots(2,1)
            # Real Part
            ax_re.set_title("Real Part")
            img_re = ax_re.imshow(array.real, **dico_opt)
            if clrbar : 
                d_re = make_axes_locatable(ax_re)
                c_re = d_re.append_axes('right', size='2%', pad=0.06)
                plt.colorbar(img_re, cax=c_re)
            ax_re.set_xlim(xmin,xmax); ax_re.set_ylim(ymin,ymax)
            # Imaginary part
            ax_im.set_title("Imaginary Part")
            img_im = ax_im.imshow(array.imag,**dico_opt)
            if clrbar :
                d_im = make_axes_locatable(ax_im)
                c_im = d_im.append_axes('right', size='2%', pad=0.06)
                plt.colorbar(img_im, cax=c_im)
            ax_im.set_xlim(xmin,xmax);ax_im.set_ylim(ymin,ymax)
            plt.subplots_adjust(left=0.03,right=0.99,bottom=0.03,top=0.93,
                    wspace=0.05,hspace=0.25)
            if show : plt.show()      
            else : return ax_re,ax_im
        else : # 1 subplot
            v_max = abs(array).max()
            dico_opt["vmin"] = -v_max ; dico_opt["vmax"] = v_max
            for k in dico_opt.keys() :
                if k in kwargs.keys() : dico_opt[k]=kwargs[k]     
            if draw_axis is None : # new figure
                plt.figure(fig_name, figsize=(1.2*Lx_over_Ly*height,height))
                draw_axis = plt.subplot(1,1,1)
            img = draw_axis.imshow(array,**dico_opt)
            if clrbar : 
                dvd = make_axes_locatable(draw_axis)
                cax = dvd.append_axes('right', size='2%', pad=0.06)
                plt.colorbar(img, cax=cax)
            draw_axis.set_xlim(xmin,xmax);draw_axis.set_ylim(ymin,ymax)
            if draw_axis is None : 
                plt.subplots_adjust(left=0.05,right=0.99,\
                                    bottom=0.05,top=0.98)
            if show : plt.show()
            else : return draw_axis
###############################################################################
if __name__ == "__main__" :
    # Space2DGrid example
    if False :  # Change True/False to show example
        g2dt = Space2DGrid(8, 6, 0.25, 0.2)
        print(g2dt)
        A = np.zeros( g2dt.shape )
        A[1,2] = 1
        #print("Direct Fourier transform:")
        B = g2dt.fft(A)
        #print(B)
        A[1,2] = 1
        #print("Inverse Fourier transform:")
        #print(g2dt.ifft(B))
        MX,MY = g2dt.MX_MY
        field = np.exp(2j*(MX+2*MY))
        g2dt.plot(field)
        g2dt.plot( 2*abs(g2dt.fft(field)), height=5, fig_name="2D FFT", \
                   show=True)
        g2dt.plot(field, colorbar=False, fig_name="Field without colorbar", \
                  show=True)
        # Drawing on existing axis
        plt.figure("Figure with two subplots", figsize=(7,7))
        ax1,ax2 = plt.subplot(2,1,1), plt.subplot(2,1,2)
        ax1.plot( np.linspace(0,20,201), np.sin(np.linspace(0,20,201)), "m-")
        ax1.grid()    
        g2dt.plot(field.real, draw_axis=ax2)
        plt.show()
        import numpy.random as rd
        R = rd.normal(0, 1, (3,g2dt.ny,g2dt.nx,2,2))
        MX,MY = g2dt.MX_MY
        R = np.einsum("lijmn,ij->lijmn", R, np.exp(-8*(MX**2+MY**2)))
        g2dzp,R2dzp = g2dt.zero_padding(R, 9, axes=(1,2))    
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure("2D Zero-padding on Random Points", figsize=(8,7))
        ax = plt.subplot(1, 1, 1, projection="3d")
        ax.plot(MX.flatten(), MY.flatten(), R[2,:,:,0,1].real.flatten(), '.b')
        MXzp,MYzp = g2dzp.MX_MY
        ax.plot(MXzp.flatten(), MYzp.flatten(), R2dzp[2,:,:,0,1].real.flatten(), \
               '.r', markersize=1)
        plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
        plt.show()

###############################################################################
from scipy.special import j0, j1
class CylSymGrid :
    """A cylindrically symmetric grid in space, with the corresponding grid
       in the wavenumber domain, and the associated transformation tools
       (Fourier-Bessel).
            'nb' is the number of discretization points.
            'step' is the discretization step in space."""
    #--------------------------------------------------------------------------
    def __init__(self, nb, step=1.0, verbose=False) :
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        self.__nr = nb
        self.__dr = step
        self.__rmax = self.__dr * (self.__nr - 1)
        self.__rmin = 0.0
        self.__tabJ0 = j0(self.mat_KR)
        self.__tabJ1 = j1(self.mat_KR)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def nb(self) : 
        """Number of points."""
        return self.__nr
    @property
    def nr(self) : 
        """Number of points."""
        return self.__nr
    @property
    def shape(self) : return (self.__nr,)
    @property
    def step(self) : 
        """Discretization step in space."""
        return self.__dr
    @property
    def dr(self) : 
        """Discretization step in space."""
        return self.__dr
    @property
    def v_max(self) :
        """Maximum value in space."""
        return self.__rmax
    @property
    def v_min(self) :
        """Minimum value in space."""
        return self.__rmin
    @property
    def rmax(self) :
        """Maximum value in space."""
        return self.__rmax
    @property
    def rmin(self) :
        """Minimum value in space."""
        return self.__rmin
    @property
    def r_max(self) :
        """Maximum value in space."""
        return self.__rmax
    @property
    def r_min(self) :
        """Minimum value in space."""
        return self.__rmin
    @property
    def space_values(self) : 
        """Space radial values."""
        return self.__dr * np.arange( self.__nr )
    @property
    def R(self) : 
        """Space radial values ."""
        return self.__dr * np.arange( self.__nr )
    @property
    def g_range(self) : 
        """Range for graphics (imshow)."""
        drs2 = 0.5*self.__dr
        return np.array( [self.__rmin - drs2, self.__rmax + drs2 ] )
    @property
    def dk(self) :
        """Discretization step in wavenumber domain."""
        return np.pi/self.__rmax
    @property
    def k_max(self):
        """Maximum value in wavenumber domain."""
        return self.dk * (self.__nr - 1)
    @property
    def k_min(self):
        """Minimum value in wavenumber domain."""
        return 0.0
    @property
    def wavenumber_values(self) :
        """Wavenumber values."""
        return self.dk * np.arange(self.__nr)
    @property
    def K(self) :
        """ = self.wavenumber_values."""
        return self.wavenumber_values
    @property
    def mat_KR(self) :
        """Matrix of products k*r (k in rows, r in columns)."""
        MK, MR = np.meshgrid( self.R, self.K )
        return MK*MR       
    def __str__(self) :
        fmt = "Cylindrically symmetric grid of {} points,\n\t" + \
              "from 0 to {:.3e} (step: {:.3e})"
        return fmt.format(self.nr, self.r_max, self.dr)
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __check(self, array, axis) :
        if array.shape[axis] != self.nb :
            msg =  "CylSymGrid :: Incompatible shape!\n\t" + \
                  f"{self.nb} is required instead of {array.shape[axis]}"
            raise ValueError(msg)
        return True 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def NHT(self, order, array, axis) :
        """Numerical Hankel (or Fourier-Bessel) Transform.
           Numerical approximation of
              \int_{0}^{r_{max}} f(r) J_{order}(k r) r dr."""
        self.__check(array, axis) 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def iNHT(self, order, array, axis) :
        """Numerical Inverse Hankel (or Fourier-Bessel) Transform.
           Numerical approximation of
              \int_{0}^{k_{max}} f(k) J_{order}(k r) k dk."""
        self.__check(array, axis)
###############################################################################
if __name__ == "__main__" :
    # CylSymGrid example
    if True :  # Change True/False to show example
        cs_gd = CylSymGrid( 11, 0.1 )
        print(cs_gd)
