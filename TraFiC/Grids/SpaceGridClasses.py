# Version 1.65 - 2025, August, 20
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
###############################################################################
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
if np.__version__[0] == "1":
    solve = np.linalg.solve
elif np.__version__[0] == "2": # solve must be redefined with numpy 2.x
    def solve(A, b):
        shp_M = A.shape
        shp_V = b.shape
        matrices_vectors =  len(shp_M) == len(shp_V)+1
        if matrices_vectors: b.shape = shp_V+(1,)
        X = np.linalg.solve(A, b)
        if matrices_vectors:
            b.shape = shp_V
            X.shape = X.shape[:-1]
        return X
else:
    raise SystemError(f"Unknown '{np.__version__}' of numpy.")
from scipy.interpolate import interp1d
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
        return np.append(np.arange(0, self.n_max+1),
                         np.arange(self.n_min, 0))
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
        return self.step*np.arange(self.n_min, self.n_max+1)
    @property
    def g_range(self) : 
        """Range for graphics (imshow)."""
        dxs2 = 0.5*self.step
        return np.array( [self.v_min - dxs2, self.v_max + dxs2] )
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
            msg = ("Space1DGrid :: Incompatible shape!\n\t"
                   + "{self.nb} is required  instead of {array.shape[axis]}")
            raise ValueError(msg)
        return True
    def sort2cent(self, array, axis=0) :
        """From "sorted" representation to "centered" representation."""
        if not self.__check(array, axis) : return None
        return np.roll(array, self.n_max-1, axis=axis)        
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
            array_indexes = array_indexes.replace("-", chr(no_char), 1)
            no_char += 1  
        fft_array = self.fft(array, axis=axis)
        Ki = (-1j*self.wavenumber_values)**i
        Ki[self.n_max] = 0.0
        fft_array = np.einsum("i,"+array_indexes+"->"+array_indexes,
                              Ki, fft_array)
        new_array = self.ifft(fft_array, axis=axis)
        if centered : new_array = self.sort2cent(new_array, axis=axis)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return new_array
        else : # Real-valued array
            Re_max,Im_max = (abs(new_array.real).max(),
                             abs(new_array.imag).max())
            if Im_max > 1e-8*Re_max :
                msg = ("Space1DGrid.derivative :: Warning: non neglictible "
                       + "imaginary part of the obtained array.")
                print(msg)
            return new_array.real
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def zero_padding(self, array, cz, axis=0, centered=False) :
        """Returns a (g_zp,array_zp) where g_zp is a Space1DGrid with a
           smaller step and array_zp is built from array."""
        TF_array = self.fft(array, axis=axis, centered=centered)
        if centered : TF_array = self.cent2sort(TF_array, axis=axis)
        mz = int(np.ceil(cz*self.__mx))
        shape_zp = list(TF_array.shape)
        new_nb = self.nb+2*mz
        shape_zp[axis] = new_nb
        g_zp = Space1DGrid(new_nb, self.step*self.nb/new_nb)
        slices = [slice(None) for i in shape_zp]
        TF_array_zp = np.zeros( shape_zp, dtype=np.complex128 )
        slices[axis] = slice(self.n_max+1)
        TF_array_zp[tuple(slices)] = TF_array[tuple(slices)]
        slices[axis] = slice(self.n_max, self.n_max+1)
        TF_array_zp[tuple(slices)] = 0.5*TF_array_zp[tuple(slices)]
        slices[axis] = slice(self.n_min-1, None)
        TF_array_zp[tuple(slices)] = TF_array[tuple(slices)]
        slices[axis] = slice(self.n_min-1, self.n_min)
        TF_array_zp[tuple(slices)] = 0.5*TF_array_zp[tuple(slices)]
        array_zp = g_zp.ifft(TF_array_zp, axis=axis)
        if centered : array_zp = g_zp.sort2cent(array_zp, axis=axis)
        data_type = array.dtype.kind
        if data_type == 'c' : # Complex-valued array
            return (g_zp, array_zp)
        else : # Real-valued array
            Re_max,Im_max = (abs(array_zp.real).max(),
                             abs(array_zp.imag).max())
            if Im_max > 1e-8*Re_max :
                msg = ("Space1DGrid.zero_padding :: Warning: non neglictible "
                       + "imaginary part of the obtained array.")
                print(msg)
            return (g_zp, array_zp.real)
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
            raise ValueError("Error in 'Space1DGrid.Inum' staticmethod:\n\t"
                             + "all parameters have to be strictly positive.")
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
        R = 0.5 * ( ( np.log(piX/piY)
                      - np.einsum("i,j->ij", CiX, np.ones_like(CiY))
                      + np.einsum("i,j->ij", np.ones_like(CiX), CiY) )
                    * np.cos(piXmY)
                    - ( np.einsum("i,j->ij", SiX, np.ones_like(SiY))
                        + np.einsum("i,j->ij", np.ones_like(SiX), SiY)
                        - np.pi )
                    * np.sin(piXmY) ) / (np.pi*piXmY)
        # xi == yj
        piX_diag = piX[idx_diag]
        SiX_diag = SiX[idx_diag[0]]
        R[idx_diag] = ( ( np.sin(piX_diag)**2 / piX_diag - SiX_diag )
                        / np.pi + 0.5 )
        if X_is_value :
            if Y_is_value :
                return R[0, 0]
            else :
                return R[0, :]
        else :
            if Y_is_value :
                return R[:, 0]
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
            raise ValueError("Error in 'Space1DGrid.OptimizedBasis' "
                             + "static method:\n\tn and m have to be "
                             + "positive integers.")
        try :
            assert 0 <= sps <= n//2
            sps = round(sps)
        except :
            raise ValueError("Error in 'Space1DGrid.OptimizedBasis' "
                             + "static method:\n\tsharpest_peaks_at_sides "
                             + f"must be an integer in [0,{n//2}].")
        w = n+2*m
        B = np.zeros( (n,w) )
        if sps > 0 :
            peak, _, _, err = SG.OptimizedBasis(1, m)
            peak = peak[0]
            npeak = peak.shape[0]
            for i in range(sps) :
                B[i, i:i+npeak] = peak
                B[n-i-1, w-i-npeak:w-i] = peak
            n -= 2 * sps
        else :
            peak = np.array([])
            err = 0.0
        if n > 0 :
            a = np.arange(n)
            c = np.append( np.arange(-m, 0), np.arange(n, m+n) )
            Maa = SG.Inum(m+1+a, m+1+a) + SG.Inum(n+m-a, n+m-a)
            Mca = SG.Inum(c+m+1, a+m+1) + SG.Inum(n+m-c, n+m-a)
            Mcc = SG.Inum(c+m+1, c+m+1) + SG.Inum(n+m-c, n+m-c)
            A2C = solve(Mcc, -Mca)
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
    def OrthonormalBasis(n, m, sharpest_peaks_at_sides=0,
                         expand_from_center=False, verification=False) :
        """Returns an orthonormal basis (L2 scalar product )of size n defined
           on 2n+4m+1 evenly spaced values (step 1/2) for which the support
           of corresponding continuous-variable functions are included in a
           common interval of length (n+2m+1) symmetrically containing the
           2n+4m+1 values."""
        if expand_from_center and sharpest_peaks_at_sides>0 :
            msg = ("Space1DGrid.OrthonormalBasis :: Warning:\n\t"
                   + "'expand_from_center' and 'sharpest_peaks_at_sidesb>b0'"
                   + " cannot be both True.\n\t'sharpest_peaks_at_sides'"
                   + "set to 0.")
            print(msg)
            sharpest_peaks_at_sides=0
        SG = Space1DGrid
        if expand_from_center :
            if n==1 :
                return SG.OrthonormalBasis(1, m, sharpest_peaks_at_sides,
                                           verification= verification)
            else :
                vrf = False
            # else (n > 1)
            peak = SG.OrthonormalBasis(1, m, sharpest_peaks_at_sides)[0]
            r = n-1
            nL = r//2
            nR = r-nL
            nL *=2
            nR *=2
            new_e = np.concatenate([np.zeros(nL), peak, np.zeros(nR)])
            OB = np.array( [new_e] )
            for r in range(2,n+1):
                if r%2 == 0 :
                    new_e = np.concatenate([np.zeros(nL+r), peak,
                                            np.zeros(nR-r)])
                else :
                    new_e = np.concatenate([np.zeros(nL-r+1), peak,
                                            np.zeros(nR+r-1)])
                new_e -= 0.5*OB.T@OB@new_e
                new_e /= np.sqrt(0.5*new_e@new_e)
                OB = np.concatenate([OB, [new_e]])
            B_ortho = OB
        else :
            if n%2 == 0 : # Even n
                nL, nR = 1,1
                size = n + 2*m + 2
            else : # Odd n
                nL, nR = 1,2
                size = n + 2*m + 3
                local_grid = SG( size )
            B, _, _, err = SG.OptimizedBasis(n, m, sharpest_peaks_at_sides)
            B = np.concatenate([ np.zeros((n,nL)), B, np.zeros((n,nR)) ],
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
        np.set_printoptions(formatter={"complex_kind":complex_prt,
                                       "float_kind":float_prt})
        gt = Space1DGrid(8,0.5)
        print(gt)
        print("(sorted) space vector:", gt.space_values)
        print("centered space vector:", gt.Xc )
        cases,opt = (["[0,1,0,0,0,0,0,0]", "[0,0,0,0,1,0,0,0]"],
                     ["",",centered=True"])
        for c,o in zip(cases,opt) :
            test = "gt.fft(np.array(" + c + ")" + o + ")"
            tf = eval(test)
            print("tf =", test, "; tf:")
            print(tf)
            test2 = "gt.ifft(tf" + o + ")"
            print(test2, "->", eval(test2))
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
    if False :
        examples = [ f"Space1DGrid.Inum({n1},{n2})" for n1,n2 in (
            (1,1), (1,2), (1,[1,2]), ([1,2],1), ([1,2,3],[1,2,3])) ]
        for ex in examples :
            result = eval(ex).round(6)
            print(f"***** {ex} *****\n{result}")            
        for n,m,sps in ((1,2,0), (1,3,0), (2,3,0), (8,3,2)) :
            B, A2C, pk, err = Space1DGrid.OptimizedBasis(n, m, sps)
            sz,pts = B.shape
            print(f"***** Space1DGrid.OptimizedBasis({n}, {m}) *****"
                  + f"\nBasis of size {sz} defined on {pts} points"
                  + f"\n{A2C.round(3)}\nRelative Error ~ {err:.2e}.")
            if sps > 0 :
                print(f"B matrix:\n{B[:,:7].round(3)}\n{B[:,7:].round(3)}")
        for n,m,sps in ((1,2,0), (1,3,0), (2,3,0), (8,3,2), (20,4,2)) :
            EFC = False # this boolean can be change
            B = Space1DGrid.OrthonormalBasis(n, m, sps,
                                             expand_from_center=EFC,
                                             verification = True)
            sz,pts = B.shape
            print(f"***** Space1DGrid.OrthonormalBasis({n}, {m}) *****"
                  + f"\nBasis of size {sz} defined on {pts} points"
                  + f"\n{B.round(2)}")
###############################################################################
############################### SharpestPeak ##################################
###############################################################################
class SharpestPeak :
    """Sharpest Peak is a function that has only n (odd) non-zero sampled
       values (or n+1 if x_sym is not an integer) and that is negligible
       outside the interval [x_sym-(n+1)/2,x_sym-(n+1)/2].
       x_sym is in [-0.5,0.5] is the center of symmetry"""
    # Values of sharpest peaks (p[0]+2*sum(p[1:])==1 and
    #                           p[0]+2*(sum(p[2::2])-sum(p[1::2]))==0)
    # Numerically obtained by minimization
    __peak_values = { 5 : [0.408970, 0.250000, 0.045515],
                      7 : [0.355831124, 0.243709059, 0.072084438, 0.006290941],
                      9 : [0.31517885508, 0.23358796170, 0.09153369578,
                           0.01641203830, 8.7687668e-4],
                      11: [0.2882690819314, 0.2240275205647,
                           0.1028871692415, 0.0258761399535,
                           2.9782897928e-3, 9.63394818e-5] }
    __relative_errors = {5:2.5e-4, 7:7.7e-6, 9:5.1e-7, 11:1.8e-8}
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __init__(self, x_sym=0, n=5, n_max=128):        
        n = round(np.clip(n,5,11))
        if n%2==0 : n -= 1 # odd value
        # Even peak values
        val = SharpestPeak.__peak_values[n]
        even_peak_nzv = np.array(val[:0:-1]+val, dtype=np.float64)
        self.__short_grid = Space1DGrid(n+1)
        self.__FTvalues = self.__short_grid.fft(
                                np.append(even_peak_nzv,[0.0]),
                                centered=True)
        if np.isclose(x_sym, 0.0):
            self.__nzv = even_peak_nzv
            self.__nmin = -(n//2)
            self.__nmaxp1 = -self.__nmin+1
        elif -0.5 <= x_sym <= 0.5:
            self.__FTvalues *= np.exp(1j*x_sym*self.__short_grid.Kc)
            self.__nzv = self.__short_grid.ifft(self.__FTvalues,
                                                centered=True).real
            if x_sym >= 0.0 : 
                self.__nmin = -(n//2)
                self.__nmaxp1 = -self.__nmin+2
            else :
                self.__nzv = np.append(self.__nzv[-1:],self.__nzv[:-1])
                self.__nmin = -(n//2)-1
                self.__nmaxp1 = -self.__nmin
        else:
            raise ValueError("SharpestPeak constructor :: Error:"
                             + "\n\t'x_sym' parameter not in [-1/2,1/2]")
        grid = Space1DGrid(2*n_max)
        disc_values = np.zeros_like(grid.Xc)
        n0 = n_max-1
        disc_values[n0+self.__nmin:n0+self.__nmaxp1] = self.__nzv
        zpg,zpv = grid.zero_padding(disc_values, 63, centered=True)
        err = SharpestPeak.__relative_errors[n]
        test = (np.abs(zpv) >= 0.2*err)
        idx_deb, idx_fin = test.argmax(), test[::-1].argmax()
        self.__interp = interp1d(zpg.Xc[idx_deb:zpg.nx-idx_fin],
                                 zpv[idx_deb:zpg.nx-idx_fin])
        self.__xmin = zpg.Xc[idx_deb]
        self.__xmax = zpg.Xc[zpg.nx-idx_fin-1]
        FT = grid.fft(disc_values, centered=True)
        self.__FT = interp1d(grid.Kc, FT)
        self.__kmin, self.__kmax = grid.Kc[0], grid.Kc[-1]
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __call__(self, x) :
        xL,xR = self.__xmin, self.__xmax
        return self.__interp(np.clip(x,xL,xR))*((xL<=x)&(x<=xR))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def FT(self, k) :
        kL,kR = self.__kmin, self.__kmax
        return self.__FT(np.clip(k,kL,kR))*((kL<=k)&(k<=kR))
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    @property
    def non_zero_values(self) : return self.__nzv.copy()
    @property
    def begin_end(self) : return (self.__nmin,self.__nmaxp1)
    @property
    def frequency_points(self) :
        return self.__short_grid.Kc, self.__FTvalues.copy()
###############################################################################
if __name__ == "__main__" :
    # SharpestPeak Examples
    if False : # Change True/False to show example
        exp = True
        for Vxc,Vn in [([0,0,0,0], [5,7,9,11]),
                       ([-0.5,0.3,-0.4,0.5], [5,5,7,7])]:
            fig = plt.figure("Sharpest Peaks", figsize=(12,6))
            axT,axF = fig.subplots(1,2)
            fig.subplots_adjust(0.05,0.08,0.99,0.94,0.2,0.2)
            clrs = ["#F00000", "#008000", "#0000F0", "#C000C0"]
            Vx = np.linspace(-7.3,7.3,500)
            Vk = np.linspace(0,3.3,500)
            if exp :
                a = 0.52
                axT.plot(Vx, np.exp(-0.5*a*Vx**2)*np.sqrt(0.5*a/np.pi),
                         dashes=(1,1), color="#B08000", lw=2.5,
                         label=(r"$\sqrt{\frac{a}{2\pi}}\exp(-a\,x^2/2)$"
                                + f"\nwith a={a:.2f}"))
                axF.plot(Vk, np.exp(-0.5*Vk**2/a), dashes=(1,1),
                         color="#B08000", lw=2.5,
                         label=(r"$\exp\!\left(\frac{-k^2}{2a}\right)$"
                                + f" with a={a:.2f}"))
                exp=False
            for xc,n,clr in zip(Vxc,Vn,clrs):
                pk = SharpestPeak(xc,n)
                nzv = pk.non_zero_values
                nzx = np.arange(*pk.begin_end)
                axT.plot(Vx, pk(Vx), "-", color=clr,
                         label=f"n={n}, $x_c$={xc:.2f}")
                axT.plot(nzx, nzv, "o", color=clr)
                FTpk_val = pk.FT(Vk)
                axF.plot(Vk, FTpk_val.real, "--", color=clr)
                axF.plot(Vk, FTpk_val.imag, ":", color=clr)
                axF.plot(Vk, np.abs(FTpk_val), "-", color=clr,
                         label=f"n={n}, $x_c$={xc:.2f}")
                K,FT = pk.frequency_points
                idx = np.argmin(np.abs(K))
                axF.plot(K[idx:],FT[idx:].real, "d", color=clr)
                axF.plot(K[idx:],FT[idx:].imag, "*", color=clr)
                axF.plot(K[idx:],np.abs(FT[idx:]), "o", color=clr)
            axT.grid() ; axT.legend()
            axT.set_xlabel("Position $x$", size=12)
            axT.set_ylabel("Sharpest Peak $p_n(x)$", size=12)
            axT.set_xlim(-6.2,6.2)
            axF.grid() ; axF.legend()
            axF.set_xlabel("Wavenumber $k$", size=12)
            fticks = ["$0$", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"]
            axF.set_xticks(np.linspace(0,np.pi,5),fticks)
            axF.set_ylabel(r"Fourier Transform $\tilde{p}_n(x)$", size=12)
            axF.set_title(r"Abs. val. ($-$, $\bullet$), real (--, $\diamond$)"
                          + r" and imaginary ($\cdots$, $*$) parts")
            axF.set_xlim(-0.1,3.3)
            plt.show()    
###############################################################################
###################### Space1DGrid_with_Subinterval ###########################
###############################################################################
class Space1DGrid_with_Subinterval(Space1DGrid):
    """A 1D grid in space, with a subinterval [a,b] included in [x_min,x_max]
       'nb' is the (even) number of discretization points.
       'step' is the discretization step 'dx' in space.
       x_min = (-nb//2+1)*dx and x_max = (nb//2)*dx.
       x_min+5*dx < a < b < x_max-5*dx.
       The size 'dim' of the sharpest-peak basis is the number of grid points 
       in the interval [a,b]. If 'adjust', the centers of the peaks, evenly
       spaced, range from a to b. Otherwise, they remain the points of the
       grid. The parameter 'sharpest_peak_number' is in [5,7,9,11].
    """
    #--------------------------------------------------------------------------
    def __init__(self, nb, dx, a, b, sharpest_peak_number=5, adjust=True,
                 verbose=False) :
        Space1DGrid.__init__(self, nb, dx, verbose)
        msg = "Space1DGrid_with_Subinterval constructor :: "
        vmin,vmax = self.Xc[4], self.Xc[-5]
        if not vmin <= a <= vmax :
            msg += f"Error:\n\t\ta ~ {a:.2e} not in [{vmin:.2e},{vmax:.2e}]."
            raise ValueError(msg)
        if not vmin <= b <= vmax :
            msg += f"Error:\n\t\tb ~ {b:.2e} not in [{vmin:.2e},{vmax:.2e}]."
            raise ValueError(msg)
        if verbose :
            self.__prt = print
        else :
            self.__prt = lambda *args : None
        prt = self.__prt
        idx_inf = np.argmax(self.Xc >= a)
        nb_pts = np.argmax(self.Xc[idx_inf:] > b)
        if nb_pts < 2 :
            msg += (f"Error:\n\t\t{nb_pts} point only in [a,b] ~ "
                    + f"[{a:.2e},{b:.2e}].")
            raise ValueError(msg)
        # Global indexes of grid values in the interval [a,b]
        self.__a, self.__b = a,b
        self.__idx_inf = idx_inf
        self.__idx_sup = idx_inf + nb_pts
        self.__dim = nb_pts
        prt(f"Number of points in [a,b] ~ [{a:.2e},{b:.2e}]: {nb_pts}")
        self.__adj = adjust
        if adjust :
            self.__vect_xp = np.linspace(a, b, nb_pts)
        else :
            self.__vect_xp = self.Xc[idx_inf:self.__idx_sup]
        sharpest_peaks = []
        self.__P = []
        zeros = np.zeros_like(self.Xc)
        i_min, i_max = self.__idx_sup, self.__idx_inf
        for xp in self.__vect_xp :
            idx = np.argmin(np.abs(self.Xc-xp))
            sh_peak = SharpestPeak((xp-self.Xc[idx])/self.dx,
                                   sharpest_peak_number)
            values = sh_peak.non_zero_values
            i_beg, i_end = np.array(sh_peak.begin_end) + idx
            sharpest_peaks.append( (i_beg, i_end) )
            i_min, i_max = min(i_beg,i_min), max(i_end,i_max)
            zer = zeros.copy()
            zer[i_beg:i_end] = values
            self.__P.append( zer )
        self.__P = np.array(self.__P).transpose()[i_min:i_max]
        # Global indexes of non-zero values around [a,b]
        self.__imin, self.__imax = i_min, i_max
        # Relative indexes of grid values in the interval [a,b]
        self.__idxL, self.__idxR = self.__idx_inf-i_min,self.__idx_sup-i_min
        self.__sharpest_peaks_idx = [((i_beg, i_end),
                                      (i_beg-i_min, i_end-i_min))
                                     for (i_beg, i_end) in sharpest_peaks]
        self.__M = np.linalg.inv(self.__P[self.__idxL:self.__idxR])
        self.__sh_pk_nb = sharpest_peak_number
    #--------------------------------------------------------------------------
    @property
    def basis_dim(self): return self.__dim
    @property
    def idx_min(self): 
        "Index of the first x non-zero values around the interval [a,b]."
        return self.__imin
    @property
    def idx_max(self): 
        """Index of the first x after the non-zero values around the
           interval [a,b]."""
        return self.__imax
    @property
    def idx_left(self):
        "Index of the first x value in the interval [a,b]."
        return self.__idx_inf
    @property
    def idx_right(self): 
        "Index of the first x value after the interval [a,b]."
        return self.__idx_sup
    @property
    def sharpest_peaks(self) :
        return [(c, self.__P[b:e,r].tolist())
                for r,(c,(b,e)) in enumerate(self.__sharpest_peaks_idx) ]
    @property
    def sharpest_peak_number(self): return self.__sh_pk_nb
    @property
    def adjusted(self): return self.__adj
    @property
    def interval_ab(self): return self.__a, self.__b
    #--------------------------------------------------------------------------
    def val_to_coef(self, values_in_ab) :
        """The dimension of the vector 'values_in_ab' is the number n of  
           grid points in the interval [a,b], which is also equal to the
           size of the basis B of sharpest peaks. Returns the coefficient
           vector of the interpolation function in the basis B."""
        return self.__M@values_in_ab
    #--------------------------------------------------------------------------
    def coef_to_val(self, coef_in_B) :
        """Returns the non-zero values on the grid of the interpolation 
           function of coefficients in the basis B given by the vector
           'coef_in_B'."""
        return self.__P@coef_in_B
    #--------------------------------------------------------------------------
    def val_to_func(self, values_in_ab):
        """The dimension of the vector 'values_in_ab' is the number n of  
           grid points in the interval [a,b], which is also equal to the
           size of the basis B of sharpest peaks. Returns the interpolation
           function in the basis B."""
        all_nonzero_values = self.__P@self.__M@values_in_ab
        return self.__all_nonzero_values_to_func(all_nonzero_values)
    #--------------------------------------------------------------------------
    def coef_to_func(self, coef_in_B):
        """Returns the interpolation function of coefficients in the basis B
           given by the vector 'coef_in_B'."""
        all_nonzero_values =  self.__P@coef_in_B
        return self.__all_nonzero_values_to_func(all_nonzero_values)
    #--------------------------------------------------------------------------
    def __all_nonzero_values_to_func(self, all_nonzero_values,
                                     nb_margin_steps=10, zero_padding_coef=63,
                                     neglictible=1e-8):
        """Returns the interpolation function built by zero-padding from the
           values on the grid in the [idx_min:idx_max] index range."""
        nb_nz_val = self.__imax - self.__imin
        if not isinstance(all_nonzero_values, np.ndarray) :
            all_nonzero_values = np.array(all_nonzero_values)
        assert all_nonzero_values.shape == (nb_nz_val,)
        if nb_nz_val%2 == 1 : 
            nb_val = nb_nz_val + 1 + 2*nb_margin_steps
        else :
            nb_val = nb_nz_val + 2*nb_margin_steps
        xgd = Space1DGrid(nb_val)
        values = np.zeros( nb_val, dtype=all_nonzero_values.dtype )
        values[nb_margin_steps:nb_margin_steps+nb_nz_val] = all_nonzero_values
        zpdg, zpval = xgd.zero_padding(values, zero_padding_coef,
                                       centered=True)
        idx0 = nb_val//2 - 1 - nb_margin_steps
        X0,dX = self.Xc[self.__imin + idx0], self.dx
        positions = X0 + zpdg.Xc*self.dx
        abs_vals = np.abs(zpval)
        ampli_max = abs_vals.max()
        non_zero_vals = (abs_vals >= neglictible*ampli_max)
        idx_deb = non_zero_vals.argmax()
        idx_fin = non_zero_vals.shape[0] - non_zero_vals[::-1].argmax()
        positions = positions[idx_deb:idx_fin]
        zpval = zpval[idx_deb:idx_fin]
        itrp_fct = interp1d(positions, zpval)
        def interp_func(x, x_min=positions[0], x_max=positions[-1],
                        ifunc=itrp_fct):
            return np.where( (x_min<=x)&(x<=x_max),
                             ifunc(np.clip(x, x_min, x_max)), 0.0 )
        return interp_func    
    #--------------------------------------------------------------------------
    def orthonormal_basis(self, from_or_indexes):
        """Creates an orthonormal basis for the sharpest-peak basis by using
           the classical Gram-Schmidt Process applied following the order
           given by 'from_or_indexes': either str in ("left", "right", "sides",
           "center") or list of numbers from 1 to the basis dimension.
           Returns the rectangular matrix B representing the orthonormal 
           basis with the square matrix C, such that B = P@C."""
        if isinstance(from_or_indexes, str) :
            indexes = np.arange(self.__dim).tolist()
            if from_or_indexes.lower() == "left" :
                pass
            elif from_or_indexes.lower() == "right" :
                indexes = indexes[::-1]
            elif from_or_indexes.lower() == "sides" :
                indexes = np.array(list(zip(indexes,indexes[::-1])))
                indexes = (indexes.flatten().tolist())[:self.__dim]
            elif from_or_indexes.lower() == "center" :
                d = (self.__dim-1)//2
                indexes = np.array(list(zip(indexes[d::-1],indexes[d+1::])))
                indexes = (indexes.flatten().tolist())
                if self.__dim%2 == 1 : indexes.append(0)
            else :
                msg = ("Space1DGrid_with_Subinterval::orthonormal_basis"
                       + f"\n\tError: '{from_or_indexes}' unknown.")
                raise ValueError(msg)
        else :
            indexes = [i-1 for i in from_or_indexes]
        return self.__orthonormal_basis(indexes)            
    #--------------------------------------------------------------------------
    def __orthonormal_basis(self, indexes) :
        """Creates an orthonormal basis for the sharpest-peak basis by using
           the classical Gram-Schmidt Process applied following the order
           given by the list 'indexes' of numbers from 0 to n-1, where n is
           the basis dimension.
           Returns the rectangular matrix B representing the orthonormal 
           basis with the square matrix C, such that B = P@C."""
        # Array of sharpest peaks including renumbering
        BT = (self.__P.transpose())[indexes].copy()
        # Matrix of coefficients
        CT = np.zeros( (self.__dim,self.__dim) )
        bi,j = BT[0], indexes[0]
        c = 1.0 / np.sqrt(self.dx * bi@bi)            
        CT[0,j] = c
        BT[0] *= c
        for i,(bi,j) in enumerate(zip(BT[1:],indexes[1:]),1) :
            # Orthogonalization:
            S = -self.dx * BT[:i]@bi 
            bi += S@BT[:i]
            # Normalization
            c = 1.0 / np.sqrt(self.dx * bi@bi)
            bi *= c
            # Update of B
            BT[i] = bi 
            # Update of C        
            CT[i,j] = c
            CT[i, indexes[:i]] = c*S@CT[:i, indexes[:i]]
        return BT.T, CT.T            
        
###############################################################################
if __name__ == "__main__" :
    # Space1DGrid_with_Subinterval example
    # A/ Interpolation of basic functions in the sharpest-peak basis
    interpol = False  # Change True/False to show example
    # B/ Orthonormal bases
    orthonorm = False # Change True/False to show example    
    # C/ Interpolation function
    interp_example = True # Change True/False to show example
    if interpol or orthonorm or interp_example : 
        colors = ('#2080B0', '#FF8000', '#30A030', '#A82828', '#A068C0',
                  '#905850', '#E078C0', '#808080', '#C0C020', '#18C0D0',
                  '#0000F0', '#CC0000', '#007000', '#808000', '#00A0A0',
                  '#FF00FF', '#C05000', '#60A060', '#6060A0', '#A06060')*5
        val_a, val_b = -0.34, 0.82
        spgd_ws = Space1DGrid_with_Subinterval(32, 0.1, val_a, val_b,
                                               sharpest_peak_number=7,
                                               adjust=True)
        B,E = spgd_ws.idx_min, spgd_ws.idx_max
        if spgd_ws.sharpest_peak_number == 5 :
            h = 0.43 ; h2 = 2.9
        elif spgd_ws.sharpest_peak_number == 7 :
            h = 0.37 ; h2 = 2.6
        title = (str(spgd_ws.basis_dim) + " Sharpest Peaks of type "
                 + str(spgd_ws.sharpest_peak_number))
        if spgd_ws.adjusted : title += " adjusted on"
        else : title += " not adjusted on"
        title += f" interval [{val_a:.2f}, {val_b:.2f}]"
        if interpol:
            for func, name in ([
                (np.ones_like,r"$\mathbb{1}_{[a,b]}(x)$"),
                (lambda x,a=val_a,b=val_b:2*(x-a)/(b-a)-1,
                 r"$\frac{2x-a-b}{b-a}\,\mathbb{1}_{[a,b]}(x)$"),
                (lambda x,a=val_a,b=val_b:np.sin(np.pi*(x-a)/(b-a)),
                 (r"$\sin\!\left(\pi\;\frac{x-a}{b-a}\right)\,"
                  + r"\mathbb{1}_{[a,b]}(x)$"))]
                + [(lambda x,a=val_a,b=val_b,j=j:np.sin(j*np.pi*(x-a)/(b-a)),
                    (r"$\sin\!\left("+str(j)+r"\,\pi\;\frac{x-a}{b-a}\right)\,"
                     + r"\mathbb{1}_{[a,b]}(x)$")) for j in range(2,9)]) :
                fig = plt.figure("Basis of sharpest Peaks", figsize=(14,7))
                axB, axC = fig.subplots(2,1)
                fig.subplots_adjust(0.05,0.08,0.99,0.94,0.2,0.1)
                # Sharpest peaks of the basis
                for i,(((b,e),v),clr) in enumerate(
                    zip(spgd_ws.sharpest_peaks,colors), 1) :
                    all_val = np.zeros_like(spgd_ws.Xc)
                    all_val[b:e] = v
                    zp_gd, val_c = spgd_ws.zero_padding(all_val, 20,
                                                        centered=True)
                    axB.plot(zp_gd.Xc, val_c, "-", color=clr)
                    axB.plot(spgd_ws.Xc[b:e], v, "o", color=clr,
                            label="$b_{"+str(i)+"}$")
                axB.plot([val_a, val_b], [h, h], "-", color="#808080", lw=4.0)
                axB.grid() ; axB.legend(fontsize=8) ; axB.set_xlim(-1.6,1.6)
                axB.set_ylabel("Basis Functions",size=14, weight="bold")
                axB.set_title(title, size=14, weight="bold")
                # Example of interpolation
                XG = spgd_ws.Xc[spgd_ws.idx_left:spgd_ws.idx_right]
                val_test = func( XG )
                coef_test = spgd_ws.val_to_coef(val_test)
                grid_val = spgd_ws.coef_to_val(coef_test)
                XNZ = spgd_ws.Xc[spgd_ws.idx_min:spgd_ws.idx_max]
                all_val = np.zeros_like(spgd_ws.Xc)
                all_val[spgd_ws.idx_min:spgd_ws.idx_max] = grid_val
                zp_gd, val_f = spgd_ws.zero_padding(all_val, 20, centered=True)
                X = zp_gd.Xc
                axC.plot(X,func(np.clip(X,val_a, val_b))*(X>=val_a)*(X<=val_b),
                         ":m", label=name)
                axC.plot(zp_gd.Xc, val_f, "-b", label="Interpolation function")
                axC.plot( XG, val_test, "om", label="grid values in [a,b]",
                          markersize=8.0)
                axC.plot( XNZ, grid_val, "dc", label="non-zero grid values",
                          markersize=4.0)
                
                axC.grid() ; axC.legend(fontsize=8) ; axC.set_xlim(-1.6,1.6)
                axC.set_xlabel("Position $x$",size=14, weight="bold")
                axC.set_ylabel("Interpolation",size=14, weight="bold")
                y0,y1 = axC.get_ylim()
                yrange = np.linspace(y1,y0,spgd_ws.basis_dim+4)[2:-2]
                for i,(c,y,clr) in enumerate(zip(coef_test, yrange, colors),1) :
                    axC.plot(-1.5, y, "o", color=clr)
                    axC.text(-1.45, y, "$c_{"+str(i)+"}$ ~ "+f"{c:.4f}",
                             verticalalignment="center")
                plt.show()
        if orthonorm :
            for test in ("left","right","sides","center",
                         [8,9,10,11,12,1,2,3,4,5,6,7]):
                fig = plt.figure("Basis of sharpest Peaks", figsize=(14,7))
                axB, axC = fig.subplots(2,1)
                fig.subplots_adjust(0.05,0.08,0.99,0.94,0.2,0.25)
                # Sharpest peaks of the basis
                for i,(((b,e),v),clr) in enumerate(
                    zip(spgd_ws.sharpest_peaks,colors), 1) :
                    all_val = np.zeros_like(spgd_ws.Xc)
                    all_val[b:e] = v
                    zp_gd, val_c = spgd_ws.zero_padding(all_val, 20,
                                                        centered=True)
                    axB.plot(zp_gd.Xc, val_c, "-", color=clr)
                    axB.plot(spgd_ws.Xc[b:e], v, "o", color=clr,
                            label="$b_{"+str(i)+"}$")
                axB.plot([val_a, val_b], [h, h], "-", color="#808080", lw=4.0)
                axB.grid() ; axB.legend(fontsize=8, loc="upper right")
                axB.set_xlim(-0.82,1.52)
                axB.set_ylabel("Basis Functions", size=14, weight="bold")
                axB.set_title(title, size=14, weight="bold")
                # Example of orthonormal basis
                OB,_ = spgd_ws.orthonormal_basis(test)
                for i,(v,clr) in enumerate(zip(OB.T, colors), 1):
                    all_val = np.zeros_like(spgd_ws.Xc)
                    all_val[B:E] = v
                    zp_gd, val_c = spgd_ws.zero_padding(all_val, 20,
                                                        centered=True)
                    axC.plot(zp_gd.Xc, val_c, "-", color=clr)
                    axC.plot(spgd_ws.Xc[B:E], v, "o", color=clr,
                            label="$b_{"+str(i)+"}^*$")
                axC.plot([val_a, val_b], [h2, h2], "-",
                         color="#808080", lw=4.0)
                axC.grid() ; axC.legend(fontsize=7, loc="upper right")
                axC.set_xlim(-0.82,1.52)
                axC.set_ylabel("Orthonormal basis", size=14, weight="bold")
                axC.set_title(f"Orthonormal basis for '{test}'",
                              size=14, weight="bold")
                axC.set_xlabel("Position $x$",size=14, weight="bold")
                plt.show()
        if interp_example :
            fig = plt.figure("Interpolation example", figsize=(14,7))
            fig.subplots_adjust(0.05,0.08,0.99,0.94,0.2,0.25)
            ax = fig.subplots(1,1)
            val_test = 0.3*np.random.randn(spgd_ws.basis_dim) + 1.0
            itrp_fct = spgd_ws.val_to_func(val_test)
            Vx = np.linspace(spgd_ws.g_range[0], spgd_ws.g_range[1], 1000)
            Vy = itrp_fct(Vx)
            y_min,y_max = Vy.min(), Vy.max()
            coef_test = 2.0*np.random.randn(spgd_ws.basis_dim) - 1.0
            itrp_fct2 = spgd_ws.coef_to_func(coef_test)
            Vy2 = itrp_fct2(Vx)
            y_min,y_max = min(y_min, Vy2.min()), max(y_max, Vy2.max())
            y_min,y_max = 1.05*y_min-0.05*y_max, 1.05*y_max-0.05*y_min
            for x in spgd_ws.Xc :
                plt.vlines(x, y_min,y_max, "b", linewidth=0.5)
            ax.plot(spgd_ws.Xc, np.zeros_like(spgd_ws.Xc), ".b",
                    markersize=3.0)
            ax.plot(Vx, Vy, "-g", label="#1 Interpolation function")
            ax.plot(spgd_ws.Xc[spgd_ws.idx_min:spgd_ws.idx_max],
                    spgd_ws.coef_to_val(spgd_ws.val_to_coef(val_test)), ".r")
            ax.plot(spgd_ws.Xc[spgd_ws.idx_left:spgd_ws.idx_right],
                    val_test, "or", label=r"#1 Random values in $\Gamma$")
            ax.plot(Vx, Vy2, "-b", label="#2 Interpolation function")
            ax.plot(spgd_ws.Xc[spgd_ws.idx_min:spgd_ws.idx_max],
                    spgd_ws.coef_to_val(coef_test), "dm",
                    label="#2 Random coefficients on sharpest-peak basis")
            ax.set_xlim(*spgd_ws.g_range)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel("Position $x$",size=14, weight="bold")
            ax.plot([val_a, val_b], 2*[0.97*y_min+0.03*y_max],
                    "-", color="#808080", lw=4.0)
                    
            ax.legend() ; ax.grid()
            plt.show()
###############################################################################
############################# Space2DGrid #####################################
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
        dico_opt = {"interpolation":"none", "extent":self.g_range,
                    "cmap":"seismic", "origin":"lower"} # Default options
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
        r"""Numerical Hankel (or Fourier-Bessel) Transform.
            Numerical approximation of
               \int_{0}^{r_{max}} f(r) J_{order}(k r) r dr."""
        self.__check(array, axis) 
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def iNHT(self, order, array, axis) :
        r"""Numerical Inverse Hankel (or Fourier-Bessel) Transform.
            Numerical approximation of
               \int_{0}^{k_{max}} f(k) J_{order}(k r) k dk."""
        self.__check(array, axis)
###############################################################################
if __name__ == "__main__" :
    # CylSymGrid example
    if False :  # Change True/False to show example
        cs_gd = CylSymGrid( 11, 0.1 )
        print(cs_gd)
