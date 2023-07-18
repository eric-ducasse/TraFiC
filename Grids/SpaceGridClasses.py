# Version 1.52 - 2023, July, 18
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
###############################################################################
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft,ifft,fft2,ifft2
###############################################################################
class Space1DGrid :
    """A 1D grid in space, with the corresponding grid in the wavenumbers 
       domain, and the associated FFT tools."""
    #--------------------------------------------------------------------------
    def __init__(self,nb,step=1.0) :
        """'nb' is the number of discretization points.
           'step' is the discretization step in space."""
        if nb%2 == 1 :
            nb -= 1
            print("The number of points as to be even:",nb,"considered")
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
        """Discretization step in wavenumbers domain."""
        return np.pi/self.v_max
    @property
    def k_max(self):
        """Maximum value in wavenumbers domain."""
        return self.dk*self.n_max
    @property
    def k_min(self):
        """Minimum value in wavenumbers domain."""
        return self.dk*self.n_min
    @property
    def wavenumber_values(self) :
        """Wavenumber values in the "sorted" representation."""
        return self.dk*self.numbers
    @property
    def K(self) :
        """ = self.wavenumber_values."""
        return self.wavenumber_values
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def __check(self, array, axis) :
        if array.shape[axis] != self.nb :
            msg = ("Space1DGrid :: Incompatible shape! {} is required "+\
                   "instead of {}").format(self.nb,array.shape[axis])
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
        return fmt.format(self.nb,self.v_min,self.v_max,self.step)
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
###############################################################################
if __name__ == "__main__" :
    float_prt = lambda x : "{:.3f}".format(x)
    def complex_prt(z) :
        a,b = z.real,z.imag
        if b >= 0 : return "{:.3f}+{:.3f}j".format(a,b)
        else : return "{:.3f}{:.3f}j".format(a,b)
    np.set_printoptions(formatter={"complex_kind":complex_prt,\
                                   "float_kind":float_prt})
    gt = Space1DGrid(8,0.5)
    print(gt)
    print("(sorted) space vector:",gt.space_values)
    print("centered space vector:",gt.sort2cent(gt.space_values))
    cases,opt = ["[0,1,0,0,0,0,0,0]","[0,0,0,0,1,0,0,0]"],\
                ["",",centered=True"]
    for c,o in zip(cases,opt) :
        test = "gt.fft(np.array("+c+")"+o+")"
        tf = eval(test)
        print("tf =",test,"; tf:")
        print(tf)
        test2 = "gt.ifft(tf"+o+")"
        print(test2,"->",eval(test2))
    print("sorted wavenumbers:",gt.wavenumber_values)
    print("centered wavenumbers:",gt.sort2cent(gt.wavenumber_values))
    A = np.array([[0,1,0,0,0,0,0,0],[0,0,0.22,0.78,1,0.78,0.22,0]])
    gzp,Azp = gt.zero_padding(A,5,axis=1,centered=True)
    plt.figure("Zero-padding 1D")
    plt.plot(gzp.sort2cent(gzp.space_values),Azp[0].real,".r")
    plt.plot(gzp.sort2cent(gzp.space_values),Azp[1].real,".m")
    plt.plot(gt.sort2cent(gt.space_values),A[0].real,"ob")
    plt.plot(gt.sort2cent(gt.space_values),A[1].real,"sg")
    plt.show()
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
        return fmt.format(self.ny,self.nx,self.__grid_x.v_min,\
                          self.__grid_x.v_max,self.__grid_y.v_min,\
                          self.__grid_y.v_max,self.dx,self.dy)
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
            plt.figure(fig_name,figsize=(0.6*Lx_over_Ly*height,height))   
            v_max = max(abs(array.real).max(),abs(array.imag).max())
            dico_opt["vmin"] = -v_max ; dico_opt["vmax"] = v_max
            for k in dico_opt.keys() :
                if k in kwargs.keys() : dico_opt[k]=kwargs[k]
            if clrbar :
                if "shrink" in kwargs.keys() :
                    dico_cb = {"shrink":kwargs["shrink"]}
                else : dico_cb = {"shrink":0.6}
            ax_re = plt.subplot(2,1,1)
            ax_re.set_title("Real Part")
            img_re = ax_re.imshow(array.real,**dico_opt)
            if clrbar : plt.colorbar(img_re,ax=ax_re,**dico_cb)
            ax_re.set_xlim(xmin,xmax);ax_re.set_ylim(ymin,ymax)
            ax_im = plt.subplot(2,1,2)
            ax_im.set_title("Imaginary Part")
            img_im = ax_im.imshow(array.imag,**dico_opt)
            if clrbar : plt.colorbar(img_im,ax=ax_im,**dico_cb)
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
            if clrbar :
                if "shrink" in kwargs.keys() :
                    dico_cb = {"shrink":kwargs["shrink"]}
                else : dico_cb = {"shrink":0.8}                
            if draw_axis is None : # new figure
                plt.figure(fig_name,figsize=(1.2*Lx_over_Ly*height,height))
                draw_axis = plt.subplot(1,1,1)
            img = draw_axis.imshow(array,**dico_opt)
            if clrbar : plt.colorbar(img,ax=draw_axis,**dico_cb)
            draw_axis.set_xlim(xmin,xmax);draw_axis.set_ylim(ymin,ymax)
            if draw_axis is None : 
                plt.subplots_adjust(left=0.05,right=0.99,\
                                    bottom=0.05,top=0.98)
            if show : plt.show()
            else : return draw_axis
        
###############################################################################
if __name__ == "__main__" :
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
    g2dt.plot( 2*abs(g2dt.fft(field)), height=5, fig_name="2D FFT", show=True)
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
