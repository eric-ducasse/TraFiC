# Version 1.27 - 2025, June 11
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution :  I2M / Arts & Metiers ParisTech
# Program name : TraFiC (Transient Field Computation)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import numpy as np
import matplotlib.pyplot as plt
outer = np.multiply.outer
import scipy.sparse as sprs
from scipy.io import savemat
from scipy.linalg import block_diag
import sys, os, gc, warnings
from PyQt5.QtWidgets import (QApplication, QWidget, QPushButton,
                             QHBoxLayout, QVBoxLayout, QLabel)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
if __name__ == "__main__" :
    cwd = os.getcwd()
    while "TraFiC" not in os.path.basename(cwd) :
        cwd = os.path.dirname(cwd)
    sys.path.append(cwd)
    from TraFiC_init import root
from Mode_Frames import Mode_Shape_Figure
from MaterialClasses import *
from USMaterialClasses import USMaterial
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Plane_Guided_Mode :
    """Guided mode in an immersed multilayer plate."""
    __unnamed_modes = 0
    __verbose = __name__ == "__main__"
    __Umax = 1e-4  # Max of displacement for normalization with respect
                   # to the x-component of the Poynting vector
    __Umean = 1e-6 # Mean value of 1 µm for normalization with respect to
                   # displacement
    __current_app = None # QApplication is running ?
    __UNDEF = "undefined" # Flag for undefined group velocity
    #---------------------------------------------------------------------
    def __init__(self, plate, f, k, mode, nu=0.0, name=None,
                 normalized=True, group_velocity=None) :
        """plate is an instance of DiscretizedMultilayerPlate, with
           n layers. f is the nonzero frequency [Hz], k wavenumber in the
           x-direction [mm^-1], nu the wavenumber in the y-direction
           [mm^-1].
           mode contains (n+2) pairs (Kz,Cz), where Kz denotes the
           vector of vertical wavenumbers and where Cz is the matrix of
           amplitudes (displacements for solids/velocity potential for
           fluids). mode[i] corresponds to layer #i. mode[-2] to the
           upper fluid half-space. mode[-1] to the lower fluid half-space.
        """
        self.__f = f
        self.__k = k
        self.__nu = nu
        self.__v_phi = 2*np.pi*f.real/k.real # In the x-direction
        self.__v_gr = None
        self.__setVgr(group_velocity)
        z, self.__Z = 0.0,[0.0]
        for lay in plate.layers :
            z += lay.thickness
            self.__Z.append(z)
        self.__Z = np.array(self.__Z)
        if name is None :
            self.__name = f"#{Plane_Guided_Mode.__unnamed_modes}"
            Plane_Guided_Mode.__unnamed_modes += 1
        else :
            self.__name = name
        self.__plate = plate
        self.__mode_data = mode
        self.__dico_fields = dict()
        self.__f_ux = plate.shape_function("Ux", k, f, mode, nu)
        self.__f_uy = plate.shape_function("Uy", k, f, mode, nu)
        self.__f_uz = plate.shape_function("Uz", k, f, mode, nu)
        # Phase adjustment
        dz = (self.__Z[-1]-self.__Z[0])*1e-6
        Vz = np.linspace(self.__Z[0]+dz, self.__Z[-1]-dz, 200)
        Ux,Uy,Uz = self.__f_ux(Vz), self.__f_uy(Vz), self.__f_uz(Vz)
        ix,iy,iz = (np.abs(U).argmax() for U in (Ux,Uy,Uz))
        no = np.argmax( (np.abs(U).max() for U in (Ux,Uy,Uz)) )
        if no == 0 :
            angle = np.angle(Ux[ix])
        elif no == 1 :
            angle = np.angle(Uy[iy])
        else : # no == 2
            angle = np.angle(Uz[iz])
        cf = np.exp(-1j*angle)
        self.__update_shape_functions(cf)
        self.__dico_fields["Px"] = self.Px
        self.__dico_fields["Py"] = self.Py
        self.__dico_fields["Pz"] = self.Pz 
        self.__dico_fields["Re(Px)"] = lambda z: self.Px(z,False)
        self.__dico_fields["Re(Py)"] = lambda z: self.Py(z,False)
        self.__dico_fields["Re(Pz)"] = lambda z: self.Pz(z,False)
        # Energy velocity
        self.__v_en = None
        self.__zmin = None
        self.__zmax = None
        self.__decreasing_amplitude_on_lhs = None
        self.__decreasing_amplitude_on_rhs = None
        self.__true_guided_mode = None
        self.__update_Energy_Velocity()
        # Normalization
        if normalized : self.normalize()
    #---------------------------------------------------------------------
    def __str__(self) :
        txt = f"Mode {self.__name}: "
        nb = len(txt)
        txt += ( f"phase velocity {self.Vphi:.3f} mm/µs; "
                 + f"Frequency {self.f:.4f} MHz;\n" + nb*" "
                 + f"Wavenumumber {self.k:.4f} mm^-1" )
        if self.is_a_true_guided_mode :
            txt += f"; Energy velocity {self.Ve:.3f} mm/µs."
        else :
            txt += " (not true guided mode)"
        return txt
    #---------------------------------------------------------------------
    def keys(self) :
        return tuple(self.__dico_fields.keys())
    #---------------------------------------------------------------------
    def __getitem__(self, key) :
        keys = self.keys()
        if key in keys :
            return self.__dico_fields[key]
        else :
            msg = f"Plane_Guides_Mode: the field '{key}' is not in {keys}"
            warnings.warn(msg)
            return np.zeros_like            
    #---------------------------------------------------------------------
    def __update_shape_functions(self, coefficient=None) :
        k,f,nu = self.__k,self.__f,self.__nu
        plate, mode = self.__plate, self.__mode_data
        if coefficient is not None :
            mode = ( [ (Kz,coefficient*C,Er) for Kz,C,Er in mode[:-2] ]
                     + [ (Kz,coefficient*C) for Kz,C in mode[-2:] ] )
            self.__mode_data = mode
        self.__f_ux = plate.shape_function("Ux", k, f, mode, nu)
        self.__dico_fields["Ux"] = self.__f_ux
        self.__f_uy = plate.shape_function("Uy", k, f, mode, nu)
        self.__dico_fields["Uy"] = self.__f_uy
        self.__f_uz = plate.shape_function("Uz", k, f, mode, nu)
        self.__dico_fields["Uz"] = self.__f_uz
        self.__f_sxx = plate.shape_function("Sxx", k, f, mode, nu)
        self.__dico_fields["Sxx"] = self.__f_sxx
        self.__f_sxy = plate.shape_function("Sxy", k, f, mode, nu)
        self.__dico_fields["Sxy"] = self.__f_sxy
        self.__f_sxz = plate.shape_function("Sxz", k, f, mode, nu)
        self.__dico_fields["Sxz"] = self.__f_sxz
        self.__f_syy = plate.shape_function("Syy", k, f, mode, nu)
        self.__dico_fields["Syy"] = self.__f_syy
        self.__f_syz = plate.shape_function("Syz", k, f, mode, nu)
        self.__dico_fields["Syz"] = self.__f_syz
        self.__f_szz = plate.shape_function("Szz", k, f, mode, nu)
        self.__dico_fields["Szz"] = self.__f_szz
        self.__f_dux = plate.shape_function("dUx/dz", k, f, mode, nu)
        self.__f_duy = plate.shape_function("dUy/dz", k, f, mode, nu)
        self.__f_duz = plate.shape_function("dUz/dz", k, f, mode, nu)
        self.__f_dsxz = plate.shape_function("dSxz/dz", k, f, mode, nu)
        self.__f_dsyz = plate.shape_function("dSyz/dz", k, f, mode, nu)
        self.__f_dszz = plate.shape_function("dSzz/dz", k, f, mode, nu)
        self.__dico_fields["dUx/dz"] = self.__f_dux
        self.__dico_fields["dUy/dz"] = self.__f_duy
        self.__dico_fields["dUz/dz"] = self.__f_duz
        self.__dico_fields["dSxz/dz"] = self.__f_dsxz
        self.__dico_fields["dSyz/dz"] = self.__f_dsyz
        self.__dico_fields["dSzz/dz"] = self.__f_dszz
        self.__dico_fields["e_tot"] = self.e_tot
        self.__dico_fields["Ve loc"] = self.Ve_loc
    #---------------------------------------------------------------------
    @property
    def f(self) :
        """Returns frequency in MHz."""
        return 1e-6*self.__f
    @property
    def k(self) :
        """Returns wavenumber in the x-direction in mm^-1."""
        return 1e-3*self.__k
    @property
    def Cphi(self) :
        """Deprecated - Returns phase velocity in mm/µs."""
        warnings.warn("Plane_Guided_Mode.Cphi is deprecated."
                      + "\n\tUse Plane_Guided_Mode.Vphi instead")
        return self.Vphi
    @property
    def Vphi(self) :
        """Returns phase velocity in mm/µs."""
        return 1e-3 * self.__v_phi
    @property
    def Ve_vector(self) :
        """Returns the energy velocity vector in mm/µs."""
        return 1e-3 * self.__v_en
    @property
    def Ve(self) :
        """Returns energy velocity vector in mm/µs.
           The real and imaginary parts are the velocity components
           in the x and y directions, respectively."""
        Vex, Vey = 1e-3 * self.__v_en
        if abs(Vey) < 1e-5 or abs(Vey) < 1e-7*abs(Vex) :
            return Vex
        return Vex + 1.0j*Vey
    @property
    def Vex(self) :
        """Returns energy velocity in the x-direction in mm/µs."""
        return 1e-3*self.__v_en[0]
    @property
    def Vey(self) :
        """Returns energy velocity in the y-direction mm/µs."""
        return 1e-3*self.__v_en[1]
    @property
    def is_a_true_guided_mode(self) :
        """Returns a boolean indicating if it is a true guided mode."""
        return self.__true_guided_mode
    @property
    def Vgr(self) :
        """Returns group velocity in mm/µs."""
        if self.__v_gr == self.__UNDEF :
            return None
        return 1e-3*self.__v_gr
    def __setVgr(self, new_Vgr, verbose = False) :
        """Set the group velocity value [in m/s]."""
        if isinstance(new_Vgr, float) :
            self.__v_gr = new_Vgr
        else :
            if verbose :
                print(f"Unable set group velocity value with {new_Vgr}")
            self.__v_gr = self.__UNDEF
    @Vgr.setter
    def Vgr(self, new_Vgr) :
        """Set the group velocity value [in mm/µs]."""
        try : 
            self.__setVgr(1e3*new_Vgr, verbose = True)
        except :
            print(f"Unable set group velocity value with {new_Vgr}")
            self.__v_gr = self.__UNDEF
    @property
    def name(self) :
        """Returns the mode name."""
        return self.__name
    @name.setter
    def name(self, new_name) :
        """Renames the mode."""
        try :
            self.__name = str(new_name)
        except :
            print(f"Unable to rename Mode {self.__name}")
    @property
    def Z(self) :
        """Returns the vector of the z-positions of the interfaces in mm.
        """
        return 1e3*self.__Z
    @property
    def zmin(self) :
        """Returns the minimum value of the vertical position in mm
           taken into account in the calculation of the energy velocity.
        """
        return 1e3*self.__zmin
    @property
    def zmax(self) :
        """Returns the maximum value of the vertical position in mm
           taken into account in the calculation of the energy velocity.
        """
        return 1e3*self.__zmax
    @property
    def fluid_half_spaces(self) :
        """Returns a pair of booleans."""
        return ( isinstance(self.__plate.left_fluid, Fluid),
                 isinstance(self.__plate.right_fluid, Fluid) )
    #---------------------------------------------------------------------
    def Ux(self, z) :
        """Returns the complex mode shape Ux, i.e. the displacement in
           the x-direction with respect to the vertical position z."""
        return self.__f_ux(z)
    #---------------------------------------------------------------------
    def Uy(self, z) :
        """Returns the complex mode shape Ux, i.e. the displacement in
           the y-direction with respect to the vertical position z."""
        return self.__f_uy(z)
    #---------------------------------------------------------------------
    def Uz(self, z) :
        """Returns the complex mode shape Ux, i.e. the displacement in
           the z-direction with respect to the vertical position z."""
        return self.__f_uz(z)
    #---------------------------------------------------------------------
    def Sxx(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the xx-direction with respect to the vertical position z."""
        return self.__f_sxx(z)
    #---------------------------------------------------------------------
    def Sxy(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the xy-direction with respect to the vertical position z."""
        return self.__f_sxy(z)
    #---------------------------------------------------------------------
    def Sxz(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the xz-direction with respect to the vertical position z."""
        return self.__f_sxz(z)
    #---------------------------------------------------------------------
    def Syy(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the yy-direction with respect to the vertical position z."""
        return self.__f_syy(z)
    #---------------------------------------------------------------------
    def Syz(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the yz-direction with respect to the vertical position z."""
        return self.__f_syz(z)
    #---------------------------------------------------------------------
    def Szz(self, z) :
        """Returns the complex mode shape Sxx, i.e. the stress component
           in the zz-direction with respect to the vertical position z."""
        return self.__f_szz(z)
    #---------------------------------------------------------------------
    def Px(self, z, complex_valued=True) :
        """Returns the x-component of the Poynting vector, i.e. the real
           part of (-1/2)Sx.V* = iw/2 Sx.U* with respect to the vertical
           position z. Option 'complex_valued=True' permits to obtain
           the x-component of the complex Poynting vector: -Sx.V*."""
        iws2 = 1.0j*np.pi*self.__f
        poy_x  = self.__f_sxx(z) * self.__f_ux(z).conjugate()
        poy_x += self.__f_sxy(z) * self.__f_uy(z).conjugate()
        poy_x += self.__f_sxz(z) * self.__f_uz(z).conjugate()
        if complex_valued : return iws2*poy_x
        else : return (iws2*poy_x).real
    #---------------------------------------------------------------------
    def Py(self, z, complex_valued=True) :
        """Returns the y-component of the Poynting vector, i.e. the real
           part of (-1/2)Sy.V* = iw/2 Sy.U* with respect to the vertical
           position z. Option 'complex_valued=True' permits to obtain
           the y-component of the complex Poynting vector: -Sy.V*."""
        iws2 = 1.0j*np.pi*self.__f
        poy_y  = self.__f_sxy(z) * self.__f_ux(z).conjugate()
        poy_y += self.__f_syy(z) * self.__f_uy(z).conjugate()
        poy_y += self.__f_syz(z) * self.__f_uz(z).conjugate()
        if complex_valued : return iws2*poy_y
        else : return (iws2*poy_y).real
    #---------------------------------------------------------------------
    def Pz(self, z, complex_valued=True) :
        """Returns the z-component of the Poynting vector, i.e. the real
           part of (-1/2)Sz.V* = iw/2 Sz.U* with respect to the vertical
           position z. Option 'complex_valued=True' permits to obtain
           the y-component of the complex Poynting vector: -Sz.V*."""
        iws2 = 1.0j*np.pi*self.__f
        poy_z  = self.__f_sxz(z) * self.__f_ux(z).conjugate()
        poy_z += self.__f_syz(z) * self.__f_uy(z).conjugate()
        poy_z += self.__f_szz(z) * self.__f_uz(z).conjugate()
        if complex_valued : return iws2*poy_z
        else : return (iws2*poy_z).real
    #---------------------------------------------------------------------
    def __a(self, z) :
        """Returns the half of the imaginary part of
           (dSzz/dz).V*-Sz.(dV/dz)* = iw (Sz.(dU/dz)*-(dSzz/dz).U*)
           with respect to the vertical position z."""
        iws2 = 1.0j*np.pi*self.__f
        val_a  = self.__f_sxz(z) * self.__f_dux(z).conjugate()
        val_a += self.__f_syz(z) * self.__f_duy(z).conjugate()
        val_a += self.__f_szz(z) * self.__f_duz(z).conjugate()
        val_a -= self.__f_dsxz(z) * self.__f_ux(z).conjugate()
        val_a -= self.__f_dsyz(z) * self.__f_uy(z).conjugate()
        val_a -= self.__f_dszz(z) * self.__f_uz(z).conjugate()
        return (iws2*val_a).imag
    #---------------------------------------------------------------------
    def __b(self, z) :
        """Returns the half of the real part of
           (dSzz/dz).V*+Sz.(dV/dz)* = -iw (Sz.(dU/dz)*+(dSzz/dz).U*)
           with respect to the vertical position z."""
        miws2 = -1.0j*np.pi*self.__f
        val_b  = self.__f_sxz(z) * self.__f_dux(z).conjugate()
        val_b += self.__f_syz(z) * self.__f_duy(z).conjugate()
        val_b += self.__f_szz(z) * self.__f_duz(z).conjugate()
        val_b += self.__f_dsxz(z) * self.__f_ux(z).conjugate()
        val_b += self.__f_dsyz(z) * self.__f_uy(z).conjugate()
        val_b += self.__f_dszz(z) * self.__f_uz(z).conjugate()
        return (miws2*val_b).real
    #---------------------------------------------------------------------
    def e_tot(self, z) :
        """Returns the average volume density of total energy
           with respect to the vertical position z:
           etot(z) = sx*px(z) + sy*py(z) + (1/2w)'*a(z) + (1/2w)''*b(z).
           """
        f,k,nu = self.__f,self.__k,self.__nu
        uns2f = 0.5/f
        etot = (0.5/np.pi)*( (k/f).real*self.Px(z).real
                             + (nu/f).real*self.Py(z).real
                             + uns2f.real*self.__a(z)
                             + uns2f.imag*self.__b(z) )
        return etot
    #---------------------------------------------------------------------
    def Ve_loc(self, z) :
        """Returns the energy velocity with respect to the vertical
           position z: (px(z),py(z))/e_tot(z)."""
        p = np.array([self.Px(z).real, self.Py(z).real])
        ve = np.zeros_like(p)
        etot = self.e_tot(z)
        indexes = np.where(etot>0)
        ve[:,indexes] = p[:,indexes]/etot[indexes]
        return ve       
    #---------------------------------------------------------------------
    def energy_Velocity(self, z_min_mm, z_max_mm, nb_val=200) :
        """ Returns Vex + i*Vey for z in [ z_min_mm, z_max_mm ],
            in mm/µs."""
        z_min, z_max = 1e-3*z_min_mm, 1e-3*z_max_mm
        EPSI = 1e-7
        ZERO_VALUE = 1e-4
        ZERO_KZ = 0.2*np.pi # Wavelength = 10 m
        ZERO_En_Ratio = 1e-6
        # Plate and layers positions
        Z, plate = self.__Z, self.__plate
        # Frequency, wavenumbers in the x and y directions
        f,k,nu = self.__f,self.__k,self.__nu
        uns2w = 0.25/(np.pi*f)
        # Slowness vector S'
        S = (0.5/np.pi)*np.array( ( (k/f).real, (nu/f).real ) )
        # Integration on interval [z_min, z_max ]
        if z_min < Z[0] :
            if plate.left_fluid not in (None,"Wall") :
                # Attention: stored Kz oriented with the decreasing z
                kz0 = -self.__mode_data[-2][0].imag
                z0m = Z[0] - EPSI*( Z[1]-Z[0] )
                dz = z0m - z_min
                mdkzdz = -2*kz0*dz
                if np.abs(mdkzdz) < 1e-12 :
                    factor0 = dz
                else :
                    factor0 = 0.5/kz0 * ( 1 - np.exp(mdkzdz) )
                # S'.P'
                Px_mean = factor0 * self.Px(z0m).real
                Py_mean = factor0 * self.Py(z0m).real
                # Additional terms
                Atot = factor0 * self.__a(z0m)
            z_min = Z[0]
        else :
            # S'.P'
            Px_mean = 0.0
            Py_mean = 0.0
            # Additional terms
            Atot = 0.0            
        if z_max >= Z[-1] :
            if plate.right_fluid not in (None,"Wall") :
                kze = self.__mode_data[-1][0].imag
                zep = Z[-1] + EPSI*( Z[-1]-Z[-2] )
                dz = z_max-zep
                dkzdz = 2*kze*dz
                if np.abs(dkzdz) < 1e-12 :
                    factore = dz
                else :
                    factore = 0.5/kze * ( np.exp(dkzdz) - 1 )
                # S'.P'
                Px_mean += factore * self.Px(zep).real
                Py_mean += factore * self.Py(zep).real
                # Additional terms
                Atot += factore * self.__a(zep)
            z_max = Z[-1]
        # Total Energy Etot
            # S'.P'
        Px_mean += self.__nintegrate("Re(Px)", nb_val, z_min, z_max)
        Py_mean += self.__nintegrate("Re(Py)", nb_val, z_min, z_max)
        P = np.array( (Px_mean,Py_mean) )
        SscalP = S@P
            # Additional terms
        Atot += self.__nintegrate("a", nb_val)            
        EA = uns2w.real * Atot
        if z_min == Z[ 0] : z_min += EPSI*(Z[1]-Z[0])
        if z_max == Z[-1] : z_max -= EPSI*(Z[-1]-Z[-2])
        Btot = self.Pz(z_min).real - self.Pz(z_max).real
        EB = uns2w.imag * Btot       
        Ve = P/(SscalP+EA+EB)
        Vex, Vey = 1e-3 * Ve
        if abs(Vey) < 1e-5 or abs(Vey) < 1e-7*abs(Vex) :
            return Vex
        return Vex + 1.0j*Vey
    #---------------------------------------------------------------------
    def __update_Energy_Velocity(self, nb_val=200) :
        EPSI = 1e-7
        ZERO_VALUE = 1e-4
        ZERO_KZ = 0.2*np.pi # Wavelength = 10 m
        ZERO_En_Ratio = 1e-6
        self.__true_guided_mode = True
        # Integration interval
        Z, plate = self.__Z, self.__plate
        e = plate.thickness
        z0,ze = Z[0],Z[-1]
        if plate.left_fluid not in (None,"Wall") :
            z0m = Z[0]-EPSI*(Z[1]-Z[0])
            # Attention: stored Kz oriented with the decreasing z
            kz0 = -self.__mode_data[-2][0].imag
            DA = kz0 > 0
            if DA :
                dz = 0.5*np.log(ZERO_VALUE)/(-kz0)
                z0 -= dz
            elif np.abs(self.__mode_data[-2][0]) <= ZERO_KZ : # SH ?
                z0p = Z[0]+EPSI*(Z[1]-Z[0])
                self.__true_guided_mode = (
                    np.abs(self.e_tot(z0m))
                    < ZERO_En_Ratio*np.abs(self.e_tot(z0p)) )
            else : # The vertical flux at the upper interface has to be
                   # negative for a leaky mode
                self.__true_guided_mode = self.Pz(z0m).real <= 0.0
            self.__decreasing_amplitude_on_lhs = DA
        if plate.right_fluid not in (None,"Wall") :
            zep = Z[-1]+EPSI*(Z[-1]-Z[-2])
            kze = self.__mode_data[-1][0].imag
            DA = kze < 0
            if DA :
                dz = 0.5*np.log(ZERO_VALUE)/kze
                ze += dz
            elif np.abs(self.__mode_data[-1][0]) <= ZERO_KZ : # SH ?
                zem = Z[-1]-EPSI*(Z[-1]-Z[-2])
                self.__true_guided_mode = (
                    np.abs(self.e_tot(zep))
                    < ZERO_En_Ratio*np.abs(self.e_tot(zem)) )
            else : # The vertical flux at the lower interface has to be
                   # positive for a leaky mode
                self.__true_guided_mode = (self.__true_guided_mode
                                           and self.Pz(zep).real >= 0.0)
            self.__decreasing_amplitude_on_rhs = DA
        self.__zmin = z0
        self.__zmax = ze
        # Total Energy Etot
            # S'.P'
        Px_mean = self.__nintegrate("Re(Px)", nb_val)
        Py_mean = self.__nintegrate("Re(Py)", nb_val)
        if self.__decreasing_amplitude_on_lhs :
            c0 = 0.5/kz0
            Px_mean += c0*self.Px(z0m).real
            Py_mean += c0*self.Py(z0m).real
        if self.__decreasing_amplitude_on_rhs :
            ce = -0.5/kze
            Px_mean += ce*self.Px(zep).real
            Py_mean += ce*self.Py(zep).real
        P = np.array( (Px_mean,Py_mean) )
        f,k,nu = self.__f,self.__k,self.__nu
        S = (0.5/np.pi)*np.array( ( (k/f).real, (nu/f).real ) )
        SscalP = S@P
            # Additional terms
        uns2w = 0.25/(np.pi*f)
        Atot = self.__nintegrate("a", nb_val)
        if self.__decreasing_amplitude_on_lhs :
            Atot += c0*self.__a(z0m)
        if self.__decreasing_amplitude_on_rhs :
            Atot += ce*self.__a(zep)
        EA = uns2w.real * Atot
        if z0 == Z[0] : z0 += EPSI*(Z[1]-Z[0])
        if ze == Z[-1] : ze -= EPSI*(Z[-1]-Z[-2])
        Btot = self.Pz(z0).real - self.Pz(ze).real
        EB = uns2w.imag * Btot       
        self.__v_en = P/(SscalP+EA+EB) 
    #---------------------------------------------------------------------
    def __nintegrate(self, field, nb_val=200, z_min="auto", z_max="auto"):
        """Average value of a field over the thickness of the plate
           (default) or over the interval [z_min_mm, z_max_mm]."""
        IOTA = 1e-4
        if field in self.keys() :
            fld = self[field]
        elif field == "|U|²" :
            """sqrt( < |Ux|**2 + |Uy|**2 + |Uz|**2 > )"""
            fld = lambda z : ( self.Ux(z)*self.Ux(z).conjugate()
                               + self.Uy(z)*self.Uy(z).conjugate()
                               + self.Uz(z)*self.Uz(z).conjugate() )
        elif field == "Sv" :
            """< (Sxx + Syy + Szz) / 3 >"""
            fld = lambda z : (self.Sxx(z)+self.Syy(z)+self.Szz(z))/3
        elif field == "a" :
            fld = self.__a
        else :
            msg = "Plane_Guided_Mode.nintegrate:"
            msg += f"\n\t\tUnknown '{field}' field name."
            warnings.warn(msg)
            return 0.0
        e = self.__plate.thickness
        dz = e/(nb_val-1)
        n_int = len(self.__Z)
        if z_min == "auto" :
            z_min = self.__Z[0]
            i0 = 1
        elif z_min >= self.__Z[-1]:
            i0 = n_int
        else: # z_min < self.__Z[-1]
            i0 = ( z_min < self.__Z ).argmax()
        if z_max == "auto" :
            z_max = self.__Z[-1]
            i1 = n_int - 1
        else :
            if z_max <= z_min :
                msg = "z_min >= z_max. 0 returned."
                warnings.warn(msg)
                return 0.0
            if z_max <= self.__Z[0] :
                i1 = 0
            else:
                i1 = n_int - (z_max > self.__Z[::-1]).argmax()    
        somme = 0.0
        Zmil = self.__Z[i0:i1].tolist()
        for z0,z1 in zip([z_min]+Zmil, Zmil+[z_max]) :
            n = round( (z1-z0)/dz )
            Vz = np.linspace(z0+IOTA*dz,z1-IOTA*dz,n+1)
            values = fld(Vz)
            somme += ( values[1:-1].sum()
                       + 0.5*(values[0]+values[-1]) ) * (z1-z0)/n
        return somme
    #---------------------------------------------------------------------
    def nintegrate(self, field, nb_val=200, z_min_mm="auto",
                   z_max_mm="auto"):
        """Average value of a field over the thickness of the plate
           (default) or over the interval [z_min_mm, z_max_mm]."""
        if z_min_mm == "auto" : z_min = "auto"
        else : z_min = 1e-3*z_min_mm
        if z_max_mm == "auto" : z_max = "auto"
        else : z_max = 1e-3*z_max_mm
        return self.__nintegrate(field, nb_val, z_min, z_max)
    #---------------------------------------------------------------------
    def abs_correlation_coefficient(self, other_mode, nb_val=200,
                                    verbose=False):
        if self.__plate is not other_mode.__plate:
            msg = ( "Plane_Guided_Mode.abs_correlation_coefficient :: "
                    + "{}:\n\tThe 2 modes are not defined on the same "
                    + "plate!" )
            if np.allclose([ a.e for a in self.__plate.layers ],
                           [ a.e for a in other_mode.__plate.layers ]):
                if verbose :
                    print(msg.format("Warning"))
            else :
                raise ValueError(msg.format("Error"))
        e = self.__plate.thickness
        dz = e/(nb_val-1)
        S1,S2,SR = 0.0,0.0,0.0
        for z0,z1 in zip(self.__Z[:-1],self.__Z[1:]) :
            n = round( (z1-z0)/dz )
            Vz = np.linspace(z0+1e-4*dz,z1-1e-4*dz,n+1)
            U1 = np.array( [ self.Ux(Vz), self.Uy(Vz), self.Uz(Vz)] )
            U2 = np.array( [ other_mode.Ux(Vz), other_mode.Uy(Vz),
                             other_mode.Uz(Vz)] )
            V1 = np.einsum("ij,ij->j", U1, U1.conjugate())
            U2c = U2.conjugate()
            V2 = np.einsum("ij,ij->j", U2, U2c)
            VR = np.einsum("ij,ij->j", U1, U2c)
            h = (z1-z0)/n
            S1 += ((V1[1:-1].sum()+0.5*(V1[0]+V1[-1])) * h).real
            S2 += ((V2[1:-1].sum()+0.5*(V2[0]+V2[-1])) * h).real
            # The correlation coefficient is complex
            SR += ((VR[1:-1].sum()+0.5*(VR[0]+VR[-1])) * h)
        return abs(SR)/np.sqrt(S1*S2) 
    #---------------------------------------------------------------------
    def nearest_mode_index(self, modes, nb_val=200):
        if len(modes) == 0 : # Should not be possible
            return None, None
        LACC = np.array( [
                self.abs_correlation_coefficient(m, nb_val)
                if m is not None else -2.0 for m in modes] )
        idx = LACC.argmax()
        return idx, LACC[idx]
    #---------------------------------------------------------------------
    def mean(self, field, nb_val=200):
        e = self.__plate.thickness
        if field == "U" :
            return np.sqrt(self.__nintegrate("|U|²", nb_val)/e)
        else :
            return self.__nintegrate(field, nb_val)/e
    #---------------------------------------------------------------------
    def normalize(self, field="Re(Px)"):
        U_mean = self.mean("U")
        if field == "Re(Px)" : # Px_mean = 1 W/mm²
            Px_mean = self.mean("Re(Px)")
            try :
                coef = np.sqrt(1e6/np.abs(Px_mean))
                ok = U_mean*coef < self.__Umax
            except :
                ok = False
            if not ok : # Normalization with respect to displacement
                coef = self.__Umean/U_mean
            if self.__verbose :
                if Px_mean < 0 :
                    if ok :
                        print("Plane_Guided_Mode::normalize:\n\t"
                              + "Warning: backward propagation")
                    else :
                        print("Plane_Guided_Mode::normalize:\n\t"
                              + "Warning: nonpropagative increasing mode")
                else :
                    if not ok :
                        print("Plane_Guided_Mode::normalize:\n\t"
                              + "Warning: nonpropagative mode")
        elif field == "U" : # U_mean = 1µm
            coef = self.__Umean/U_mean
        else :
            print("Plane_Guided_Mode::normalize:\n\t"
                  + "Error: normalization with respect to '{field}'\n\t"
                  + "not available")            
        self.__update_shape_functions(coef)
    #---------------------------------------------------------------------
    def export(self, path, dz_mm=0.01, c_half_space=5.0,
               file_format = "Matlab", verbose=False):
        """'path' : None -> return a dict
                    dir  -> Automatic name. Needs 'file_format'
                            (Matlab/numpy)
                    file -> .mat extension (Matlab) or
                            .npz extension (numpy)
        """
        if verbose : prt = print
        else : prt = lambda *args,**kwargs : None
        dico_to_save = {"f_MHz": self.f, "k_mm^-1": self.k,
                        "Vph_mm/µs": self.Vphi}
        edz = 0.01*dz_mm
        if path is None :
            file_format = ""
        elif os.path.isdir(path) : # path is a directory
            file_path = os.path.join(path, self.name + ".")
        elif path.endswith(".mat") :
            file_path,file_format = path[:-3],"Matlab"
        elif path.endswith(".npz") :
            file_path,file_format = path[:-3],"numpy"
        else :
            msg = ("*** Plane_Guided_Mode.export ***\n\t"
                   + f"path '{path}' not recognized")
            raise ValueError(msg)            
        if self.__plate.left_fluid is not None :
            Z_hs = c_half_space*(self.Z[-1]-self.Z[0])
            nb = round(Z_hs/dz_mm) + 1
            Lz = (np.linspace(-Z_hs, 0, nb) + self.Z[0]).tolist()
            Lz[-1] -= edz
        else :
            Lz = []
        for z0_mm,z1_mm in zip(self.Z[:-1],self.Z[1:]) :
            nb = round((z1_mm-z0_mm)/dz_mm) + 1
            Mz = np.linspace(z0_mm,z1_mm, nb).tolist()
            Mz[0] += edz ; Mz[-1] -= edz
            Lz.extend(Mz)
        if self.__plate.right_fluid is not None :
            Z_hs = c_half_space*(self.Z[-1]-self.Z[0])
            nb = round(Z_hs/dz_mm) + 1
            Mz = (np.linspace(0, Z_hs, nb) + self.Z[-1]).tolist()
            Mz[0] += edz
            Lz.extend(Mz)
        Z_mm = np.array(Lz)
        dico_to_save["z_mm"] = Z_mm
        Z_m = 1e-3*Z_mm
        for field,fonc in self.__dico_fields.items() :
            if field.startswith("U") :
                key,c = field+"_µm",1e6
            elif field.startswith("P") :
                key,c = field+"_W/mm²",1e-6
            elif field.startswith("S") :
                key,c = field+"_MPa",1e-6
            elif field == "e_tot" :
                key,c = "Et_µJ/mm³",1e-3
            elif field == "Ve loc" :
                key,c = "Ve_mm/µs",1e-3
            else :
                prt(f"Field '{field}' not exported.")
                continue
            dico_to_save[key] = c*fonc(Z_m)
        prt(f"Exported Fields : {dico_to_save.keys()}")
        if "matlab" in file_format.lower() :
            return savemat(file_path + "mat", dico_to_save)
        elif "numpy" in file_format.lower() :
            return np.savez(file_path + "npz", **dico_to_save)
        return dico_to_save # Just returns the dict
    #---------------------------------------------------------------------
    def show_mode_shapes(self, z_min="auto", z_max="auto"):
        if Plane_Guided_Mode.__current_app is None :
            Plane_Guided_Mode.__current_app = QApplication(sys.argv)
        my_app = ModeShapeViewer(self, z_min, z_max)
        Plane_Guided_Mode.__current_app.exec_()
    #---------------------------------------------------------------------
    def parameters_to_be_saved(self) :
        """Returns a dictionary of the parameters of the mode."""
        Ve_plate_only = self.energy_Velocity(self.Z[0], self.Z[-1])
        sp = {"f_MHz": 1e-6*self.__f, "k_mm^-1": 1e-3*self.__k,
              "nu_mm^-1": 1e-3*self.__nu, "Ve_mm/µs": self.Ve,
              "Ve_plate_only_mm/µs": Ve_plate_only,
              "Vph_mm/µs": self.Vphi, "name": self.__name}
        if self.__v_gr == self.__UNDEF : sp["Vg_mm/µs"] = None
        else : sp["Vg_mm/µs"] = 1e-3*self.__v_gr
        for no,(lay,md) in enumerate(zip(self.__plate.layers,
                                         self.__mode_data), 1) :
            sp[f"Layer #{no}"] = {"Thickness_mm": 1e3*lay.thickness,
                                  "Kz_mm^-1": 1e-3*md[0],
                                  "Cz": md[1]}
        md = self.__mode_data[-2]
        sp["Upper Half-Space"] =  {"Kz_mm^-1": 1e-3*md[0], "Cz": md[1]}
        md = self.__mode_data[-1]
        sp["Lower Half-Space"] =  {"Kz_mm^-1": 1e-3*md[0], "Cz": md[1]}
        return sp
    #---------------------------------------------------------------------
    @property
    def Kz_up(self) :
        """Vertical wavenumber in mm^-1 (0 if upper half-space is vacuum).
        """
        return 1e-3*self.__mode_data[-2][0]
    #---------------------------------------------------------------------
    @property
    def Kz_down(self) :
        """Vertical wavenumber in mm^-1 (0 if lower half-space is vacuum).
        """
        return 1e-3*self.__mode_data[-1][0] 
    #---------------------------------------------------------------------
    @staticmethod
    def from_dict(plate, dict_mode) :
        """Returns a Plane_Guided_Mode instance from the discretized
           plate 'plate' and the dictionary 'dict_mode'.
        """
        f  = 1e6*dict_mode["f_MHz"]
        k  = 1e3*dict_mode["k_mm^-1"]
        nu = 1e3*dict_mode["nu_mm^-1"]
        name = dict_mode["name"]
        mode_data = []
        for no,_ in enumerate(plate.layers,1) :
            sd = dict_mode[f"Layer #{no}"]
            mode_data.append( [1e3*sd["Kz_mm^-1"], sd["Cz"],None] )
        for key in ("Upper Half-Space","Lower Half-Space") :
            sd = dict_mode[key]
            mode_data.append( [1e3*sd["Kz_mm^-1"], sd["Cz"]] )
        # No normalization when restoring a plane guided mode
        return Plane_Guided_Mode(plate, f, k, mode_data, nu, name, False)
#=========================================================================
class ModeShapeViewer(QWidget) :
    def __init__(self, mode, z_min="auto", z_max="auto"):
        QWidget.__init__(self)        
        ok_but = QPushButton("OK",self)
        ok_but.released.connect(self.close)
        title = QLabel( "  "+mode.__str__()+"  " )
        title.setFont( QFont("Arial", 14, QFont.Bold))
        title.setStyleSheet("color: rgb(0,0,200) ;"
                            + "background-color: rgb(255,240,230) ;")
        title.setFixedHeight(60)
        vlay = QVBoxLayout()
        hlay1 = QHBoxLayout()
        hlay1.addStretch()
        hlay1.addWidget(title)
        hlay1.addStretch()
        vlay.addLayout(hlay1)
        vlay.addWidget( Mode_Shape_Figure(self, mode, z_min, z_max) )
        hlay2 = QHBoxLayout()
        hlay2.addStretch()
        hlay2.addWidget(ok_but)
        hlay2.addStretch()
        vlay.addLayout(hlay2)
        self.setLayout(vlay)
        self.setGeometry(200, 200, 1200, 600)
        self.show()   
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    from Modes_Immersed_Multilayer_Plate import DiscretizedMultilayerPlate
    # Import materials
    mat_dir = root+"/Data/Materials/"
    ImpMatFromFile = ImportMaterialFromFile
    CASE = "import/export"      # in ("Tests", "Nylon", "GlassEpoxy",
                                #     "A_Bernard", "Lame Fluide",
                                #     "import/export")
    if CASE in ("Tests", "Nylon","import/export") :
        nylon,_,_ = ImpMatFromFile(mat_dir+"FS_Nylon_21-07.txt")
        water,_,_ = ImpMatFromFile(mat_dir+"FS_Water.txt")
        # Definition of the plate
        epaisseur = 2.26e-3 # 2.26 mm
        plq = DiscretizedMultilayerPlate(nylon, epaisseur, 50)
        plq.set_right_fluid(water)
    if CASE == "Tests" :
        from random import choice
        bicouche = False
        if bicouche :
            plq = DiscretizedMultilayerPlate(nylon, epaisseur-0.8e-3, 40)
            plq.add_discretizedLayer(nylon, 0.8e-3, 50)
            plq.set_right_fluid("Wall")
        freq = 240.0e3
        modes = plq.modes_for_given_frequency( freq, rel_err=1e-3,
                                               rel_kappa_err=0.1 )
        modes = [ m for m in modes if 0<=-m.k.imag<=0.1 ]
##
        for X in modes :
        #X = choice(modes)
            X.name = "X"
            print(f"Freq. {1e-3*freq:.0f} kHz, X : ",
                  f"Vphi ~ {X.Vphi:.4f} mm/µs, ",
                  f"att ~ {-X.k.imag:.5f} Neper/m,",
                  f"\n\t\tVe ~ {X.Vex:.5f} mm/µs")
            n_plq = (2*113+1)
            Vz_plq = np.linspace(0, plq.e, n_plq)
            dz = Vz_plq[1]-Vz_plq[0]
            e_L = 0.4*plq.e + max(0.4*plq.e, -0.5e-3*X.zmin)
            Vz_L = dz*np.arange( -(e_L//dz)+1,1)
            e_R = max(0.4*plq.e, 0.5*(1e-3*X.zmax-plq.e))
            Vz_R = plq.e + dz*np.arange(e_R//dz)
            ddz = 1e-4*dz
            Vz_plq[0] += ddz
            Vz_plq[-1] -= ddz
            Vz_L[-1] -= ddz
            Vz_R[0] += ddz
            Vz = np.append(Vz_L,np.append(Vz_plq,Vz_R))
            VUx = X["Ux"](Vz)
            VUy = X["Uy"](Vz)
            VUz = X["Uz"](Vz)
            VSxz = X["Sxz"](Vz)
            VSyz = X["Syz"](Vz)
            VSzz = X["Szz"](Vz)
            VdUx = X["dUx/dz"](Vz)
            VdUy = X["dUy/dz"](Vz)
            VdUz = X["dUz/dz"](Vz)
            VdSxz = X["dSxz/dz"](Vz)
            VdSyz = X["dSyz/dz"](Vz)
            VdSzz = X["dSzz/dz"](Vz)
            VEtot = X.e_tot(Vz)
            VVex = X.Ve_loc(Vz)[0]
            plt.figure("Champs et leurs dérivées en z", figsize=(18,9))
            opt = {"family":"Arial", "weight":"bold", "size":14}
            axes = [plt.subplot(2,2,i) for i in (1,2,3,4)]
            ax_Ux, ax_Uz, ax_Sxz, ax_Szz = axes               
            for ax in (ax_Sxz, ax_Szz) :
                ax.set_xlabel("Vertical position $z$ [mm]", **opt)
            ax_Ux.set_ylabel("Displacements [µm]", **opt)
            ax_E = ax_Uz.twinx()
            ax_E.set_ylabel("Volume Total Energy [mJ/mm³]", **opt)
            ax_Sxz.set_ylabel("Stresses [MPa]", **opt)
            ax_Ve = ax_Szz.twinx()
            ax_Ve.set_ylabel("Energy Velocity [mm/µs]", **opt)
            plt.subplots_adjust(left=0.05, right=0.95, bottom=0.06,
                                top=0.94, hspace=0.1, wspace=0.1)
            Vz_mm = 1e3*Vz
            ax_Ux.plot( Vz_mm, 1e6*VUx.real, "-g", label="Re$(u_x)$")
            ax_Ux.plot( Vz_mm, 1e6*VUx.imag, "--g", label="Im$(u_x)$")
            ax_Ux.plot( Vz_mm, 1e6*VUy.real, "-b", label="Re$(u_y)$")
            ax_Ux.plot( Vz_mm, 1e6*VUy.imag, "--b", label="Im$(u_y)$")
            ax_Ux.plot( Vz_mm, 1e3*VdUx.real, "-m",
                        label=r"Re$(u_x^{\prime})$")
            ax_Ux.plot( Vz_mm, 1e3*VdUx.imag, "--m",
                        label="Im$(u_x^{\prime})$")
            ax_Ux.plot( Vz_mm, 1e3*VdUy.real, "-c",
                        label=r"Re$(u_y^{\prime})$")
            ax_Ux.plot( Vz_mm, 1e3*VdUy.imag, "--c",
                        label="Im$(u_y^{\prime})$")
            ax_Uz.plot( Vz_mm, 1e6*VUz.real, "-g", label="Re$(u_z)$")
            ax_Uz.plot( Vz_mm, 1e6*VUz.imag, "--g", label="Im$(u_z)$")
            ax_Uz.plot( Vz_mm, 1e3*VdUz.real, "-m",
                        label=r"Re$(u_z^{\prime})$")
            ax_Uz.plot( Vz_mm, 1e3*VdUz.imag, "--m",
                        label="Im$(u_z^{\prime})$")
            ax_E.plot( Vz_mm, 1e-3*VEtot, "b", linewidth=2.0,
                       label = "Volume Energy")
            ax_Sxz.plot( Vz_mm, 1e-6*VSxz.real, "-g",
                         label=r"Re$(\sigma_{xz})$")
            ax_Sxz.plot( Vz_mm, 1e-6*VSxz.imag, "--g",
                         label=r"Im$(\sigma_{xz})$")
            ax_Sxz.plot( Vz_mm, 1e-6*VSyz.real, "-b",
                         label=r"Re$(\sigma_{yz})$")
            ax_Sxz.plot( Vz_mm, 1e-6*VSyz.imag, "--b",
                         label=r"Im$(\sigma_{yz})$")
            ax_Sxz.plot( Vz_mm, 1e-9*VdSxz.real, "-m",
                         label=r"Re$(\sigma_{xz}^{\prime})$")
            ax_Sxz.plot( Vz_mm, 1e-9*VdSxz.imag, "--m",
                         label=r"Im$(\sigma_{xz}^{\prime})$")
            ax_Sxz.plot( Vz_mm, 1e-9*VdSyz.real, "-c",
                         label=r"Re$(\sigma_{yz}^{\prime})$")
            ax_Sxz.plot( Vz_mm, 1e-9*VdSyz.imag, "--c",
                         label=r"Im$(\sigma_{yz}^{\prime})$")
            ax_Szz.plot( Vz_mm, 1e-6*VSzz.real, "-g",
                         label=r"Re$(\sigma_{zz})$")
            ax_Szz.plot( Vz_mm, 1e-6*VSzz.imag, "--g",
                         label=r"Im$(\sigma_{zz})$")
            ax_Szz.plot( Vz_mm, 1e-9*VdSzz.real, "-m",
                         label=r"Re$(\sigma_{zz}^{\prime})$")
            ax_Szz.plot( Vz_mm, 1e-9*VdSzz.imag, "--m",
                         label=r"Im$(\sigma_{zz}^{\prime})$")
            ax_Ve.plot( Vz_mm, 1e-3*VVex, "b", linewidth=2.0,
                        label="Energy Velocity")
            plt.suptitle(f"Vitesse de phase {X.Vphi:.3f} mm/µs, "
                         + f"Vitesse d'énergie {X.Vex:.3f} mm/µs, "
                         + f"Nombre d'onde {X.k:.3f} "
                         + "$\mathrm{mm}^{-1}$, "
                         + f"Fréquence {1e3*X.f:.1f} kHz, "
                         + f"vrai mode : {X.is_a_true_guided_mode}",
                         **opt)
            for ax in axes :
                ax.grid() ; ax.legend(loc="upper left")
            for ax in (ax_E,ax_Ve) :
                ax.grid(color="cyan") ; ax.legend(loc="upper right")
            e_max = 1e-3*VEtot.max()
            ax_E.set_ylim(-0.02*e_max,1.02*e_max)
            ve_max = 1e-3*VVex.max()
            ax_Ve.set_ylim(-0.02*ve_max,1.22*ve_max)
            plt.show()
##
    elif CASE in ("Nylon","import/export") :
        # Mode computation
        for freq in (240.0e3,) : #(180.0e3,195.0e3,210.0e3,240.0e3
            modes = plq.modes_for_given_frequency( freq, rel_err=1e-3,
                                                   rel_kappa_err=0.1 )
            S01 = None
            for mode in modes :
                if np.abs(mode.Vphi-1.5)<0.2 and -0.1<mode.k.imag<=0.1 :
                    S01 = mode
                    S01.name = "S01"
                    print(f"Freq. {1e-3*freq:.0f} kHz, S01 : ",
                          f"Vphi ~ {S01.Vphi:.4f} mm/µs, ",
                          f"att ~ {-S01.k.imag:.5f} mm^-1,",
                          f"\n\t\tVex ~ {S01.Vex:.5f} mm/µs")
            # Interactive drawing of mode shapes
            S01.show_mode_shapes()
        if CASE == "Nylon" :
            sel_modes = [ m for m in modes if abs(m.k.imag) < 0.1 ]
            MC = np.array([ [ m1.abs_correlation_coefficient(m2)
                              for m2 in sel_modes] for m1 in sel_modes ])
            print(MC.round(3))
        else : # CASE == "import/export"
            dico_S01 = S01.parameters_to_be_saved()
            restored_mode = Plane_Guided_Mode.from_dict(plq, dico_S01)
            restored_mode.show_mode_shapes()
##
    elif CASE == "GlassEpoxy" :
        glass_epoxy,_,_ = ImpMatFromFile(mat_dir+"FS_GlassEpoxy.txt")
        water,_,_ = ImpMatFromFile(mat_dir+"FS_Water.txt")
        # Definition of the plate
        epaisseur = 2.65e-3 # 2.65 mm
        plq = DiscretizedMultilayerPlate(glass_epoxy, epaisseur, 30)
        plq.set_right_fluid(water)
        # Mode computation
        freq = 215.0e3      # 215 kHz
        modes = plq.modes_for_given_frequency( freq, rel_err=1e-3,
                                               rel_kappa_err=0.1 )
        A01,S02 = None,None
        for mode in modes :
            if np.abs(mode.Vphi-1.0)<0.2 and -0.1<mode.k.imag<=0.1 :
                A01 = mode
                A01.name = "A01"
            elif np.abs(mode.Vphi-3.5)<0.2 and -0.1<mode.k.imag<=0. :
                S02 = mode
                S02.name = "S02"
        print("A01 :", f"Vphi ~ {A01.Vphi:.4f} mm/µs, ",
                       f"att ~ {-A01.k.imag:.5f} mm^-1.",
              "\nS02 :", f"Vphi ~ {S02.Vphi:.4f} mm/µs, ",
                         f"att ~ {-S02.k.imag:.5f} mm^-1.")
        # Interactive drawing of mode shapes
        A01.show_mode_shapes()
        S02.show_mode_shapes()
##
    elif CASE == "A_Bernard" :
        # Mode S1, pages 69 à 71 de la thèse d"Arnaud Bernard
        alu,_,_ = ImpMatFromFile(mat_dir+"AB_Aluminum.txt")
        #+print(alu)
        water,_,_ = ImpMatFromFile(mat_dir+"AB_Water.txt")
        #print(water)
        # Definition of the plate
        epaisseur = 1.0e-3 # 1 mm
        plq = DiscretizedMultilayerPlate(alu, epaisseur, 30)
        plq.set_left_fluid(water)
        plq.set_right_fluid(water)
        # Mode computation
        for f_mHz,v_phi in ( (4.0,6.06), (6.0,4.82), (8.0,3.76),
                             (10.0,3.43) ) :
            freq = f_mHz*1e6
            modes = plq.modes_for_given_frequency( freq, rel_err=1e-3,
                                                   rel_kappa_err=0.1 )
            S1 = None
            for mode in modes :
                if np.abs(mode.Vphi-v_phi)<0.01 and -0.3<mode.k.imag<=0 :
                    S1 = mode
                    S1.name = "S1"
                    print("S1 :", f"Vphi ~ {S1.Vphi:.4f} mm/µs, ",
                          f"att ~ {-S1.k.imag:.5f} mm^-1.")
        # Interactive drawing of mode shapes
                    S1.show_mode_shapes(-0.7, 1.7)
##
    elif CASE == "Lame Fluide" :
        water,_,_ = ImpMatFromFile(mat_dir+"FS_Water.txt")
        light_water,_,_ = ImpMatFromFile(mat_dir+"Light_Water.txt")
        plq = DiscretizedMultilayerPlate(water, 3.0e-3, 30)
        plq.set_right_fluid(light_water)
        modes = plq.modes_for_given_frequency( 0.3e6, rel_err=1e-3,
                                                   rel_kappa_err=0.1 )
        modes = [m for m in modes if m.Vphi < 20.0 and
                                  np.abs(m.k.imag) < 10.0]
        modes[-1].show_mode_shapes()
