# Version 3.78 - 2024, February, 14
# Copyright (Eric Ducasse 2020)
# Licensed under the EUPL-1.2 or later
# Institution:  I2M / Arts & Metiers ParisTech
# Program name: TraFiC (Transient Field Computation)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
import os
from os.path import basename, dirname, abspath, isfile
if __name__ == "__main__" :
    import sys
    sys.path.append( abspath("..") )
    import TraFiC_init
import numpy as np
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def clean(chaine) :
    """ function converting to lowercase and dropping accents"""
    c = chaine.lower()
    for o,n in [('à','a'), ('é','e'), ('è','e'), ('ê','e'), ('ë','e'), \
                ('î','i'), ('ï','i'), ('ô','o'), ('ö','o'), ('û','u'), \
                ('ü','u'), ('ù','u'), ('  ',' '),('   ',' '), \
                ('\t',' ') ] :
        c = c.replace(o,n)
    return c
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ImportMaterialFromText(txt,verbose=False) :
    txt = txt.lower()
    if "[mg/mm³]" in txt and ("[gpa]" in txt or "[mm/µs]" in txt) :
        units = "mm_µs_mg_GPa"
    else :
        units = "SI"
    if verbose : print("Units:", units)
    param = dict()
    lignes = txt.split("\n")
    cplx = False
    for line in lignes :
        lu = [m.strip() for m in line.split(":")] 
        if len(lu) == 2 :
            key = clean(lu[0])
            try :
                value = float(lu[1])
                param[key] = value
                try_again = False
                err = ""
            except Exception as e1 :
                try_again = True
                err = f"{e1} / "                
            if try_again :
                try :
                    value = complex(lu[1])
                    param[key] = value
                    cplx = True
                except Exception as e2 :                  
                    if verbose :
                        print(f"{err}{e2}\nNon numerical value :",lu)
                    param[key] = lu[1]
        else :
            if verbose :
                print(f"line '{lu}' not taken into account")
    if "type" in param.keys() :
        matType = param.pop("type")
        if ("complex" in matType and not cplx) :
            print(f"Warning: type '{matType}' without complex value")
        elif ("complex" not in matType and cplx) :
            print(f"Warning: type '{matType}' with complex value(s)")
    else :
        matType = "Unknown"
    if "name" in param.keys() :
        matName = param.pop("name").title()
    elif "nom" in param.keys() : # French word
        matName = param.pop("nom").title()
    else :
        matName = "Unknown"
    prm = dict()
    for k in param.keys() :
        if isinstance(param[k],float) :
            prm[k] = param[k]
        elif isinstance(param[k],complex) :
            prm[k] = param[k]
        else :
            if verbose :
                print(f"Parameter '{k}': '{param[k]}' non numerical!")
    if matType.lower().endswith(" with complex stiffnesses") :
        matType = matType[:-25]
    if matType.lower() in ["fluid", "gas", "liquid", "fluide", \
                           "liquide", "gaz"] :
        matType = "Fluid"
    elif matType.lower() in ["materiau elastique", "solide elastique", \
                             "elastic", "elastic solid", \
                             "elastic material", "elastic medium", \
                             "materiau elastique anisotrope", \
                             "solide elastique anisotrope", \
                             "anisotropic elastic", \
                             "anisotropic elastic solid", \
                             "anisotropic elastic material", \
                             "anisotropic elastic medium"] :
        matType = "AnisotropicElasticSolid"
    elif matType.lower() in ["materiau elastique isotrope", \
                             "solide elastique isotrope", \
                             "milieu elastique isotrope", \
                             "isotropic elastic","elastic isotropic", \
                             "isotropic elastic solid", \
                             "elastic isotropic solid", \
                             "isotropic elastic material", \
                             "elastic isotropic material", \
                             "isotropic elastic medium", \
                             "elastic isotropic medium"] :
        matType = "IsotropicElasticSolid"
    elif matType.lower() in ["materiau elastique isotrope transverse", \
                             "solide elastique isotrope transverse", \
                             "milieu elastique isotrope transverse", \
                             "elastique isotrope transverse", \
                             "transversely isotropic elastic", \
                             "transversely isotropic elastic solid", \
                             "transversely isotropic solid", \
                             "transversely isotropic elastic material", \
                             "transversely isotropic material", \
                             "transversely isotropic elastic medium", \
                             "transversely isotropic medium" ] :
        matType = "TransverselyIsotropicElasticSolid"
    elif matType.lower() in [] :
        pass
    if matType in Material.subclasses() : # known material type
        const_instr = f"{matType}({prm},'{matName}',units='{units}')"
        if verbose : print(const_instr)
        newMat = eval(const_instr)
    else :
        newMat = (matName,matType,prm)
    return newMat
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def ImportMaterialFromFile(filePath=None,verbose=False) :
    """ function importing a material from a file"""
    if filePath is not None :
        if isfile(filePath) :
            filePath = abspath(filePath)
        else :
            msg = f"Material file path\n\t'{filePath}'" + \
                  "\n does not exist"
            raise FileNotFoundError(msg)
    if filePath is None :
        msg = "Material file path not given."
        raise FileNotFoundError(msg)            
    dossier = dirname(filePath)
    fichier = basename(filePath)
    with open(filePath,"r",encoding="utf-8") as f :
        txt = f.read()
    newMat = ImportMaterialFromText(txt,verbose)
    return newMat,fichier,dossier
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Material(object) :
    """ Homogeneous material
        'param' is a dictionary giving the parameters values,
        'units':    'SI'           -> kg/m³ ;   m/s;   Pa.
                    'mm_µs_mg_GPa' -> mg/mm³ ; mm/µs; GPa.
        'name' is a string."""
    # List of the class attributes
    __attrNames = tuple( "_Material__" + nm for nm in \
                              ("rho", "name", "cplx") )
    # Subclasses of the 'Material' classes
    __subclasses = []
    # Minimum coefficient for significant imaginary parts
    zero_num = 1e-7
    #--------------------------------------------------------------------
    # to print or not messages
    __VERBOSE = False
    @staticmethod
    def VERBOSE() : return Material.__VERBOSE
    #--------------------------------------------------------------------
    @staticmethod
    def attrNames() : return list(Material.__attrNames)
    @staticmethod
    def subclasses() : return Material.__subclasses
    @staticmethod
    def addsubclass(className) : Material.__subclasses.append(className)
    def prt(self,*args) : # to manage optional printing
        if Material.__VERBOSE : print(*args)
        else : pass
    #--------------------------------------------------------------------
    def __init__(self, param, name, units="SI") :
        self.prt(f"+++ Constructor of '{self.type}' +++")
        self.__cplx = False
        self.__rho = None
        SI_units = ( units != "mm_µs_mg_GPa" )
        if isinstance(param,dict) : # building by parameters dictionary
            for k,val in param.items() :
                if "rho" in k.lower() or "density" in k.lower() \
                   or "mass" in k.lower() :
                    if SI_units :
                        self.__rho = param[k]     # [kg/m³]
                    else :
                        self.__rho = 1e3*param[k] # [mg/mm³]
                if isinstance(val,complex) :
                    self.__cplx = True
        else :
            msg = "Material constructor error: the first parameter\n" + \
                  "must be of type 'dict' and not " + \
                 f"'{type(param).__name__}'"
            raise ValueError(msg)
        if self.__rho is None :
            self.prt("Constructor error: mass density not found")
        self.__name = str(name).title()
    #--------------------------------------------------------------------
    @property
    def rho(self) : return self.__rho
    def setRho(self, newrho, units="SI") :
        try :
            if units == "mm_µs_mg_GPa" :
                newrho = 1e3*float(newrho)
            else :
                newrho = float(newrho)
            if newrho >=0 :
                self.__rho = newrho
                self.prt(f"New mass density: {1e-3*newrho:.4f} mg/mm³")
                self.check() # Updating the parameters of the derived
                             # classes
                return
            raise
        except :
            print(f"'{newrho}' is not a positive number!")
            return
    #--------------------------------------------------------------------
    @property
    def name(self) : return self.__name
    def setName(self,newname) :
        try :
            self.__name = str(newname)
            self.prt(f"New name: {self.__name}")
            return
        except :
            print(f"'{newname}' cannot be converted to 'str'")
            return
    #--------------------------------------------------------------------
    @property
    def iscomplex(self) :
        return  self.__cplx
    #--------------------------------------------------------------------
    def setcomplex(self, true_false) :
        self.__cplx = true_false
    #--------------------------------------------------------------------
    @property
    def mtype(self) :
        nm = "Unknown"
        if self.iscomplex : nm += " with complex stiffnesses"
        return nm
    #--------------------------------------------------------------------
    def __str__(self) :
        """ for printing a Material instance"""
        chaine = f"Material '{self.name}' of type '{self.type}',\n"
        if self.rho is None :
            chaine += "<undefined mass density>,"
        else :
            chaine += f"with {1e-3*self.rho:.4f} mg/mm³ mass density,"
        return chaine
    #--------------------------------------------------------------------
    def prt_attributes(self,nbdec=3) :
        """ checking tool: printing all attributes 
            'nbdec' is the number of decimal digits """
        chaine = ""
        msg = "{}: {:."+str(nbdec)+"e}\n"
        for n in self.__attrNames :
            value = eval("self."+n)
            try : 
                chaine += msg.format(n,value)
            except :
                chaine += f"{n}: {value}\n"
        print(chaine[:-1])
    #--------------------------------------------------------------------
    def check(self) :
        if self.rho is not None and self.rho > 0.0 :
            return True,""
        else :
            return False, "\nUndefined mass density"
    #--------------------------------------------------------------------
    def tosave(self) :
        chaine = "Name: "+self.name+"\n"
        chaine += "Type: "+self.mtype+"\n"
        chaine += f"Density [mg/mm³]: {1e-3*self.rho:.6f}\n"
        return chaine
    #--------------------------------------------------------------------  
    def save(self,filePath) :
        f = open(filePath,"w",encoding="utf-8")
        f.write(self.tosave())
        f.close()
    #--------------------------------------------------------------------
    # DO NOT MODIFY THIS METHOD
    @property
    def type(self) : return type(self).__name__
#------------------------------------------------------------------------
# overloading the __setattr__ method to avoid unwanted attribute
# generation
    def __setattr__(self,name,value) :
        cls = self.type
        if cls == "Material" : # this protection is not inherited
            if name in Material.__attrNames :
                object.__setattr__(self,name,value) # Standard assignment
            else :
                msg = "{}: unautorized creation of '{}' attribute"
                print(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Fluid(Material) :
    """ homogeneous fluid
        'param' is a dictionary giving the parameters values,
        'units':    'SI'           -> kg/m³ ;   m/s;   Pa.
                    'mm_µs_mg_GPa' -> mg/mm³ ; mm/µs; GPa.
        'name' is a string."""
    #--------------------------------------------------------------------
    # Add this class in the list of the 'Material' subclasses
    Material.addsubclass("Fluid")
    #--------------------------------------------------------------------
    # List of the class attributes
    __attrNames = tuple(Material.attrNames() + ["_Fluid__"+n for n in \
                                                ("c","K","a")])
    #--------------------------------------------------------------------
    def __init__(self, param, name, units="SI") :
        Material.__init__(self, param, name, units)
        self.__c = None
        self.__a = None
        self.__K = None
        cplx = False
        SI_units = ( units != "mm_µs_mg_GPa" )
        for k in param.keys() :
            # Bulk Modulus [Pa - GPa]
            if "k" == k.lower() or "modul" in k.lower() :
                if SI_units : self.__K = param[k]
                else : self.__K = 1e9*param[k]
                cplx = np.abs(self.__K.imag) > \
                              np.abs(self.__K.real)*self.zero_num
                if not cplx :
                    self.__K = self.__K.real
            # Sound Speed [m/s - mm/µs]
            if "c" == k.lower() or "celerity" in k.lower() or\
               "speed" in k.lower() or "velocity" in k.lower() :
                if SI_units : self.__c = param[k]
                else : self.__c = 1e3*param[k]
            # attenuation coefficient []
            if "a" == k.lower() or "att" in k.lower() :
                self.__a = param[k]
                if self.__a > 1e-6 : # nonzero attenuation
                    cplx = True
        self.setcomplex(cplx)
        ok, msg = self.check()
        if not ok :
            self.prt("Fluid constructor error:\n"+msg)
    #--------------------------------------------------------------------
    @property
    def c(self) :
        """Returns complex sound speed = sqrt(K/rho)."""
        if self.__c is None : return None
        if self.iscomplex :
            return self.__c*np.sqrt( 1.0 + 1.0j*self.__a )
        else :
            return self.__c
    @property
    def real_c(self) :
        """Returns real sound speed = sqrt(Re(K)/rho)."""
        return self.__c
    @property
    def K(self) : return self.__K
    #--------------------------------------------------------------------
    def __str__(self) :
        if self.__c is not None :
            ch = f"\nwith {1e-3*self.__c:.4f} mm/µs speed of sound"
            if self.iscomplex :
                ch +=  "\n and attenuation coefficient " + \
                      f"{1e2*self.__a:.3f}%"
            ch += f"\n(bulk modulus {1e-9*self.__K:.3e} GPa)"
            return Material.__str__(self) + ch
        else :
            return Material.__str__(self) + \
                   "\nwith <undefined> speed of sound"
    #--------------------------------------------------------------------    
    def prt_attributes(self,nbdec=3) :
        """ checking tool: printing all attributes 
            'nbdec' is the number of decimal digits """
        chaine = ""
        msg = "{}: {:."+str(nbdec)+"e}\n"
        for n in self.__attrNames :
            value = eval("self."+n)
            try :
                chaine += msg.format(n,value)
            except :
                chaine += "{}: {}\n".format(n,value)
        print(chaine[:-1])
    #--------------------------------------------------------------------
    def check(self) :
        verif,msg = Material.check(self)
        if verif : # Mass density is defined            
            if self.__c is None :
                if self.__a is not None :
                    msg += "\nWarning: given attenuation coefficient\n" + \
                           "\tnot taken into account"
                if self.__K is None :
                    verif = False
                    msg += "\nUndefined sound speed"
                    msg += "\nUndefined bulk modulus"
                elif self.iscomplex : # Complex bulk modulus
                    if self.__K.real <= 0.0 :
                        verif = False
                        msg += "\nNonpositive real part of bulk modulus"
                    if self.__K.imag <= 0.0 :                        
                        verif = False
                        msg += "\nNonpositive imaginary part of " + \
                               "bulk modulus"
                    if verif : 
                        self.__c = np.sqrt(self.__K.real/self.rho)
                        self.__a = self.__K.imag/self.__K.real
                    else :
                        msg += "\nUndefined sound speed"
                else : # Real bulk modulus
                    if self.__K <= 0.0 :
                        verif = False
                        msg += "\nNonpositive bulk modulus"
                        msg += "\nUndefined sound speed"
                    else :
                        self.__c = np.sqrt(self.__K/self.rho)
                        self.__a = 0.0
            else : # The sound speed is defined
                if self.__c <= 0.0 :
                    verif = False
                    msg += "\nNonpositive sound speed"
                else :
                    if self.__a is None :
                        self.__a = 0.0
                    else :
                        if self.__a < 0.0 :
                            verif = False
                            msg += "\nNegative attenuation"
                        elif self.__a > self.zero_num : 
                            self.setcomplex(True)
                        else :
                            self.__a = 0.0
                if verif :
                    K_cal = self.rho*self.__c**2
                    if self.iscomplex : 
                        K_cal = K_cal*(1.0+1.0j*self.__a)
                    if self.__K is not None :
                        # redundant information
                        if np.abs(K_cal-self.__K) > \
                           self.zero_num*np.abs(self.__K) :
                            msg += "\nWarning: inconsistent informa" + \
                                   "tion\n\trecalculated bulk modulus"
                    self.__K = K_cal                    
            return verif,msg
        else :
            if self.__c is None :
                msg += "\nUndefined sound speed"
            elif self.__c <= 0.0 :
                msg += "\nNonpositive sound speed"
            if self.__K is None :
                msg += "\nUndefined bulk modulus"
            elif self.iscomplex :
                if self.__K.real <= 0.0 :
                    msg += "\nNonpositive real part of bulk modulus"
                if self.__K.imag <= 0.0 :
                    msg += "\nNonpositive imaginary part of bulk modulus"
            else :
                if self.__K <= 0.0 :
                    msg += "\nNonpositive bulk modulus"                
            return False,msg
    #--------------------------------------------------------------------
    @property
    def mtype(self) :
        nm = "Fluid"
        if self.iscomplex : nm += " with complex stiffnesses"
        return nm
    #--------------------------------------------------------------------
    def tosave(self) :
        ch = Material.tosave(self) + \
            f"Speed of sound [mm/µs]: {1e-3*self.__c:.7e}\n"
        if self.iscomplex :
            ch += f"Attenuation coefficient: {self.__a:.5f}\n"
        return ch
#------------------------------------------------------------------------    
# overloading the __setattr__ method to avoid unwanted attribute
# generation
    def __setattr__(self,name,value) :
        cls = self.type
        if cls == "Fluid" : # this protection is not inherited
            if name in Fluid.__attrNames :
                object.__setattr__(self,name,value) # Standard assignment
            else :
                msg = "{}: unautorized '{}' attribute"
                print(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class AnisotropicElasticSolid(Material) :
    """ Anisotropic elastic medium (the most general anisotropy)
        'param' is a dictionary giving the parameters values,
        'units':    'SI'           -> kg/m³ ;   m/s;   Pa.
                    'mm_µs_mg_GPa' -> mg/mm³ ; mm/µs; GPa.
        'name' is a string."""
    #--------------------------------------------------------------------
    # Add this class in the list of the 'Material' subclasses
    Material.addsubclass("AnisotropicElasticSolid") 
    # List of the class attributes
    __attrNames = tuple(Material.attrNames() +\
            ["_AnisotropicElasticSolid__"+n for n in ["stiffnesses", \
                        "lol", "lom", "lon", "mol", "mom", "mon", \
                        "nol", "nom", "non", "compliances"] ])
    #--------------------------------------------------------------------
    __stiffnessIndexes = []
    for i in range(1,7) :
        for j in range(i,7) :
            __stiffnessIndexes.append(str(i)+str(j))
    __stiffnessIndexes = tuple(__stiffnessIndexes)
    __ij2n = ((-1,-1,-1,-1,-1,-1,-1),(-1,0,1,2,3,4,5),(-1,1,6,7,8,9,10),\
              (-1,2,7,11,12,13,14),(-1,3,8,12,15,16,17),\
              (-1,4,9,13,16,18,19),(-1,5,10,14,17,19,20) )
    __Voigt2index = {1:(0,0), 2:(1,1), 3:(2,2), \
                     4:(1,2), 5:(0,2), 6:(0,1)}
    #--------------------------------------------------------------------
    @staticmethod
    def n(i,j) : return AnisotropicElasticSolid.__ij2n[i][j]
    @staticmethod
    def __c2n(i,j) :
        """Voigt subscript. i=0,1,2 & j=0,1,2"""
        if i == j : return i+1
        return 7-i-j
    @staticmethod
    def __n2c(n) :
        """Voigt subscript"""
        voigt_dict = AnisotropicElasticSolid.__Voigt2index
        if n in voigt_dict.keys() :
            return voigt_dict[n]
        print("Error in AnisotropicElasticSolid.__n2c")
    #--------------------------------------------------------------------
    def __init__(self, param, name, units="SI") :
        Material.__init__(self, param, name, units)
        self.__stiffnesses = []
        cplx = False
        SI_units = ( units != "mm_µs_mg_GPa" )
        for c in self.__stiffnessIndexes : # [Pa - GPa]
            Cij = "c"+c
            cherche = True
            for k in param.keys() :
                if Cij == k.lower() :
                    if SI_units : val_cij = param[k]
                    else : val_cij = 1e9*param[k]
                    if not cplx and val_cij is not None:
                        threshold = self.zero_num*np.abs(val_cij.real)
                        cplx = np.abs(val_cij.imag) > threshold
                    self.__stiffnesses.append(val_cij)
                    cherche = False
                    break
            if cherche :
                self.prt(f"Constructor error: '{Cij}' not found")
                self.__stiffnesses.append(None)
        self.setcomplex(cplx)
        ok, msg = self.check()
        # Compliance computation :
        self.__compliances = 21*[None]
        if not ok :
            self.__lol = None
            self.__lom = None
            self.__lon = None
            self.__mol = None
            self.__mom = None
            self.__mon = None
            self.__nol = None
            self.__nom = None
            self.__non = None
            return # Stop computation
        mat_stiff = np.array([ \
                        [self.c(i,j) for j in range(1,7)] \
                        for i in range(1,7)])
        mat_comp = np.linalg.inv(mat_stiff)
        mat_comp[:,3:] *= 0.5
        mat_comp[3:,:] *= 0.5
        for i in range(6) :
            for j in range(i,6) :
                self.__compliances[self.n(i+1,j+1)] = mat_comp[i,j]
        # Diamond products
        self.__lol = np.array([[self.c(1,1),self.c(1,6),self.c(1,5)],\
                               [self.c(1,6),self.c(6,6),self.c(5,6)],\
                               [self.c(1,5),self.c(5,6),self.c(5,5)]])
        self.__lom = np.array([[self.c(1,6),self.c(1,2),self.c(1,4)],\
                               [self.c(6,6),self.c(2,6),self.c(4,6)],\
                               [self.c(5,6),self.c(2,5),self.c(4,5)]])
        self.__lon = np.array([[self.c(1,5),self.c(1,4),self.c(1,3)],\
                               [self.c(5,6),self.c(4,6),self.c(3,6)],\
                               [self.c(5,5),self.c(4,5),self.c(3,5)]])
        self.__mol = self.__lom.transpose()
        self.__mom = np.array([[self.c(6,6),self.c(2,6),self.c(4,6)],\
                               [self.c(2,6),self.c(2,2),self.c(2,4)],\
                               [self.c(4,6),self.c(2,4),self.c(4,4)]])
        self.__mon = np.array([[self.c(5,6),self.c(4,6),self.c(3,6)],\
                               [self.c(2,5),self.c(2,4),self.c(2,3)],\
                               [self.c(4,5),self.c(4,4),self.c(3,4)]])
        self.__nol = self.__lon.transpose()
        self.__nom = self.__mon.transpose()
        self.__non = np.array([[self.c(5,5),self.c(4,5),self.c(3,5)],\
                               [self.c(4,5),self.c(4,4),self.c(3,4)],\
                               [self.c(3,5),self.c(3,4),self.c(3,3)]])
    #--------------------------------------------------------------------
    @property
    def lol(self) : return self.__lol
    @property
    def lom(self) : return self.__lom
    @property
    def lon(self) : return self.__lon
    @property
    def mol(self) : return self.__mol
    @property
    def mom(self) : return self.__mom
    @property
    def mon(self) : return self.__mon
    @property
    def nol(self) : return self.__nol
    @property
    def nom(self) : return self.__nom
    @property
    def non(self) : return self.__non 
    #--------------------------------------------------------------------
    def c(self,i,j) :
        """Stiffness at Voigt indexes i and j."""
        return self.__stiffnesses[self.n(i,j)]
    #--------------------------------------------------------------------
    def setC(self,i,j,newCij) :
        if newCij >= 0.0 :
            self.__stiffnesses[self.n(i,j)] = newCij
        else :
            print("Incorrect value '{}' for Cij".format(newCij))
    #--------------------------------------------------------------------
    def s(self,i,j) :
        """Compliance at Voigt indexes i and j."""
        return self.__compliances[self.n(i,j)]
    #--------------------------------------------------------------------
    @property
    def stiffness_matrix(self) :
        return  np.array([ [self.c(i,j) for j in range(1,7)] \
                           for i in range(1,7) ])
    @property
    def compliance_matrix(self) :
        return  np.array([ [self.s(i,j) for j in range(1,7)] \
                           for i in range(1,7) ])
    @property
    def fourth_order_stiffness_tensor(self) :
        voigt_dict = AnisotropicElasticSolid.__Voigt2index
        if self.iscomplex :
            EST = np.empty( (3,3,3,3), dtype=np.complex128 )
        else :
            EST = np.empty( (3,3,3,3), dtype=np.float64 )
        for i in range(1,7) :
            i1,i2 = voigt_dict[i]        
            for j in range(i,7) :
                j1,j2 = voigt_dict[j]  
                cij = self.c(i,j)
                EST[i1,i2,j1,j2] = cij
                EST[i2,i1,j1,j2] = cij
                EST[i1,i2,j2,j1] = cij
                EST[i2,i1,j2,j1] = cij
                EST[j1,j2,i1,i2] = cij
                EST[j2,j1,i1,i2] = cij
                EST[j1,j2,i2,i1] = cij
                EST[j2,j1,i2,i1] = cij
        return EST
    #--------------------------------------------------------------------
    def __str__(self) :
        if self.iscomplex : # Complex-valued stiffnesses
            chaine = ""
            for i in range(1,7) :
                chaine += "\n" + (i-1)*15*" "
                for j in range(i,7) :
                    cij = self.__stiffnesses[self.n(i,j)]
                    if isinstance(cij,float) or \
                       isinstance(cij,complex) :
                        chaine += f"{1e-9*cij.real:8.3f}"
                        imag = f"{1e-9*cij.imag:.3f}j"
                        if imag[0] in ("-","+") :
                            chaine += imag
                        else :
                            chaine += "+"+imag
                    else :
                        chaine += " < ??? >+< ??? >j "
        else : # Real-valued stiffnesses
            chaine = ""
            for i in range(1,7) :
                chaine += "\n" + (i-1)*8*" "
                for j in range(i,7) :
                    cij = self.__stiffnesses[self.n(i,j)]
                    if isinstance(cij,float) :
                        chaine += f"{1e-9*cij:8.3f}"
                    else :
                        chaine += " < ??? >"
        return Material.__str__(self)+"\nof stiffnesses [GPa]"+chaine
    #--------------------------------------------------------------------    
    def prt_attributes(self,nbdec=3) :
        """ checking tool: printing all attributes 
            'nbdec' is the number of decimal digits """
        chaine = ""
        msg = "{}: {:."+str(nbdec)+"e}\n"
        for n in self.__attrNames :
            value = eval("self."+n)
            try :
                if isinstance(value,float) :
                    chaine += msg.format(n,value)
                elif isinstance(value,list) :
                    chaine += "{} :\n".format(n)
                    ft = "\t{:."+str(nbdec)+"e}\n"
                    for v in value :
                        chaine += ft.format(v)
                else :
                    chaine += "{}: {}\n".format(n,value)
            except :
                chaine += "{}: {}\n".format(n,value)
        print(chaine[:-1])
    #--------------------------------------------------------------------        
    def check(self) :
        """Calculation of the missing parameters and consistency check"""
        verif,msg = Material.check(self)
        if self.iscomplex : # Complex stiffnesses
            for i,cij in enumerate(self.__stiffnesses) :        
                if not isinstance(cij,float) and \
                   not isinstance(cij,complex) :
                    verif = False
                    msg += "\nc"+self.__stiffnessIndexes[i]+" undefined"
        else :  # Real stiffnesses
            for i,cij in enumerate(self.__stiffnesses) :        
                if not isinstance(cij,float) :
                    verif = False
                    msg += "\nc"+self.__stiffnessIndexes[i]+" undefined"
        # All principal minors have to be strictly positive:
        if verif :
            for n in range(65,128) :
                ib = [ i for i,e in enumerate(list(bin(n)[-6:]),1) \
                       if e == "1" ]
                M = np.array( [ [ self.c(i,j).real for j in ib ] \
                                for i in ib ] )
                if np.linalg.det(M) <= 0 :
                    verif = False
                    msg += f"\nPrincipal minor {ib} is not strictly " + \
                            "positive"
        return verif,msg 
    #--------------------------------------------------------------------          
    @property
    def mtype(self) :
        nm = "Anisotropic Elastic Medium"
        if self.iscomplex : nm += " with complex stiffnesses"
        return nm
    #--------------------------------------------------------------------
    def tosave(self) :
        texte = Material.tosave(self)+"Stiffnesses [GPa]\n"
        for c,Cij in zip(self.__stiffnessIndexes,self.__stiffnesses) :
            texte += f"c{c}: {1e-9*Cij:.7e}\n"
        return texte
    #--------------------------------------------------------------------
    def diamond(self,u,v) :
        return u[0]*(v[0]*self.lol+v[1]*self.lom+v[2]*self.lon)+\
               u[1]*(v[0]*self.mol+v[1]*self.mom+v[2]*self.mon)+\
               u[2]*(v[0]*self.nol+v[1]*self.nom+v[2]*self.non)
    #--------------------------------------------------------------------    
    def newC(self,P) :
        """ returns a dictionary containing the stiffnesses after a
            change of basis.
            The change-of-basis orthogonal matrix P contains the new
            colum-vectors expressed in the old basis."""
        nc = dict()
        for I2 in range(1,7) :
            i2,j2 = self.__n2c(I2)
            for J2 in range(I2,7) :
                key = "c{}{}".format(I2,J2)
                k2,m2 = self.__n2c(J2)
                value = 0.0
                for i1 in range(3) :
                    for j1 in range(3) :
                        I1 = self.__c2n(i1,j1)
                        for k1 in range(3) :
                            for m1 in range(3) :
                                J1 = self.__c2n(k1,m1)
                                value += self.c(I1,J1)*P[i1,i2]*\
                                         P[j1,j2]*P[k1,k2]*P[m1,m2]
                nc[key] = value
        return nc
    #--------------------------------------------------------------------
    def rotate(self, phi, theta=None, alpha=None, diamond=False, \
               verbose=False) :
        """ change the orientation of the crystallographic axes in the
            xyz axes (in degrees). At the beginning x = #1, y = #2,
            z = #3.
            First, 'phi' is the angle between axis #1 and x-axis, before
            tilting axis #3.
            Second, 'theta' is the angle between axis #3 and z-axis
            after tilting (in [0,90°]) and 'phi' is the azimuth of the
            projection of axis #3 on xy-plane (undefined if theta = 0.0,
            the axis of rotation is directed by (sin(phi),-cos(phi))."""
        if round(phi) == 0 :
            name = self.name
        else :
            name = self.name+f"_rotated-by-phi{int(round(phi)):d}"
        phi = np.radians(phi)
        if theta is None or round(theta) == 0 :
            c1,s1 = 1.0,0.0
            c2,s2 = np.cos(phi),-np.sin(phi)
            c3,s3 = 1.0,0.0
        else :
            if alpha is None : alpha = 0.0
            if phi == 0.0 :
                name += f"_rotated-by-phi{int(round(phi)):d}"
            name += f"-theta{int(round(theta))%360:d}" + \
                    f"-alpha{int(round(alpha))%360:d}"
            theta = np.radians(theta)
            alpha = np.radians(alpha)
            c1,s1 = np.cos(alpha),np.sin(alpha)
            c2,s2 = np.cos(alpha-phi),np.sin(alpha-phi)
            c3,s3 = np.cos(theta),np.sin(theta)
        Ux = np.array([c1*c2*c3+s1*s2,c1*s2*c3-s1*c2,c1*s3])
        Uy = np.array([s1*c2*c3-c1*s2,s1*s2*c3+c1*c2,s1*s3])
        Uz = np.array([-c2*s3,-s2*s3,c3])
        Pt = np.array([Ux,Uy,Uz])
        P = Pt.transpose() # change-of-basis matrix
        #print("Determinant of P: {:.3f}".format(np.linalg.det(P)))
        #print(P.round(3))
        #print(P.dot(Pt))
        if diamond : # With diamond product
            LoL = Pt.dot(self.diamond(Ux,Ux)).dot(P)
            nc = {"c11":LoL[0,0],"c16":LoL[0,1],"c15":LoL[0,2],\
                                 "c66":LoL[1,1],"c56":LoL[1,2],\
                                                "c55":LoL[2,2]}
            LoM = Pt.dot(self.diamond(Ux,Uy)).dot(P)
            nc["c12"] = LoM[0,1] ; nc["c14"] = LoM[0,2] 
            nc["c26"] = LoM[1,1] ; nc["c46"] = LoM[1,2] 
            nc["c25"] = LoM[2,1] ; nc["c45"] = LoM[2,2] 
            LoN = Pt.dot(self.diamond(Ux,Uz)).dot(P)
            nc["c13"] = LoN[0,2]
            nc["c36"] = LoN[1,2]
            nc["c35"] = LoN[2,2] 
            MoM = Pt.dot(self.diamond(Uy,Uy)).dot(P)
            nc["c22"] = MoM[1,1]
            nc["c24"] = MoM[1,2]
            nc["c44"] = MoM[2,2]
            MoN = Pt.dot(self.diamond(Uy,Uz)).dot(P)
            nc["c23"] = MoN[1,2]
            nc["c34"] = MoN[2,2]
            NoN = Pt.dot(self.diamond(Uz,Uz)).dot(P)
            nc["c33"] = NoN[2,2]
        else :
            nc = self.newC(P)  # With standard change of basis
        nc["rho"] = self.rho
        if verbose :
            msg = "+++ nc in AnisotropicElasticSolid.rotate:"
            for k,v in nc.items() :
                msg += "\n\t{}\t~ {:.3e}".format(k,v)
            print(msg)
        return AnisotropicElasticSolid(nc,name)
#------------------------------------------------------------------------    
# overloading the __setattr__ method to avoid unwanted attribute
# generation
    def __setattr__(self,name,value) :
        cls = self.type
        if cls == "AnisotropicElasticSolid" : # This protection is not
                                              # inherited
            if name in AnisotropicElasticSolid.__attrNames :
                object.__setattr__(self,name,value) # Standard assignment
            else :
                msg = "{}: unautorized '{}' attribute"
                print(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class IsotropicElasticSolid(Material) :
    """Isotropic elastic medium  
            'param' is a dictionary giving the parameters values,
            'units':    'SI'           -> kg/m³ ;   m/s;   Pa.
                        'mm_µs_mg_GPa' -> mg/mm³ ; mm/µs; GPa.
            'name' is a string.
       Required parameters :
          mass density [kg/m³] + (cL, cT) sound speeds (real values)
                                 or (lambda, mu) Lame's coefficients
                                 or two of (E, G, nu)
                                               Young's modulus
                                               shear modulus
                                               Poisson's ratio
                                 or two of (c11, c12, c44) stiffnesses.                                   
    """
    #--------------------------------------------------------------------
    # Add this class in the list of the 'Material' subclasses
    Material.addsubclass("IsotropicElasticSolid")
    #-------------------------------------------------------------------- 
    # List of the class attributes
    __attrNames = tuple(Material.attrNames() + \
                  ["_IsotropicElasticSolid__"+c for c in \
                   ["cL","cT","c11","c44","c12","E","nu",\
                    "msg","verif","modified"]\
                  ])
    __Ynames = ("young modulus", "young's modulus", "module d'young", \
                "module de young", "e", "y")
    __Pnames = ("poisson ratio", "poisson's ratio", "nu", \
                "coefficient de poisson")
    __Gnames = ("c44", "c66", "shear modulus", \
                "lame's second parameter",\
                "deuxieme coefficient de lame", "mu", "g", \
                "second coefficient de lame")
    __Lnames = ("c12","lame's first parameter", "lambda",\
                "premier coefficient de lame")
    #--------------------------------------------------------------------
    def __init__(self, param, name, units="SI") :
        Material.__init__(self, param, name, units)
        self.__cL,self.__cT,self.__c11,self.__c44,self.__c12 = 5*[None]
        self.__E,self.__nu,self.__verif,self.__msg = 4*[None]
        cplx = False
        SI_units = ( units != "mm_µs_mg_GPa" )
        for NOM in param.keys() :
            nom = NOM.lower().strip()
            if "cl" == nom or "cl " in nom : # c_L [m/s - mm/µs]
                if SI_units : self.__cL = param[NOM]
                else : self.__cL = 1e3*param[NOM]
            elif "ct" == nom or "ct " in nom : # c_T [m/s - mm/µs]
                if SI_units : self.__cT = param[NOM]
                else : self.__cT = 1e3*param[NOM]
            elif "c11" in nom : # c11 [Pa - GPa]
                if SI_units : val = param[NOM]
                else : val = 1e9*param[NOM]
                if not cplx and isinstance(val, complex) :
                    threshold = self.zero_num*np.abs(val.real)
                    cplx = np.abs(val.imag) > threshold
                self.__c11 = val
            elif sum( [n == nom for n in self.__Ynames] ) >= 1 :
                # Young's Modulus E [Pa - GPa]
                if SI_units : val = param[NOM]
                else : val = 1e9*param[NOM]
                if not cplx and isinstance(val, complex) :
                    threshold = self.zero_num*np.abs(val.real)
                    cplx = np.abs(val.imag) > threshold
                self.__E = val
            elif sum( [n == nom for n in self.__Pnames] ) >= 1 :
                # Poisson's Ratio nu
                val = param[NOM]
                if not cplx and isinstance(val, complex) :
                    threshold = self.zero_num*np.abs(val.real)
                    cplx = np.abs(val.imag) > threshold
                self.__nu = val                 
            elif sum( [n == nom for n in self.__Gnames] ) >= 1 :
                # Shear Modulus G = c44 [Pa - GPa]              
                if SI_units : val = param[NOM]
                else : val = 1e9*param[NOM]
                if not cplx and isinstance(val, complex) :
                    threshold = self.zero_num*np.abs(val.real)
                    cplx = np.abs(val.imag) > threshold
                self.__c44 = val                  
            elif sum( [n == nom for n in self.__Lnames] ) >= 1 :
                # c12 [Pa - GPa]              
                if SI_units : val = param[NOM]
                else : val = 1e9*param[NOM]
                if not cplx and isinstance(val, complex) :
                    threshold = self.zero_num*np.abs(val.real)
                    cplx = np.abs(val.imag) > threshold
                self.__c12 = val
        self.setcomplex(cplx)
        self.__modified = True
        verif,msg = self.check()
        if not verif : self.prt(msg)
    #--------------------------------------------------------------------
    @property
    def cL(self) :
        """Complex longitudinal speed = sqrt(c11/rho)."""
        if self.__c11 is None or self.rho is None :
            return None
        else :
            return np.sqrt(self.__c11/self.rho)
    @property
    def real_cL(self) :
        """Real longitudinal speed = sqrt(Re(c11)/rho)."""
        if self.__c11 is None or self.rho is None :
            return None
        else :
            return np.sqrt(self.__c11.real/self.rho)
    @property
    def cT(self) : 
        """Complex transverse speed = sqrt(c44/rho)."""
        if self.__c44 is None or self.rho is None :
            return None
        else : return np.sqrt(self.__c44/self.rho)
    @property
    def real_cT(self) : 
        """Real transverse speed = sqrt(Re(c44)/rho)."""
        if self.__c44 is None or self.rho is None :
            return None
        else : return np.sqrt(self.__c44.real/self.rho)
    @property
    def c11(self) : return self.__c11
    @property
    def c12(self) : return self.__c12
    @property
    def c44(self) : return self.__c44
    @property
    def E(self) : return self.__E
    @property
    def nu(self) : return self.__nu
    @property
    def Lame_coefficients(self) : return self.c12,self.c44
    @property
    def fourth_order_stiffness_tensor(self) :
        ani_mat = self.export().export()
        return ani_mat.fourth_order_stiffness_tensor
    #--------------------------------------------------------------------        
    def __str__(self) :
        chaine =  Material.__str__(self)
        try : cL = f"{1e-3*self.__cL:.3f}"
        except : cL = "???"
        try : cT = f"{1e-3*self.__cT:.3f}"
        except : cT = "???"
        chaine += f"\nSpeeds [mm/µs]: cL ~ {cL} and cT ~ {cT}"        
        try : E = f"{1e-9*self.__E:.1f}"
        except : E = "???"
        try : nu = f"{1.0*self.__nu:.3f}"
        except : nu = "???"
        chaine += f"\nYoung's modulus: {E} GPa ; Poisson's ratio: {nu}"
        try : c12 = f"{1e-9*self.__c12:.1f}"
        except : c12 = "???"
        try : c44 = f"{1e-9*self.__c44:.1f}"
        except : c44 = "???"
        chaine += f"\nLamé's coefficients: {c12} GPa and {c44} GPa"
        return chaine
    #--------------------------------------------------------------------
    def prt_attributes(self,nbdec=3) :
        """ checking tool: printing all attributes 
            'nbdec' is the number of decimal digits """
        chaine = ""
        msg = "{}: {:."+str(nbdec)+"e}\n"
        for n in self.__attrNames :
            value = eval("self."+n)
            try :
                if isinstance(value,float) :
                    chaine += msg.format(n,value)
                else :
                    chaine += "{}: {}\n".format(n,value)
            except :
                chaine += "{}: {}\n".format(n,value)
        print(chaine[:-1])
    #--------------------------------------------------------------------
    def check(self) :
        """Calculation of the missing parameters and consistency check"""
        # to avoid useless check...
        if not self.__modified : return self.__verif,self.__msg
        self.__modified = False
        # Modification has been done. check...
        verif,msg = Material.check(self)
        self.prt("================\nIsotropicMaterial.check:\n" + \
                 "=== At the beginning ===")
        if self.VERBOSE() : self.prt_attributes(7)
        if verif :
            rho = self.rho
        else :
            rho = None
        if self.__c12 is not None and self.__c44 is not None :
                                                # Lamé's coefficients
            c12, c44 = self.__c12, self.__c44
            c11 = c12 + 2*c44
            if self.__c11 is None :
                self.__c11 = c11
            elif self.__c11 != c11 :
                self.__c11 = c11
                msg += "\nIncoherence on c11 calculated from c12" + \
                       " and c44...corrected"
                self.__modified = True
            if rho is not None :
                cL = np.sqrt(c11.real/rho)
                if self.__cL is None or self.iscomplex :
                    self.__cL = cL
                elif self.__cL != cL :
                    self.__cL = cL
                    msg += "\nIncoherence on cL calculated from" + \
                           " c12 and c44...corrected"
                    self.__modified = True
                cT = np.sqrt(c44.real/rho)
                if self.__cT is None or self.iscomplex :
                    self.__cT = cT
                elif self.__cT != cT :
                    self.__cT = cT
                    msg += "\nIncoherence on cT calculated from c12" + \
                           " and c44...corrected"
                    self.__modified = True
            E = c44*(3*c12+2*c44)/(c12+c44)
            if self.__E is None :
                self.__E = E
            elif self.__E != E :
                self.__E = E
                msg += "\nIncoherence on E calculated from c12 and" + \
                       " c44...corrected"
                self.__modified = True
            nu = 0.5*c12/(c12+c44)
            if self.__nu is None :
                self.__nu = nu
            elif self.__nu != nu :
                self.__nu = nu
                msg += "\nIncoherence on nu calculated from c12" + \
                       " and c44...corrected"
                self.__modified = True 
        elif self.__c11 is not None and self.__c44 is not None :
                                                        # Stiffnesses
            c11, c44 = self.__c11, self.__c44
            c12 = c11-2*c44
            if self.__c12 is None :
                self.__c12 = c12
            elif self.__c12 != c12 :
                self.__c12 = c12
                msg += "\nIncoherence on c12 calculated from c11" + \
                       " and c44...corrected"
                self.__modified = True
            if rho is not None :
                cL = np.sqrt(c11.real/rho)
                if self.__cL is None or self.iscomplex :
                    self.__cL = cL
                elif self.__cL != cL :
                    self.__cL = cL
                    msg += "\nIncoherence on cL calculated from c11" + \
                           " and c44...corrected"
                    self.__modified = True
                cT = np.sqrt(c44.real/rho)
                if self.__cT is None or self.iscomplex :
                    self.__cT = cT
                elif self.__cT != cT :
                    self.__cT = cT
                    msg += "\nIncoherence on cT calculated from c11" + \
                           " and c44...corrected"
                    self.__modified = True
            E = c44*(3*c12+2*c44)/(c12+c44)
            if self.__E is None :
                self.__E = E
            elif self.__E != E :
                self.__E = E
                msg += "\nIncoherence on E calculated from c11 and" + \
                       " c44...corrected"
                self.__modified = True
            nu = 0.5*c12/(c12+c44)
            if self.__nu is None :
                self.__nu = nu
            elif self.__nu != nu :
                self.__nu = nu
                msg += "\nIncoherence on nu calculated from c11 and" + \
                       " c44...corrected"
                self.__modified = True 
        elif self.__c11 is not None and self.__c12 is not None :
                                                        # Stiffnesses
            c11, c12 = self.__c11, self.__c12
            c44 = 0.5*(c11-c12)
            if self.__c44 is None :
                self.__c44 = c44
            elif self.__c44 != c44 :
                self.__c44 = c44
                msg += "\nIncoherence on c44 calculated from c11 and" + \
                       " c12...corrected"
                self.__modified = True
            if rho is not None :
                cL = np.sqrt(c11.real/rho)
                if self.__cL is None or self.iscomplex :
                    self.__cL = cL
                elif self.__cL != cL :
                    self.__cL = cL
                    msg += "\nIncoherence on cL calculated from c11" + \
                           " and c12...corrected"
                    self.__modified = True
                cT = np.sqrt(c44.real/rho)
                if self.__cT is None or self.iscomplex :
                    self.__cT = cT
                elif self.__cT != cT :
                    self.__cT = cT
                    msg += "\nIncoherence on cT calculated from c11" + \
                           " and c12...corrected"
                    self.__modified = True
            E = c44*(3*c12+2*c44)/(c12+c44)
            if self.__E is None :
                self.__E = E
            elif self.__E != E :
                self.__E = E
                msg += "\nIncoherence on E calculated from c11 and" + \
                       " c12...corrected"
                self.__modified = True
            nu = 0.5*c12/(c12+c44)
            if self.__nu is None :
                self.__nu = nu
            elif self.__nu != nu :
                self.__nu = nu
                msg += "\nIncoherence on nu calculated from c11 and" + \
                       " c12...corrected"
                self.__modified = True
        elif self.__E is not None and self.__nu is not None :
                                # Young's modulus and Poisson's ratio
            E, nu = self.__E, self.__nu
            c12 = nu*E/(1+nu)/(1-2*nu)
            if self.__c12 is None :
                self.__c12 = c12
            elif self.__c12 != c12 :
                self.__c12 = c12
                msg += "\nIncoherence on c12 calculated from E and" + \
                       "nu...corrected"
                self.__modified = True
            c44 = 0.5*E/(1+nu)
            if self.__c44 is None :
                self.__c44 = c44
            elif self.__c44 != c44 :
                self.__c44 = c44
                msg += "\nIncoherence on c44 calculated from E and" + \
                       " nu...corrected"
                self.__modified = True
            c11 = c12 + 2*c44
            if self.__c11 is None :
                self.__c11 = c11
            elif self.__c11 != c11 :
                self.__c11 = c11
                msg += "\nIncoherence on c11 calculated from E and" + \
                       " nu...corrected"
                self.__modified = True
            if rho is not None :
                cL = np.sqrt(c11.real/rho)
                if self.__cL is None or self.iscomplex :
                    self.__cL = cL
                elif self.__cL != cL :
                    self.__cL = cL
                    msg += "\nIncoherence on cL calculated from E" + \
                           " and nu...corrected"
                    self.__modified = True
                cT = np.sqrt(c44.real/rho)
                if self.__cT is None or self.iscomplex :
                    self.__cT = cT
                elif self.__cT != cT :
                    self.__cT = cT
                    msg += "\nIncoherence on cT calculated from E" + \
                           " and nu...corrected"
                    self.__modified = True
        elif self.__E is not None and self.__c44 is not None :
                                                # Young's and Shear moduli
            E, c44 = self.__E, self.__c44
            nu = 0.5*E/c44 - 1
            if self.__nu is None :
                self.__nu = nu
            elif self.__nu != nu :
                self.__nu = nu
                msg += "\nIncoherence on nu calculated from E" + \
                           " and G...corrected"
                self.__modified = True
            c12 = nu*E/(1+nu)/(1-2*nu)
            if self.__c12 is None :
                self.__c12 = c12
            elif self.__c12 != c12 :
                self.__c12 = c12
                msg += "\nIncoherence on c12 calculated from E" + \
                           " and G...corrected"
                self.__modified = True
            c11 = c12 + 2*c44
            if self.__c11 is None :
                self.__c11 = c11
            elif self.__c11 != c11 :
                self.__c11 = c11
                msg += "\nIncoherence on c11 calculated from E" + \
                           " and G...corrected"
                self.__modified = True
            if rho is not None :
                cL = np.sqrt(c11.real/rho)
                if self.__cL is None or self.iscomplex :
                    self.__cL = cL
                elif self.__cL != cL :
                    self.__cL = cL
                    msg += "\nIncoherence on cL calculated from E" + \
                           " and G...corrected"
                    self.__modified = True
                cT = np.sqrt(c44.real/rho)
                if self.__cT is None or self.iscomplex :
                    self.__cT = cT
                elif self.__cT != cT :
                    self.__cT = cT
                    msg += "\nIncoherence on cT calculated from E" + \
                           " and G...corrected"
                    self.__modified = True
        elif self.__cL is not None and self.__cT is not None and \
             rho is not None :
            # Given speeds, with non-zero mass density
            self.setcomplex(False)
            cL, cT = self.__cL, self.__cT
            c11 = rho*cL**2
            if self.__c11 is None :
                self.__c11 = c11
            elif self.__c11 != c11 :
                self.__c11 = c11
                msg += "\nIncoherence on c11 calculated from cL and" + \
                           " cT...corrected"
                self.__modified = True
            c44 = rho*cT**2
            if self.__c44 is None :
                self.__c44 = c44
            elif self.__c44 != c44 :
                self.__c44 = c44
                msg += "\nIncoherence on c44 calculated from cL and" + \
                           " cT...corrected"
                self.__modified = True
            c12 = c11-2*c44
            if self.__c12 is None :
                self.__c12 = c12
            elif self.__c12 != c12 :
                self.__c12 = c12
                msg += "\nIncoherence on c12 calculated from cL and" + \
                           " cT...corrected"
                self.__modified = True
            E = c44*(3*c12+2*c44)/(c12+c44)
            if self.__E is None :
                self.__E = E
            elif self.__E != E :
                self.__E = E
                msg += "\nIncoherence on E calculated from cL and" + \
                           " cT...corrected"
                self.__modified = True
            nu = 0.5*c12/(c12+c44)
            if self.__nu is None :
                self.__nu = nu
            elif self.__nu != nu :
                self.__nu = nu
                msg += "\nIncoherence on nu calculated from cL and" + \
                           " cT...corrected"
                self.__modified = True
        else :
            verif = False
            msg += "\nIncomplete elastic data"
        if isinstance(self.__c12,float) :
            if self.__c12 <= 0 :
                msg += "\nA Lamé's coefficient has to be strictly" + \
                           " positive!"
                verif = False
        if isinstance(self.__c44,float) :
            if self.__c44 <= 0 :
                msg += "\nA Lamé's coefficient has to be strictly" + \
                           " positive!"
                verif = False
        self.prt(msg)
        self.__verif,self.__msg = verif,msg
        self.prt("=== At the end ===")
        if self.VERBOSE() : self.prt_attributes(7)
        self.prt("================")
        return verif,msg
    #--------------------------------------------------------------------
    @property
    def mtype(self) :
        nm = "Isotropic Elastic Medium"
        if self.iscomplex : nm += " with complex stiffnesses"
        return nm
    #--------------------------------------------------------------------    
    def tosave(self) :
        texte = Material.tosave(self) + \
                "Stiffnesses [GPa] (Lamé's Coefficients)\n"
        for n,v in zip(["c12","c44"],[self.c12,self.c44]) :
            texte += f"{n}: {1e-9*v:.7e}\n"
        return texte
    #--------------------------------------------------------------------
    def export(self) :
        """returns a transversely isotropic material with the
           same stiffnesses"""
        TIE = TransverselyIsotropicElasticSolid
        return TIE({"rho":self.rho,"c11":self.c11,"c12":self.c12,\
                    "c13":self.c12,"c33":self.c11,"c44":self.c44,\
                    "c66":self.c44},self.name)
#------------------------------------------------------------------------    
# overloading the __setattr__ method to avoid unwanted attribute
# generation
    def __setattr__(self,name,value) :
        cls = self.type
        if cls == "IsotropicElasticSolid" : # This protection is not inherited
            if name in IsotropicElasticSolid.__attrNames :
                object.__setattr__(self,name,value) # Standard assignment
            else :
                msg = "{}: unautorized '{}' attribute"
                print(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class TransverselyIsotropicElasticSolid(Material) :
    """ Transversely isotropic elastic medium (axis #3 is the axis
        of symmetry)
            'param' is a dictionary giving the parameters values,
            'units':    'SI'           -> kg/m³ ;   m/s;   Pa.
                        'mm_µs_mg_GPa' -> mg/mm³ ; mm/µs; GPa.
            'name' is a string."""
    #--------------------------------------------------------------------
    # Add this class in the list of the 'Material' subclasses
    Material.addsubclass("TransverselyIsotropicElasticSolid")
    #--------------------------------------------------------------------
    # List of the class attributes
    __attrNames = tuple(Material.attrNames() + \
                  ["_TransverselyIsotropicElasticSolid__"+c for c in \
                   ["cPV", "cSV", "cSH", "cPH", "c11", "c12", "c33", \
                    "c13", "c44", "c66", "modified", "verif", "msg"]\
                  ])
    #--------------------------------------------------------------------
    def __init__(self, param, name, units="SI") :
        SI_units = ( units != "mm_µs_mg_GPa" )
        Material.__init__(self, param, name, units)
        self.__cPV,self.__cSV,self.__cPH,self.__cSH,self.__c11 = 5*[None]
        self.__c12,self.__c66,self.__c33,self.__c13,self.__c44 = 5*[None]
        self.__verif,self.__msg = None,None
        for NOM in param.keys() :
            nom = NOM.lower()
            if "cpv" == nom or "cpv " in nom : # c_PV [m/s - mm/µs]
                if SI_units : self.__cPV = param[nom]
                else : self.__cPV = 1e3*param[nom]
            elif "cph" == nom or "cph " in nom : # c_PH [m/s - mm/µs]
                if SI_units : self.__cPH = param[nom]
                else : self.__cPH = 1e3*param[nom]
            elif "csv" == nom or "csv " in nom : # c_SV [m/s - mm/µs]
                if SI_units : self.__cSV = param[nom]
                else : self.__cSV = 1e3*param[nom]
            elif "csh" == nom or "csh " in nom : # c_SH [m/s - mm/µs]
                if SI_units : self.__cSH = param[nom]
                else : self.__cSH = 1e3*param[nom]
            elif "c11" in nom : # c11 [Pa - GPa]
                if SI_units : self.__c11 = param[nom]
                else : self.__c11 = 1e9*param[nom]
            elif "c12" in nom : # c12 [Pa - GPa]
                if SI_units : self.__c12 = param[nom]
                else : self.__c12 = 1e9*param[nom]
            elif "c33" in nom : # c33 [Pa - GPa]
                if SI_units : self.__c33 = param[nom]
                else : self.__c33 = 1e9*param[nom]
            elif "c13" in nom : # c13 [Pa - GPa]
                if SI_units : self.__c13 = param[nom]
                else : self.__c13 = 1e9*param[nom]
            elif "c44" in nom : # c44 [Pa - GPa]
                if SI_units : self.__c44 = param[nom]
                else : self.__c44 = 1e9*param[nom]
            elif "c66" in nom : # c66 [Pa - GPa]
                if SI_units : self.__c66 = param[nom]
                else : self.__c66 = 1e9*param[nom]
        self.__modified = True
        verif,msg = self.check()
        if not verif : self.prt(msg)
    #--------------------------------------------------------------------
    @property
    def c11(self) : return self.__c11
    @property
    def c12(self) : return self.__c12
    @property
    def c66(self) : return self.__c66
    @property
    def c33(self) : return self.__c33
    @property
    def c13(self) : return self.__c13
    @property
    def c44(self) : return self.__c44
    @property
    def cPV(self) : return self.__cPV
    @property
    def cSV(self) : return self.__cSV
    @property
    def cPH(self) : return self.__cPH
    @property
    def cSH(self) : return self.__cSH
    @property
    def fourth_order_stiffness_tensor(self) :
        ani_mat = self.export()
        return ani_mat.fourth_order_stiffness_tensor
    #--------------------------------------------------------------------        
    def __str__(self) :
        chaine =  Material.__str__(self)+"\nSpeeds of waves [km/s]: "
        for n,c in zip(["PV","SV","PH","SH"],\
                       [self.__cPV,self.__cSV,self.__cPH,self.__cSH]) :
            try : chaine += f"c{n} ~ {1e-3*c:.4f} mm/µs; "
            except : chaine += f"c{n} ~ ??? ; "
        chaine += "\nStiffnesses [GPa] :\n\t" + \
                  f"c11 ~ {1e-9*self.__c11:.1f}, " + \
                  f"c12 ~ {1e-9*self.__c12:.1f}, " + \
                  f"c66 ~ {1e-9*self.__c66:.1f}\n\t" + \
                  f"c33 ~ {1e-9*self.__c33:.1f}, " + \
                  f"c13 ~ {1e-9*self.__c13:.1f}, " + \
                  f"c44 ~ {1e-9*self.__c44:.1f}"
        return chaine
    #--------------------------------------------------------------------    
    def prt_attributes(self,nbdec=3) :
        """ checking tool: printing all attributes 
            'nbdec' is the number of decimal digits """
        chaine = ""
        msg = "{}: {:."+str(nbdec)+"e}\n"
        for n in self.__attrNames :
            value = eval("self."+n)
            try :
                if isinstance(value,float) :
                    chaine += msg.format(n,value)
                else :
                    chaine += f"{n}: {value}\n"
            except :
                chaine += f"{n}: {value}\n"
        print(chaine[:-1])
    #--------------------------------------------------------------------
    def check(self) :
        """Calculation of missing parameters and consistency check"""
        if not self.__modified : return self.__verif,self.__msg
        self.__modified = False
        verif,msg = Material.check(self)
        if self.VERBOSE() : self.prt_attributes(1)
        vrf = True
        if self.__c33 is None :
            vrf = False
            msg += "\nc33 missing"
        if self.__c13 is None :
            vrf = False
            msg += "\nc13 missing"
        if self.__c44 is None :
            vrf = False
            msg += "\nc44 missing"
        if self.__c11 is not None and self.__c12 is not None : 
            c11, c12 = self.__c11, self.__c12
            c66 = 0.5*(c11-c12)
            if self.__c66 is None :
                self.__c66 = c66
            elif self.__c66 != c66 :
                self.__c66 = c66
                msg += "\nIncoherence on c66 calculated frome c11" + \
                       " and c12...corrected"
                self.__modified = True
        elif self.__c11 is not None and self.__c66 is not None :
                                                    # self.__c12 is None
            c11, c66 = self.__c11, self.__c66
            c12 = c11 - 2*c66
            self.__c12 = c12
        elif self.__c12 is not None and self.__c66 is not None :
                                                    # self.__c11 is None
            c12, c66 = self.__c12, self.__c66
            c11 = c12 + 2*c66
            self.__c11 = c11
        else :
            vrf = False
        if not vrf :
            verif = False
            msg += "\nIncomplete data"
        elif verif :
            rho = self.rho
            self.__cPV = np.sqrt(self.__c33/rho)
            self.__cSV = np.sqrt(self.__c44/rho)
            self.__cPH = np.sqrt(self.__c11/rho)
            self.__cSH = np.sqrt(self.__c66/rho)
        self.__verif,self.__msg = verif,msg
        if self.VERBOSE() : self.prt_attributes(3)
        return verif,msg
    #--------------------------------------------------------------------           
    @property
    def mtype(self) :
        nm = "Transversely Isotropic Elastic Medium"
        if self.iscomplex : nm += " with complex stiffnesses"
        return nm
    #--------------------------------------------------------------------    
    def tosave(self) :
        texte = Material.tosave(self)+"Stiffnesses [GPa] \n"
        for n,v in zip(["c12","c66","c33","c13","c44"],\
                       [self.__c12,self.__c66,self.__c33,\
                        self.__c13,self.__c44]) :
            texte += f"{n}: {1e-9*v:.7e}\n"
        return texte
    #--------------------------------------------------------------------
    # New in Version 3.7:
    def get_anisotropic_parameters(self,verbose=False) :
        params = {"rho":self.rho, "c11":self.c11, "c12":self.c12, \
                  "c13":self.c13, "c22":self.c11, "c23":self.c13, \
                  "c33":self.c33, "c44":self.c44, "c55":self.c44, \
                  "c66":self.c66}
        for j in [4,5,6] :
            for i in range(1,j) :
                params["c{}{}".format(i,j)] = 0.0
        if verbose :
            msg = "+++ params in TransverselyIsotropicElastic" + \
                  "Solid.__anisoPrm:"
            for k,v in params.items() :
                msg += "\n\t{}\t~ {:.3e}".format(k,v)
            print(msg)
        return params
    #--------------------------------------------------------------------
    @property
    def __anisoPrm(self) :
        return self.get_anisotropic_parameters()
    #--------------------------------------------------------------------
    def export(self) :
        """returns a 'AnisotropicElasticSolid' instance with the same
           stiffnesses."""
        AE = AnisotropicElasticSolid       
        return AE(self.__anisoPrm,self.name)
    #--------------------------------------------------------------------
    def rotate(self,theta,alpha) :
        """rotates the transversely isotropic material: the position of
           the chrystallographic symmetry axis #3 is characterized by
           the angles alpha and theta (spherical coordinates in degrees)
        """
        if abs(theta%180) < 1e-3 : # unchanged
            params = self.__anisoPrm
            return TransverselyIsotropicElasticSolid(params,name) # copy
        aniso = self.export()
        return aniso.rotate(0.0,theta,alpha)
#------------------------------------------------------------------------    
# overloading the __setattr__ method to avoid unwanted attribute
# generation
    def __setattr__(self,name,value) :
        cls = self.type
        if cls == "TransverselyIsotropicElasticSolid" :
            # This protection is not inherited
            if name in TransverselyIsotropicElasticSolid.__attrNames :
                object.__setattr__(self,name,value) # Standard assignment
            else :
                msg = "{}: unautorized '{}' attribute"
                print(msg.format(self.__class__,name))
                raise AttributeError()
        else : object.__setattr__(self,name,value)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#+++++++++++ Basic Tests ++++++++++++++++++++++++++++++++++++++++++++++++
if __name__ == "__main__" :
    print("=== Test of the 'Material' superclass ===")
    m0 = Material({"rho":2.5},"mat n°0")
    print(m0)
    print(m0.tosave())
    print("=== Test dof the 'Fluid' subclass ===")
    m1 = Fluid({"rho":1.3,"c": 340},"mat n°1")
    print(m1)
    print(m1.tosave())
    list_mat = [m1]
# reading/writing in text format :
    m2 = IsotropicElasticSolid({"rho":2700.0,"cL": 5600,"cT": 3600}, \
                                "iso #2")
    print(m2)
    print(m2.tosave())
    list_mat.append(m2)
    for ordre,materiau in enumerate(list_mat,1) :
        print(15*"*"+\
              " reading/writing in text format #{} ".format(ordre)+\
              15*"*")
        texte = materiau.tosave()
        print(texte)
        newm = ImportMaterialFromText(texte)
        print(newm)
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
