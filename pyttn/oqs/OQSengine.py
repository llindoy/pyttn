from . import BosonicBath, FermionicBath
from pyttn import system_modes, validate_SOP, liouville_space_superoperator
from pyttn import is_SOP, is_sSOP, is_operator_dictionary
from .exponential_fit_bath import *

def is_bath(b):
    return isinstance(b, FermionicBath) or isinstance(b, BosonicBath)

def setup_OQS_bath(b, sys, fit, opdict=None, Lopdict=None):
    if not is_bath(b):
        raise RuntimeError("Failed to set up bath type.  Invalid type.")

    dk = None
    zk = None
    try:
        dk, zk = b.expfit(fit)
        print(dk, zk)

    except:
        print("Failed to setup up OQS bath object.  Error when performing exponential bath fit.")
        raise

    #if this is a bosonic bath type 
    if isinstance(b, BosonicBath):
        Sl = liouville_space_superoperator(b.S, sys, opdict, Lopdict, 'L')
        Sr = liouville_space_superoperator(b.S, sys, opdict, Lopdict, 'R')
        print("SR")
        print(Sr)
        return ExpFitBosonicBath(Sl, Sr, dk, zk)
    else:
        Spl = liouville_space_superoperator(b.Sp, sys, opdict, Lopdict, 'L')
        Spr = liouville_space_superoperator(b.Sp, sys, opdict, Lopdict, 'R')
        Sml = liouville_space_superoperator(b.Sm, sys, opdict, Lopdict, 'L')
        Smr = liouville_space_superoperator(b.Sm, sys, opdict, Lopdict, 'R')
        return ExpFitFermionicBath(Spl, Spr, Sml, Smr, dk, zk)


class OQSEngine:
    #static function for ensuring that an operator is consistent with the system information object
    def __init__(self, sysinf, Hs, baths, opdict = None):
        self._Hs = None
        self._sys = None
        self._baths = None
        self._opdict = None
        self._contains_fermion = False

        self._Ls = None
        self._Lsys = None
        self._sysinfo = None

        self._Lopdict = None
        self._exp_baths = None
        self._observables = None

        self.initialise(sysinf, Hs, baths, opdict = opdict)

    def system_liouvillian(self):
        return self._Ls

    def initialise(self, sysinf, Hs, baths, opdict=None):
        _sysinf = system_modes(sysinf)

        self._contains_fermion = _sysinf.contains_fermion()
        self._Hs = validate_SOP(Hs, _sysinf)

        #handle the construction of the operator dictionary
        self._Lopdict = None
        if opdict is None:
            self._opdict = None
        elif is_operator_dictionary(opdict):
            self._opdict = opdict
        #handle the case when it is defined and is the correct type

        if isinstance(baths, list):
            self._baths = baths
        elif is_bath(baths):
            self._baths = [baths]
        else:
            raise RuntimeError("Invalid type passed as baths.")

        #now iterate over all of the bath objects and ensure that they are valid.  And if any
        #of the bath objects are sSOP objects convert them to SOP objects
        for b in self._baths:
            if not is_bath(b):
                raise RuntimeError("Cannot bind non-bath type to baths array.")
            if isinstance(b, FermionicBath):
                b.Sp = validate_SOP(b.Sp, _sysinf)
                b.Sm = validate_SOP(b.Sm, _sysinf)
            else:
                b.S = validate_SOP(b.S, _sysinf)
    
        #now construct a new composite mode object from the sysinf type
        self._sys = system_modes(_sysinf.as_combined_mode())

        self.setup_system_liouville_space()

    def setup_system_liouville_space(self):
        #setup the system operator
        self._Lsys = self._sys.liouville_space()

        import copy
        self._sysinfo = copy.deepcopy(self._Lsys)

        #now we generate the system Liouvillian operator
        self._Ls = liouville_space_superoperator(self._Hs, self._sys, self._opdict, self._Lopdict, '-')


    def fit_baths(self, fit_parameters):
        if isinstance(fit_parameters, list):
            if len(fit_parameters) != len(self._baths):
                raise RuntimeError("The list of bath fitting parameters is not the same length as the number of baths.")
        else:
            fit_parameters = [fit_parameters for i in range(len(self._baths))]

        #go through and construct the bath objects needed for the HEOM or Pseudomode style calculations
        self._exp_baths = []
        for b, fit in zip(self._baths, fit_parameters):
            print(fit)
            binfo = setup_OQS_bath(b, self._sys, fit, opdict=self._opdict, Lopdict = self._Lopdict)
            if binfo.is_fermionic:
                self._contains_fermion = True
            self._exp_baths.append(binfo)

        self._baths_fit = True

    def truncate_baths(self, truncation_parameters):
        if isinstance(truncation_parameters, list):
            if len(truncation_parameters) != len(self._baths):
                raise RuntimeError("The list of bath truncation parameters is not the same length as the number of baths.")
        else:
            truncation_parameters = [truncation_parameters for i in range(len(self._baths))]

        #now go through and truncate the baths.  If there s
        for b, trunc in zip(self._exp_baths, truncation_parameters):
            if trunc is None:
                b.truncate_modes()
            else:
                b.truncate_modes(trunc)

    def bath_mode_combination(self, mode_combination_parameters):
        if isinstance(mode_combination_parameters, list):
            if len(mode_combination_parameters) != len(self._baths):
                raise RuntimeError("The list of bath mode combination parameters is not the same length as the number of baths.")
        else:
            mode_combination_parameters = [mode_combination_parameters for i in range(len(self._baths))]

        for b, mode_comb in zip(self._exp_baths, mode_combination_parameters):
            if mode_comb is None:
                b.mode_combination()
            else:
                b.mode_combination(mode_comb)


    #def setup_system_bath_models(self):
    #    #we start by building the total sysinfo object which is going to be the system degrees of freedom augemented with the bath degrees of freedom


    def add_observable(self, observable):
        self._observables.append(validate_SOP(observable,self._sys))


    def bath_fitting_info(self, t, index = None):
        if index == None:
            ret = []
            for j in range(len(self._baths)):
                ret.append(self.bath_fitting_info(t, index=j))
            return ret
        else:
            return bath_fitting_quality(self._baths[index], self._exp_baths[index], t)


    def __str__(self):
        ret = 'system: ' + str(self._sys) + '\n'
        for b in self._exp_baths:
            ret += 'bath: ' + str(b) + '\n'
        return ret
