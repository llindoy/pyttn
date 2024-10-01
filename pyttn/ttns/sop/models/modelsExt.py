from pyttn._pyttn import AIM_real, AIM_complex, electronic_structure_real, electronic_structure_complex, TFIM_real, TFIM_complex
from pyttn._pyttn import spin_boson_generic_complex, spin_boson_generic_real, spin_boson_star_complex, spin_boson_star_real, spin_boson_chain_complex, spin_boson_chain_real
import numpy as np

def AIM(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return AIM_complex(*args)
    elif(dtype == np.float64):
        return AIM_real(*args)
    else:
        raise RuntimeError("Invalid dtype for AIM")

def electronic_structure(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return electronic_structure_complex(*args)
    elif(dtype == np.float64):
        return electronic_structure_real(*args)
    else:
        raise RuntimeError("Invalid dtype for electronic_structure")


def TFIM(*args, dtype = np.complex128):
    if(dtype == np.complex128):
        return TFIM_complex(*args)
    elif(dtype == np.float64):
        return TFIM_real(*args)
    else:
        raise RuntimeError("Invalid dtype for TFIM")


def init_sb_type(sbt1, sbt2, *args, dtype=None):
    if(args):
        if args[-1].dtype == np.complex128 or dtype == np.complex128:
            return sbt1(*args)
        else:
            return sbt2(*args)
    else:
        if(dtype == np.complex128):
            return sbt1()
        elif(dtype == np.float64):
            return sbt2()
        else:
            raise RuntimeError("Invalid dtype for spin boson")

def spin_boson(*args, geom="star", dtype=np.complex128):
    if(geom == "star"):
        return init_sb_type(spin_boson_star_complex, spin_boson_star_real, *args, dtype = dtype)
    elif geom == "chain":
        return init_sb_type(spin_boson_chain_complex, spin_boson_chain_real, *args, dtype = dtype)
    elif geom == "generic":
        return init_sb_type(spin_boson_generic_complex, spin_boson_generic_real, *args, dtype = dtype)
    else:
        raise RuntimeError("Invalid geom for spin boson.")

