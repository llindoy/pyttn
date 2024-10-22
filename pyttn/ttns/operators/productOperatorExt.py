import numpy as np

from pyttn._pyttn import product_operator_complex, ttn_complex, SOP_complex, system_modes
try:
    from pyttn._pyttn import product_operator_real, ttn_real, SOP_real

except ImportError:
    product_operator_real = None
    ttn_real = None 
    SOP_real = None

def product_operator(h, sysinf, dtype = np.complex128, *args, **kwargs):
    if(dtype == np.complex128 or product_operator_real is None):
        return product_operator_complex(h, sysinf, *args, **kwargs)
    else:
        return product_operator_real(h, sysinf, *args, **kwargs)

