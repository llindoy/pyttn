from pyttn._pyttn import product_operator_real, product_operator_complex, ttn_real, ttn_complex, SOP_real, SOP_complex, system_modes
import numpy as np

def product_operator(h, sysinf, dtype = np.complex128, *args, **kwargs):
    if(dtype == np.complex128):
        return product_operator_complex(h, sysinf, *args, **kwargs)
    else:
        return product_operator_real(h, sysinf, *args, **kwargs)

