import numpy as np

from pyttn.ttnpp import product_operator_complex, ttn_complex, SOP_complex, system_modes
try:
    from pyttn.ttnpp import product_operator_real, ttn_real, SOP_real

except ImportError:
    product_operator_real = None
    ttn_real = None 
    SOP_real = None

def product_operator(h, sysinf, *args, dtype = np.complex128, **kwargs):
    """Function for constructing a product_operator

    :param h: The product operator representation of the Hamiltonian
    :type h: sOP or sPOP or sNBO_real or sNBO_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type \*args: Variable length list of arguments. See product_operator_real/product_operator_complex for options
    :param dtype: The internal variable type for the product operator.(Default: np.complex128) 
    :type dtype: {np.float64, np.complex128}, optional
    :type \*\*kwargs: Additional keyword arguments. See product_operator_real/product_operator_complex for options
    """
    if(dtype == np.complex128 or product_operator_real is None):
        return product_operator_complex(h, sysinf, *args, **kwargs)
    else:
        return product_operator_real(h, sysinf, *args, **kwargs)

