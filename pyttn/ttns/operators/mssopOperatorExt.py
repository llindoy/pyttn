from pyttn.ttnpp import multiset_sop_operator_complex
from pyttn.ttnpp import ms_ttn_complex
from pyttn.ttnpp import multiset_SOP_complex

try:
    from pyttn.ttnpp import multiset_sop_operator_real
    from pyttn.ttnpp import ms_ttn_real
    from pyttn.ttnpp import multiset_SOP_real

    __real_ttn_import = True

except ImportError:
    __real_ttn_import = False
    multiset_sop_operator_real = None
    ms_ttn_real = None
    multiset_SOP_real = None


# and attempt to import the cuda backend
try:
    from pyttn.ttnpp.cuda import (
        multiset_sop_operator_complex as multiset_sop_operator_complex_cuda,
    )
    from pyttn.ttnpp.cuda import ms_ttn_complex as ms_ttn_complex_cuda

    __cuda_import = True

    # and if we have imported real ttns we import the cuda versions
    if __real_ttn_import:
        from pyttn.ttnpp.cuda import (
            multiset_sop_operator_real as multiset_sop_operator_real_cuda,
        )
        from pyttn.ttnpp.cuda import ms_ttn_real as ms_ttn_real_cuda

    else:
        multiset_sop_operator_real_cuda = None
        ms_ttn_real_cuda = None

except ImportError:
    __cuda_import = False
    multiset_sop_operator_real_cuda = None
    multiset_sop_operator_complex_cuda = None
    ms_ttn_real_cuda = None
    ms_ttn_complex_cuda = None


def __multiset_sop_operator_blas(h, A, sysinf, *args, **kwargs):
    if not __real_ttn_import:
        if isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )
    else:
        if (
            isinstance(A, ms_ttn_real)
            and isinstance(h, multiset_SOP_real)
            and multiset_SOP_real is not None
        ):
            return multiset_sop_operator_real(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ms_ttn_complex) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )


def __multiset_sop_operator_cuda(h, A, sysinf, *args, **kwargs):
    if not __real_ttn_import:
        if isinstance(A, ms_ttn_complex_cuda) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex_cuda(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )
    else:
        if (
            isinstance(A, ms_ttn_real_cuda)
            and isinstance(h, multiset_SOP_real)
            and multiset_SOP_real is not None
        ):
            return multiset_sop_operator_real_cuda(h, A, sysinf, *args, **kwargs)
        elif isinstance(A, ms_ttn_complex_cuda) and isinstance(h, multiset_SOP_complex):
            return multiset_sop_operator_complex_cuda(h, A, sysinf, *args, **kwargs)
        else:
            raise RuntimeError(
                "Invalid argument for the creation of a multiset_sop_operator."
            )


def multiset_sop_operator(h, A, sysinf, *args, **kwargs):
    r"""Function for constructing the multiset hierarchical sum of product operator of a string operator

    :param h: The sum of product operator representation of the Hamiltonian
    :type h: multiset_SOP_real or multiset_SOP_complex
    :param A: A TTN object with defining the topology of output hierarchical SOP object
    :type A: ms_ttn_real or ms_ttn_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type *args: Variable length list of arguments. See multiset_sop_operator_complex/multiset_sop_operator_real for options
    :type **kwargs: Additional keyword arguments. See /multiset_sop_operator_complex/multiset_sop_operator_real for options
    """
    if len(args) > 0:
        if args[0].backend() != A.backend():
            raise RuntimeError(
                "Attempted to construct multiset_sop_operator with opdict but opdict backend is not compatible with ms_ttn backend."
            )

    if A.backend() == "blas":
        return __multiset_sop_operator_blas(h, A, sysinf, *args, **kwargs)
    elif __cuda_import and A.backend() == "cuda":
        return __multiset_sop_operator_cuda(h, A, sysinf, *args, **kwargs)
    else:
        raise RuntimeError("Invalid backend type for multiset_sop_operator")


def ms_sop_operator(h, A, sysinf, *args, **kwargs):
    r"""Function for constructing the multiset hierarchical sum of product operator of a string operator

    :param h: The sum of product operator representation of the Hamiltonian
    :type h: multiset_SOP_real or multiset_SOP_complex
    :param A: A TTN object with defining the topology of output hierarchical SOP object
    :type A: ms_ttn_real or ms_ttn_complex
    :param sysinf: The composition of the system defining the default dictionary to be considered for each node
    :type sysinf: system_modes
    :type *args: Variable length list of arguments. See multiset_sop_operator_complex/multiset_sop_operator_real for options
    :type **kwargs: Additional keyword arguments. See multiset_sop_operator_complex/multiset_sop_operator_real for options
    """
    return multiset_sop_operator(h, A, sysinf, *args, **kwargs)
