import numpy as np


class matrix_element_dtype:
    r"""A class defining the general interface for the pybind11 wrappers generated for the matrix_element object.  These wrapper classes are
    :class:`matrix_element_complex` and :class:`matrix_element_real` (with the real variant only present if the pybind11 wrapper has been built with support
    for real valued TTNs.  These classes provide a set of functions for evaluating matrix elements of operators with respect to single and multiset TTNs

    :param *args: A variable length list of arguments. Valid options are

        - none - in this case we call the default constructor of the matrix element class to construct an empty matrix element.
        - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.
        - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Two TTNs with the same topology defining the size of the bra and ket respectively.
        - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **H** (:class:`sop_operator_dtype` or :class:`ms_sop_operator_dtype`)- Two TTNs and a Hamiltonian operator with the same topology defining the size of the bra, ket and operator respectively.

    :type *args: [Arguments (variable number and type)]
    :param **kwargs: A dictionary containing optional input arguments.

        - **nbuffers** (int, optional) - The number of buffers to allocate.
        - **use_capacity** (bool, optional) - Whether or not to use the capacity of the TTNs to determine the size of the buffers.
    :type **kwargs: dict(Arguments (variable number and type))
    """

    def __init__(self, *args, dtype=np.complex128, **kwargs):
        raise RuntimeError(
            r"The ttn_dtype class is not constructable.  This class is present to provide cleaner documentation for the pybind11 classes."
        )

    def assign(self, o):
        r"""Assign the value of this matrix element from another matrix element

        :param o: The other matrix_element object
        :type o: matrix_element_dtype
        """
        pass

    def clear(self):
        r"Clear and deallocate all internal buffers of the matrix element"
        pass

    def resize(self, *args, **kwargs):
        r"""Resize the matrix element to fit the required wavefunctions

        :param *args: A variable length list of arguments. Valid options are

            - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - A TTN to define the topology and size of buffers needed to evaluate the tensor network.
            - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Two TTNs with the same topology defining the size of the bra and ket respectively.

        :type *args: [Arguments (variable number and type)]
        :param **kwargs: A dictionary containing optional input arguments.

            - **nbuffers** (int, optional) - The number of buffers to allocate.
            - **use_capacity** (bool, optional) - Whether or not to use the capacity of the TTNs to determine the size of the buffers.
        :type **kwargs: dict(Arguments (variable number and type))
        """
        pass

    def __call__(self, *args, use_sparsity=True):
        r"""Function for evaluating the inner product of tensor network states with matrix elements

        :param *args: A variable length list of arguments. See below for a list of valid arguments.
        :type *args: [Arguments (variable number and type)]
        :param use_sparsity: Whether or not to exploit sparsity in the evaluation of the matrix elements. (Default: True)
        :type use_sparsity: bool, optional

        :*args options:
        Valid options for the argument list depend on the type of quantity we are evaluating.

        :math:`\langle A | A \rangle`:

            - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the inner product of the ttn with itself.

        :math:`\langle A | O | A \rangle`:

            - **op** (:class:`site_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the single site operator op.
            - **op** (:class:`site_operator_dtype`), **mode** (int), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the single site operator op acting on mode mode.
            - **op** (list[:class:`site_operator_dtype`]), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.
            - **op** (list[:class:`site_operator_dtype`]), **modes** (list[int]), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.  Each site operator acts on the corresponding mode stored in the modes list.
            - **op** (:class:`product_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.
            - **op** (:class:`ms_sop_operator_dtype` or :class:`ms_sop_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of sum-of-product operator op.

        :math:`\langle A | B \rangle`:

            - **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the overlad .

        :math:`\langle A | O | B \rangle`:

            - **op** (:class:`site_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the matrix element of the single site operator op.
            - **op** (:class:`site_operator_dtype`), **mode** (int), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the single site operator op acting on mode mode.
            - **op** (list[:class:`site_operator_dtype`]), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.
            - **op** (list[:class:`site_operator_dtype`]), **modes** (list[int]), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.  Each site operator acts on the corresponding mode stored in the modes list.
            - **op** (:class:`product_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of site operator op.
            - **op** (:class:`ms_sop_operator_dtype` or :class:`ms_sop_operator_dtype`), **A** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`), **B** (:class:`ttn_dtype` or :class:`ms_ttn_dtype`) - Compute the expectation value of the product of sum-of-product operator op.
        """
        pass


"""
        .def("__call__", [](matel& o, siteop& op, const _ttn& A, const _ttn& B){return o(op, A, B);})
        .def("__call__", [](matel& o, siteop& op, size_t mode, const _ttn& A, const _ttn& B){return o(op, mode, A, B);})
        //inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const _ttn& A, const _ttn& B){return o(ops, A, B);})
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const std::vector<size_t>& modes, const _ttn& A, const _ttn& B){return o(ops, modes, A, B);})
        .def("__call__", [](matel& o, prodop& ops, const _ttn& A, const _ttn& B){return o(ops, A, B);})
        .def("__call__", [](matel& o, sop_op& sop, const _ttn& A, const _ttn& B){return o(sop, A, B);})
        .def(
              "__call__", 
              [](matel& o, const _msttn& A, bool use_sparsity){return o(A, use_sparsity);}, 
              py::arg(), py::arg("use_sparsity")=true
            )
        .def(
              "__call__", 
              [](matel& o, siteop& op, size_t mode, const _msttn& A, bool use_sparsity){return o(op, mode, A, use_sparsity);}, 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparsity")=true
            )
        .def(
              "__call__", 
              [](matel& o, std::vector<siteop>& ops, const std::vector<size_t>& modes, const _msttn& A, bool use_sparsity){return o(ops, modes, A, use_sparsity);}, 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparsity")=true
            )
        .def("__call__", [](matel& o, ms_sop_op& sop, const _msttn& A){return o(sop, A);})
        .def("__call__", [](matel& o, const _msttn& A, const _msttn& B){return o(A, B);})
        .def("__call__", [](matel& o, siteop& op, const _msttn& A){return o(op, A);})
        .def("__call__", [](matel& o, siteop& op, const _msttn& A, const _msttn& B){return o(op, A, B);})
        .def("__call__", [](matel& o, siteop& op, size_t mode, const _msttn& A, const _msttn& B){return o(op, mode, A, B);})
        //inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const _msttn& A, const _msttn& B){return o(ops, A, B);})
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const std::vector<size_t>& modes, const _msttn& A, const _msttn& B){return o(ops, modes, A, B);})
        .def("__call__", [](matel& o, ms_sop_op& sop, const _msttn& A, const _msttn& B){return o(sop, A, B);});


"""
