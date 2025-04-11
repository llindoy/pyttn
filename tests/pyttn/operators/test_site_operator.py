"""
py::class_<siteop>(m, (std::string("site_operator_") + label).c_str())
        .def(py::init())
        .def(py::init<const siteop &>())

        .def(py::init<const dmat &>())
        .def(py::init<const spmat &>())
        .def(py::init<const diagmat &>())

        .def(py::init<const dmat &, size_t>())
        .def(py::init<const spmat &, size_t>())
        .def(py::init<const diagmat &, size_t>())

        .def(py::init<const sOP &, const system_modes &, bool>(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<const sOP &, const system_modes &, const opdict &, bool>(), py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)

        .def("initialise", static_cast<void (siteop::*)(const sOP &, const system_modes &, bool)>(&siteop::initialise), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("initialise", static_cast<void (siteop::*)(const sOP &, const system_modes &, const opdict &, bool)>(&siteop::initialise), py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)

        .def("complex_dtype", [](const siteop &)
             { return !std::is_same<T, real_type>::value; })

        .def("transpose", &siteop::transpose)
        .def("todense", &siteop::todense)

        .def("assign", [](siteop &op, const siteop &o)
             { return op = o; })
        .def("assign", [](siteop &op, const ident &o)
             { return op = o; })
        .def("assign", [](siteop &op, const dmat &o)
             { return op = o; })
        .def("assign", [](siteop &op, const spmat &o)
             { return op = o; })
        .def("assign", [](siteop &op, const diagmat &o)
             { return op = o; })

        .def("bind", [](siteop &op, const ident &o)
             { return op.bind(o); })
        .def("bind", [](siteop &op, const dmat &o)
             { return op.bind(o); })
        .def("bind", [](siteop &op, const spmat &o)
             { return op.bind(o); })
        .def("bind", [](siteop &op, const diagmat &o)
             { return op.bind(o); })

        .def("__copy__", [](const siteop &o)
             { return siteop(o); })
        .def("__deepcopy__", [](const siteop &o, py::dict)
             { return siteop(o); }, py::arg("memo"))
        .def("size", &siteop::size)
        .def("is_identity", &siteop::is_identity)
        .def("is_resizable", &siteop::is_resizable)
        .def_property("mode", static_cast<size_t (siteop::*)() const>(&siteop::mode), [](siteop &o, size_t val)
                      { o.mode() = val; })

        .def("resize", &siteop::resize)
        .def("apply", static_cast<void (siteop::*)(const_matrix_ref, matrix_ref)>(&siteop::apply))
        .def("apply", static_cast<void (siteop::*)(const_matrix_ref, matrix_ref, real_type, real_type)>(&siteop::apply))
        .def("apply", static_cast<void (siteop::*)(const_vector_ref, vector_ref)>(&siteop::apply))
        .def("apply", static_cast<void (siteop::*)(const_vector_ref, vector_ref, real_type, real_type)>(&siteop::apply))
        .def("backend", [](const siteop &)
             { return backend::label(); });

def site_operator(
    *args, mode=None, optype=None, dtype=np.complex128, backend="blas", **kwargs
):
    Factory function for constructing a one site operator.

    :param *args: Variable length list of arguments. There are several valid options for the *args parameters.  If the optype variable is None the allowed options are

        - site_op (site_operator_real or site_operato_complex) - Construct a new site_operator object from the existing object
        - op (sOP), sysinf (system_modes) - Construct a new site_operator from the string operator and system information
        - op (sOP), sysinf (system_modes), opdict (operator_dictionary_real or operator_dictionary_complex) -  Construct a new site_operator from the string operator, system information and used defined operator dictionary.

        Otherwise, if the optype variable has been set then the valid arguments are determined by the specified optype see opsExt.py for details.

    :param mode: The mode the site operator is acting on. (Default: None)
    :type mode: int or None, optional
    :param optype: The type of the operator to be constructed. (Default: None)
    :type optype: {'identity', 'matrix', 'sparse_matrix', 'diagonal_matrix'} or None, optional
    :param dtype: The dtype to use for the site operator.  (Default: np.complex128)
    :type dtype: {np.float64, np.complex128}, optional
    :param backend: The computational backend to use for the product operator  (Default: "blas")
    :type backend: {"blas", "cuda"}, optional
    :param **kwargs: Additional keyword arguments. To construct the site_operator object
"""

from pyttn import ops
from pyttn import site_operator
from pyttn.linalg import vector, matrix
import pytest
import numpy as np
import os
from scipy.sparse import csr_matrix as spcsr


os.environ["OMP_NUM_THREADS"] = "1"


@pytest.mark.parametrize("N, mode", [(2, 0), (3, 4), (4, 2), (5, 6), (6, 0), (120, 2)])
def test_identity(N, mode):
    # identity constructor
    siteop = site_operator(ops.identity(N))
    assert siteop.is_identity()
    assert siteop.size() == N
    assert str(siteop) == "I_" + str(N)
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the identity matrix
    assert np.allclose(np.array(siteop.todense()), np.identity(N))

    # identity constructor with mode info
    siteop = site_operator(ops.identity(N), mode)
    assert siteop.is_identity()
    assert siteop.size() == N
    assert str(siteop) == "I_" + str(N)
    assert siteop.mode == mode

    def apply_test(op):
        # apply to vector
        m1 = np.random.uniform(size=(N)) * (1.0 + 0.0j)
        m2 = vector(m1)
        m3 = vector(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), m1)

    apply_test(siteop)

    siteopT = siteop.transpose()
    assert siteopT.is_identity()
    assert siteopT.size() == N
    assert str(siteopT) == "I_" + str(N)
    assert siteopT.mode == mode

    apply_test(siteopT)


@pytest.mark.parametrize(
    "mat, mode",
    [
        (np.array([[1, 2], [3, 4]], dtype=np.float64), 0),
        (np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float64), 12),
        (np.random.uniform(0, 1, size=(25, 25)), 5),
    ],
)
def test_matrix(mat, mode):
    siteop = site_operator(ops.matrix(mat))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), mat)

    siteop = site_operator(ops.matrix(mat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N)) * (1.0 + 0.0j)
        m2 = vector(m1)
        m3 = vector(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(mat, siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[1]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), mat.T)

    apply_test(mat.T, siteopT)


def test_csr():
    mode = 3
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    spmat = spcsr((data, indices, indptr), shape=(3, 3))
    mat = spmat.toarray()

    siteop = site_operator(ops.sparse_matrix(spmat, dtype=np.complex128))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), mat)

    siteop = site_operator(ops.sparse_matrix(spmat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N))
        m2 = vector(m1, dtype=np.complex128)
        m3 = vector(m1, dtype=np.complex128)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(mat, siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[0]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), mat.T)

    apply_test(mat.T, siteopT)


@pytest.mark.parametrize(
    "mat, mode",
    [
        (np.array([1, 2], dtype=np.float64), 0),
        (np.array([1, 2, 3], dtype=np.float64), 12),
        (np.random.uniform(0, 1, size=(25)), 5),
    ],
)
def test_diagonal(mat, mode):
    siteop = site_operator(ops.diagonal_matrix(mat))
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == 0

    # check that the matrix we have constructed is all close to the input dense matrix
    assert np.allclose(np.array(siteop.todense()), np.diag(mat))

    siteop = site_operator(ops.diagonal_matrix(mat), mode)
    assert not siteop.is_identity()
    assert siteop.size() == mat.shape[0]
    assert siteop.mode == mode

    def apply_test(M, op):
        N = M.shape[0]
        # apply to vector
        m1 = np.random.uniform(size=(N))
        m2 = vector(m1, dtype=np.complex128)
        m3 = vector(m1, dtype=np.complex128)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

        # apply to matrix
        m1 = np.random.uniform(size=(N, N)) * (1.0 + 0.0j)
        m2 = matrix(m1)
        m3 = matrix(m1)
        op.apply(m2, m3)
        assert np.allclose(np.array(m3), M @ m1)

    apply_test(np.diag(mat), siteop)

    siteopT = siteop.transpose()
    assert not siteopT.is_identity()
    assert siteopT.size() == mat.shape[0]
    assert siteopT.mode == mode
    assert np.allclose(np.array(siteopT.todense()), np.diag(mat).T)

    apply_test(np.diag(mat).T, siteopT)
