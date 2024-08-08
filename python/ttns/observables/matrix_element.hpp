#ifndef PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP
#define PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP

#include <ttns_lib/observables/matrix_element.hpp>
#include "../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_matrix_element(py::module &m, const std::string& label)
{
    using namespace ttns;


    using siteop = site_operator<T, linalg::blas_backend>;
    using matel = matrix_element<T, linalg::blas_backend>;
    using _ttn = ttn<T, linalg::blas_backend>;
    using _msttn = ms_ttn<T, linalg::blas_backend>;
    using sop_op = sop_operator<T, linalg::blas_backend>;
    using ms_sop_op = multiset_sop_operator<T, linalg::blas_backend>;

    py::class_<matel>(m, (std::string("matrix_element_")+label).c_str())
        .def(py::init())
        .def(py::init<const _ttn&, size_t, bool>(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _ttn&, const _ttn&, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _ttn&, const sop_op&, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _ttn&, const _ttn&, const sop_op&, size_t, bool>(), py::arg(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)

        .def(py::init<const _msttn&, size_t, bool>(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _msttn&, const _msttn&, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _msttn&, const ms_sop_op&, size_t, bool>(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)
        .def(py::init<const _msttn&, const _msttn&, const ms_sop_op&, size_t, bool>(), py::arg(), py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity") = false)

        .def("assign", [](matel& op, const matel& o){return op=o;})
        .def("__copy__", [](const matel& o){return matel(o);})
        .def("__deepcopy__", [](const matel& o, py::dict){return matel(o);}, py::arg("memo"))
        .def("clear", &matel::clear)

        .def(
              "resize", 
              [](matel& o, const _ttn& A, size_t nbuffers, bool use_capacity){o.resize(A, nbuffers, use_capacity);}, 
              py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity")=false
            )
        .def(
              "resize", 
              [](matel& o, const _ttn& A, const _ttn& B, size_t nbuffers, bool use_capacity){o.resize(A, B, nbuffers, use_capacity);}, 
              py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity")=false
            )
        .def(
              "__call__", 
              [](matel& o, const _ttn& A, bool use_sparsity){return o(A, use_sparsity);}, 
              py::arg(), py::arg("use_sparsity")=true
            )
        .def(
              "__call__", 
              [](matel& o, siteop& op, size_t mode, const _ttn& A, bool use_sparsity){return o(op, mode, A, use_sparsity);}, 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparsity")=true
            )
        .def(
              "__call__", 
              [](matel& o, std::vector<siteop>& ops, const std::vector<size_t>& modes, const _ttn& A, bool use_sparsity){return o(ops, modes, A, use_sparsity);}, 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparsity")=true
            )
        .def("__call__", [](matel& o, sop_op& sop, const _ttn& A){return o(sop, A);})
        .def("__call__", [](matel& o, const _ttn& A, const _ttn& B){return o(A, B);})
        .def("__call__", [](matel& o, siteop& op, const _ttn& A){return o(op, A);})
        .def("__call__", [](matel& o, siteop& op, const _ttn& A, const _ttn& B){return o(op, A, B);})
        .def("__call__", [](matel& o, siteop& op, size_t mode, const _ttn& A, const _ttn& B){return o(op, mode, A, B);})
        //inline T operator()(one_body_operator<T, backend>& op, const state_type& bra, const state_type& ket)
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const _ttn& A, const _ttn& B){return o(ops, A, B);})
        .def("__call__", [](matel& o, std::vector<siteop>& ops, const std::vector<size_t>& modes, const _ttn& A, const _ttn& B){return o(ops, modes, A, B);})
        .def("__call__", [](matel& o, sop_op& sop, const _ttn& A, const _ttn& B){return o(sop, A, B);})

        .def(
              "resize", 
              [](matel& o, const _msttn& A, size_t nbuffers, bool use_capacity){o.resize(A, nbuffers, use_capacity);}, 
              py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity")=false
            )
        .def(
              "resize", 
              [](matel& o, const _msttn& A, const _msttn& B, size_t nbuffers, bool use_capacity){o.resize(A, B, nbuffers, use_capacity);}, 
              py::arg(), py::arg(), py::arg("nbuffers") = 1, py::arg("use_capacity")=false
            )
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

        //inline typename std::enable_if<std::is_base_of<op_base, op_type>::value, std::vector<T>&>::type operator()(std::vector<site_operator<T, backend>>& op, size_type mode, const state_type& bra, const state_type& ket, std::vector<T>& res)
        //inline std::vector<T>& operator()(std::vector<std::vector<site_operator<T, backend>>& ops, const std::vector<size_type>& modes, const state_type& bra, const state_type& ket, std::vector<T>& res)
}

void initialise_matrix_element(py::module& m);

#endif  //PYTHON_BINDING_TTNS_MATRIX_ELEMENTS_HPP


