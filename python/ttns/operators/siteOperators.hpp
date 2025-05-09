/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP
#define PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP

#include <ttns_lib/operators/site_operators/matrix_operators.hpp>
#include <ttns_lib/operators/site_operators/site_operator.hpp>
#include "../../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, typename backend>
void init_site_operators(py::module &m, const std::string &label)
{
    using namespace ttns;
    using prim = ops::primitive<T, backend>;
    using ident = ops::identity<T, backend>;
    using dmat = ops::dense_matrix_operator<T, backend>;
    using spmat = ops::sparse_matrix_operator<T, backend>;
    using diagmat = ops::diagonal_matrix_operator<T, backend>;

    using size_type = typename prim::size_type;
    using real_type = typename prim::real_type;

    using matrix_type = linalg::matrix<T, backend>;
    using matrix_ref = typename prim::matrix_ref;
    using const_matrix_ref = typename prim::const_matrix_ref;
    using vector_ref = typename prim::vector_ref;
    using const_vector_ref = typename prim::const_vector_ref;

    using conv = linalg::pybuffer_converter<backend>;

    using siteop = site_operator<T, backend>;
    using opdict = operator_dictionary<T, backend>;
    // the base primitive operator type
    py::class_<siteop>(m, (std::string("site_operator_") + label).c_str())
        .def(py::init())
        .def(py::init<const siteop &>())
        .def(py::init<const ident &>())
        .def(py::init<const dmat &>())
        .def(py::init<const spmat &>())
        .def(py::init<const diagmat &>())

        .def(py::init<const ident &, size_t>())
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
        .def("todense", [](const siteop& op){return op.todense();})
        .def("todense", [](const siteop& op, const std::vector<size_type>& mode_dims){return op.todense(mode_dims);})

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
        .def("__str__", &siteop::to_string)
        .def("backend", [](const siteop &)
             { return backend::label(); });

    // the base primitive operator type
    py::class_<prim>(m, (std::string("primitive_") + label).c_str())
        .def("size", &prim::size)
        .def("size", &prim::size)
        .def("size", &prim::size)
        .def("is_identity", &prim::is_identity)
        .def("is_resizable", &prim::is_resizable)
        .def("resize", &prim::resize)
        .def("clone", &prim::clone)
        .def("transpose", &prim::transpose)
        .def("complex_dtype", [](const prim &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", &prim::to_string)
        .def("backend", [](const prim &)
             { return backend::label(); });

    // a type for storing a trivial representation of the identity operator
    py::class_<ident, prim>(m, (std::string("identity_") + label).c_str())
        .def(py::init())
        .def(py::init<size_type>())
        .def("complex_dtype", [](const ident &)
             { return !std::is_same<T, real_type>::value; });

    // a dense matrix representation of the operator
    py::class_<dmat, prim>(m, (std::string("matrix_") + label).c_str())
        .def(py::init())
        .def(py::init<matrix_type>())
        .def(py::init([](py::buffer &b)
                      {
                    linalg::matrix<T, backend> mat;
                    conv::copy_to_tensor(b, mat);
                    return dmat(mat); }))
        .def("complex_dtype", [](const dmat &)
             { return !std::is_same<T, real_type>::value; })
        .def("matrix", &dmat::mat);

    // a csr matrix representation of an operator
    using csr_type = linalg::csr_matrix<T, backend>;
    using index_type = typename csr_type::index_type;
    py::class_<spmat, prim>(m, (std::string("sparse_matrix_") + label).c_str())
        .def(py::init())
        .def(py::init<const std::vector<T> &, const std::vector<index_type> &, const std::vector<index_type> &, size_t>(), py::arg(), py::arg(), py::arg(), py::arg("ncols") = 0)
        .def(py::init<const csr_type &>())
        .def("complex_dtype", [](const spmat &)
             { return !std::is_same<T, real_type>::value; })
        .def("matrix", &spmat::mat);

    // a diagonal matrix representation of an operator
    using diag_type = linalg::diagonal_matrix<T, backend>;
    py::class_<diagmat, prim>(m, (std::string("diagonal_matrix_") + label).c_str())
        .def(py::init())
        .def(py::init<const diag_type &>())
        .def(py::init([](py::buffer &b)
                      {
                    diag_type mat;
                    conv::copy_to_diagonal_matrix(b, mat);
                    return diagmat(mat); }))
        .def(py::init<const std::vector<T> &>())
        .def(py::init<const std::vector<T> &, size_t>())
        .def(py::init<const std::vector<T> &, size_t, size_t>())
        .def(py::init<const linalg::tensor<T, 1> &>())
        .def(py::init<const linalg::tensor<T, 1> &, size_t>())
        .def(py::init<const linalg::tensor<T, 1> &, size_t, size_t>())
        .def("complex_dtype", [](const diagmat &)
             { return !std::is_same<T, real_type>::value; })
        .def("matrix", &diagmat::mat);
}

template <typename real_type, typename backend>
void initialise_site_operators(py::module &m)
{
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_site_operators<real_type, backend>(m, "real");
#endif
    init_site_operators<complex_type, backend>(m, "complex");
}

#endif // PYTHON_BINDING_TTNS_SITE_OPERATORS_HPP
