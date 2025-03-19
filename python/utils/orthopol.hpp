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

#ifndef PYTHON_BINDING_UTILS_ORTHOPOL_HPP_
#define PYTHON_BINDING_UTILS_ORTHOPOL_HPP_

#include <linalg/linalg.hpp>
#include <utils/orthopol.hpp>

#include <pybind11/cast.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "../utils.hpp"

namespace py = pybind11;

template <typename T>
void init_orthopol(py::module &m)
{
    using namespace utils;

    using _orthopol = orthopol<T>;
    using conv = linalg::pybuffer_converter<linalg::blas_backend>;
    // wrapper for the orthopol manager class
    py::class_<_orthopol>(m, "orthopol")
        .def(py::init())
        .def(py::init<size_t, T, T>())
        .def(py::init<size_t, T, T, T>())
        .def(py::init<const _orthopol &>())
        .def("assign", [](_orthopol &self, const _orthopol &o)
             { self = o; })
        .def("__copy__", [](const _orthopol &o)
             { return _orthopol(o); })
        .def(
            "__deepcopy__",
            [](const _orthopol &o, py::dict)
            { return _orthopol(o); },
            py::arg("memo"))
        .def("clear", &_orthopol::clear)
        .def("resize", &_orthopol::resize)
        .def("set_domain", &_orthopol::set_domain)
        .def("scale", &_orthopol::scale)
        .def("shift", &_orthopol::shift)
        .def("set_weight_function_integral",
             &_orthopol::set_weight_function_integral)
        .def("xmin", &_orthopol::xmin)
        .def("xmax", &_orthopol::xmax)
        .def("__call__", &_orthopol::operator())
        .def("monic", &_orthopol::monic)
        .def("set_recurrence_relation",
             [](_orthopol &i, const std::vector<T> &a, const std::vector<T> &b)
             {
                 i.set_recurrence_relation(a, b);
             })
        .def("set_recurrence_relation",
             [](_orthopol &i, const linalg::vector<T> &a,
                const linalg::vector<T> &b)
             { i.set_recurrence_relation(a, b); })
        .def("set_alpha", &_orthopol::set_alpha)
        .def("set_beta", &_orthopol::set_beta)
        .def("compute_nodes_and_weights",
             static_cast<void (_orthopol::*)(size_t, T)>(
                 &_orthopol::compute_nodes_and_weights),
             py::arg(), py::arg("normalisation") = T(1.0))
        .def("compute_nodes_and_weights",
             static_cast<void (_orthopol::*)(T)>(
                 &_orthopol::compute_nodes_and_weights),
             py::arg("normalisation") = T(1.0))
        .def("alpha", &_orthopol::alpha)
        .def("beta", &_orthopol::beta)
        .def("nodes", &_orthopol::nodes)
        .def("weights", &_orthopol::weights)
        .def("Nmax", &_orthopol::Nmax)
        .def("npoints", &_orthopol::npoints)
        .def("size", &_orthopol::size)
        .def("pi0", &_orthopol::pi0)
        .def("set_nodes_and_weights",
             [](_orthopol &i, const std::vector<T> &a, const std::vector<T> &b)
             {
                 i.set_nodes_and_weights(a, b);
             })
        .def("set_nodes_and_weights",
             [](_orthopol &i, const linalg::vector<T> &a,
                const linalg::vector<T> &b)
             { i.set_nodes_and_weights(a, b); });

    // functions for specific realisations of the orthopol type
    m.def("jacobi_polynomial", [](size_t nmax, T alpha, T beta)
          {
    _orthopol ret;
    jacobi_polynomial<T>(ret, nmax, alpha, beta);
    return ret; });
    m.def("gegenbauer_polynomial", [](size_t nmax, T alpha)
          {
    _orthopol ret;
    gegenbauer_polynomial<T>(ret, nmax, alpha);
    return ret; });
    m.def("chebyshev_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    chebyshev_polynomial<T>(ret, nmax);
    return ret; });
    m.def("chebyshev_second_kind_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    chebyshev_second_kind_polynomial<T>(ret, nmax);
    return ret; });
    m.def("chebyshev_third_kind_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    chebyshev_third_kind_polynomial<T>(ret, nmax);
    return ret; });
    m.def("chebyshev_fourth_kind_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    chebyshev_fourth_kind_polynomial<T>(ret, nmax);
    return ret; });
    m.def("legendre_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    legendre_polynomial<T>(ret, nmax);
    return ret; });
    m.def("associated_laguerre_polynomial", [](size_t nmax, T alpha)
          {
    _orthopol ret;
    associated_laguerre_polynomial<T>(ret, nmax, alpha);
    return ret; });
    m.def("laguerre_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    laguerre_polynomial<T>(ret, nmax);
    return ret; });
    m.def("hermite_polynomial", [](size_t nmax)
          {
    _orthopol ret;
    hermite_polynomial<T>(ret, nmax);
    return ret; });
    m.def(
        "nonclassical_polynomial",
        [](const orthopol<T> &orthref, const linalg::vector<T> &modified_moments,
           T scale_factor = 1)
        {
            orthopol<T> orth;
            nonclassical_polynomial<T>(orth, orthref, modified_moments,
                                       scale_factor);
            return orth;
        },
        py::arg(), py::arg(), py::arg("scale_factor") = T(1.0));

    m.def(
        "nonclassical_polynomial",
        [](const orthopol<T> &orthref, const py::buffer &modified_moments,
           T scale_factor = 1)
        {
            orthopol<T> orth;
            linalg::vector<T> mom;
            conv::copy_to_tensor(modified_moments, mom);
            nonclassical_polynomial<T>(orth, orthref, mom, scale_factor);
            return orth;
        },
        py::arg(), py::arg(), py::arg("scale_factor") = T(1.0));
    m.def(
        "nonclassical_polynomial",
        [](const orthopol<T> &orthref, size_t nmax, const std::function<T(T)> &f,
           T rel_tol = 1e-14, T tol = 1e-15, T scale_factor = 1,
           size_t quadrature_order = 100, size_t max_order = 100)
        {
            orthopol<T> orth;
            nonclassical_polynomial<T>(orth, orthref, nmax, f, rel_tol, tol,
                                       scale_factor, quadrature_order, max_order);
            return orth;
        },
        py::arg(), py::arg(), py::arg(), py::arg("rel_tol") = T(1e-12),
        py::arg("tol") = T(1e-14), py::arg("scale_factor") = T(1),
        py::arg("quadrature_order") = 100, py::arg("max_order") = 100);

    m.def(
        "nonclassical_polynomial",
        [](const orthopol<T> &orthref, T xmin, T xmax, size_t nmax,
           const std::function<T(T)> &f, T rel_tol = 1e-14, T tol = 1e-15,
           T scale_factor = 1, size_t quadrature_order = 100,
           size_t max_order = 100)
        {
            orthopol<T> orth;
            nonclassical_polynomial<T>(orth, orthref, xmin, xmax, nmax, f, rel_tol,
                                       tol, scale_factor, quadrature_order,
                                       max_order);
            return orth;
        },
        py::arg(), py::arg(), py::arg(), py::arg(), py::arg(),
        py::arg("rel_tol") = T(1e-12), py::arg("tol") = T(1e-14),
        py::arg("scale_factor") = T(1), py::arg("quadrature_order") = 100,
        py::arg("max_order") = 100);
    m.def("nonclassical_polynomial",
          [](const std::vector<T> &x, const std::vector<T> &w)
          {
              orthopol<T> orth;
              nonclassical_polynomial<T>(orth, x, w);
              return orth;
          });
    m.def("nonclassical_polynomial",
          [](const linalg::vector<T> &x, const linalg::vector<T> &w)
          {
              orthopol<T> orth;
              nonclassical_polynomial<T>(orth, x, w);
              return orth;
          });

    m.def("nonclassical_polynomial",
          [](const py::buffer &_x, const py::buffer &_w)
          {
              orthopol<T> orth;
              linalg::vector<T> x, w;
              conv::copy_to_tensor(_x, x);
              conv::copy_to_tensor(_w, w);
              nonclassical_polynomial<T>(orth, x, w);
              return orth;
          });
}

template <typename T>
void initialise_orthopol(py::module &m);

#endif // PYTHON_BINDING_UTILS_ORTHOPOL_HPP_
