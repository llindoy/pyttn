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

#ifndef PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP
#define PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP

#include <linalg/linalg.hpp>
#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, typename backend>
void init_csr_matrix(py::module &m, const std::string &label)
{
    using namespace linalg;
    using ttype = csr_matrix<T, backend>;
    using index_type = typename ttype::index_type;
    using coo_type = std::vector<std::tuple<index_type, index_type, T>>;
    using real_type = typename get_real_type<T>::type;

#ifdef PYTTN_BUILD_CUDA
    using obackend = typename other_backend<backend>::type;
#endif
    // to do figure out a way of exposing the c++ buffers to python
    py::class_<ttype>(m, (label).c_str())
        .def(py::init<ttype>())
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<csr_matrix<T, obackend>>())
#endif
        .def(py::init<const std::vector<T> &, const std::vector<index_type> &, const std::vector<index_type> &, size_t>(), py::arg(), py::arg(), py::arg(), py::arg("ncols") = 0)
        .def(py::init([](py::buffer &b, const py::buffer &_indices, const py::buffer &_rowptr, size_t ncols)
                      {
                std::vector<T> tens;
                std::vector<index_type> indices, rowptr;
                pybuffer_to_vector(b, tens);
                pybuffer_to_vector(_indices, indices);
                pybuffer_to_vector(_rowptr, rowptr);

                return ttype(tens, indices, rowptr, ncols); }),
             py::arg(), py::arg(), py::arg(), py::arg("ncols") = 0)
        .def(py::init<const coo_type &, size_t, size_t>(), py::arg(), py::arg("nrows") = 0, py::arg("ncols") = 0)
        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__matmul__",
             [](const ttype &a, linalg::matrix<T, backend> &b)
             {
                 linalg::matrix<T, backend> ret;
                 ret = a * b;
                 return ret;
             })
        .def("__matmul__",
             [](const ttype &a, linalg::vector<T, backend> &b)
             {
                 linalg::vector<T, backend> ret;
                 ret = a * b;
                 return ret;
             })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;  oss << o; return oss.str(); })
        .def("backend", [](const ttype &)
             { return backend::label(); });
}

template <typename T>
void init_diagonal_matrix(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = blas_backend;
    using ttype = diagonal_matrix<T, backend>;
    using real_type = typename get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    py::class_<ttype>(m, (label).c_str(), py::buffer_protocol())
        .def(py::init<ttype>())
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<diagonal_matrix<T, cuda_backend>>())
#endif
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                conv::copy_to_diagonal_matrix(b, tens);
                return tens; }))
        .def(py::init<const std::vector<T> &>())
        .def(py::init<const std::vector<T> &, size_t>())
        .def(py::init<const std::vector<T> &, size_t, size_t>())
        .def(py::init<const tensor<T, 1> &>())
        .def(py::init<const tensor<T, 1> &, size_t>())
        .def(py::init<const tensor<T, 1> &, size_t, size_t>())

#ifdef PYTTN_BUILD_CUDA
        .def(py::init<const tensor<T, 1, cuda_backend> &>())
        .def(py::init<const tensor<T, 1, cuda_backend> &, size_t>())
        .def(py::init<const tensor<T, 1, cuda_backend> &, size_t, size_t>())
#endif

        .def_buffer([](ttype &mi) -> py::buffer_info
                    { return py::buffer_info(
                          mi.buffer(),                        // pointer to buffer
                          sizeof(T),                          // size of one scalar
                          py::format_descriptor<T>::format(), // Python struct-style format descriptor
                          1,                                  // Number of dimensions D
                          std::vector<size_t>{mi.nrows()},    // shape of the array
                          std::vector<size_t>{sizeof(T)}      // strides of the array
                      ); })
        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("backend", [](const ttype &)
             { return backend::label(); });

    // expose the ttn node class.  This is our core tensor network object.
}

template <typename real_type>
void initialise_sparse_matrices(py::module &m)
{
    using complex_type = linalg::complex<real_type>;
    init_csr_matrix<real_type, linalg::blas_backend>(m, "csr_matrix_real");
    init_csr_matrix<complex_type, linalg::blas_backend>(m, "csr_matrix_complex");
    init_diagonal_matrix<real_type>(m, "diagonal_matrix_real");
    init_diagonal_matrix<complex_type>(m, "diagonal_matrix_complex");
}

#ifdef PYTTN_BUILD_CUDA
template <typename T>
void init_diagonal_matrix_cuda(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = cuda_backend;
    using ttype = diagonal_matrix<T, backend>;
    using real_type = typename get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    py::class_<ttype>(m, (label).c_str())
        .def(py::init<ttype>())
        .def(py::init<diagonal_matrix<T, blas_backend>>())
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                conv::copy_to_diagonal_matrix(b, tens);
                return tens; }))
        .def(py::init<const std::vector<T> &>())
        .def(py::init<const std::vector<T> &, size_t>())
        .def(py::init<const std::vector<T> &, size_t, size_t>())
        .def(py::init<const tensor<T, 1> &>())
        .def(py::init<const tensor<T, 1> &, size_t>())
        .def(py::init<const tensor<T, 1> &, size_t, size_t>())
        .def(py::init<const tensor<T, 1, cuda_backend> &>())
        .def(py::init<const tensor<T, 1, cuda_backend> &, size_t>())
        .def(py::init<const tensor<T, 1, cuda_backend> &, size_t, size_t>())

        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("backend", [](const ttype &)
             { return backend::label(); });

    // expose the ttn node class.  This is our core tensor network object.
}
template <typename real_type>
void initialise_sparse_matrices_cuda(py::module &m)
{
    using complex_type = linalg::complex<real_type>;
    init_csr_matrix<real_type, linalg::cuda_backend>(m, "csr_matrix_real");
    init_csr_matrix<complex_type, linalg::cuda_backend>(m, "csr_matrix_complex");
    init_diagonal_matrix_cuda<real_type>(m, "diagonal_matrix_real");
    init_diagonal_matrix_cuda<complex_type>(m, "diagonal_matrix_complex");
}
#endif

#endif // PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP
