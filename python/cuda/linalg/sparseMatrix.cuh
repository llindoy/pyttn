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

#ifndef PYTHON_BINDING_LINALG_SPARSE_MATRIX_CUH
#define PYTHON_BINDING_LINALG_SPARSE_MATRIX_CUH


#include <linalg/linalg.cuh>
#include "../../linalg/sparseMatrix.hpp"
#include "../../linalg/sparseMatrix.tpp"

namespace py = pybind11;

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
        //.def(py::init<const tensor<T, 1, cuda_backend> &>())
        //.def(py::init<const tensor<T, 1, cuda_backend> &, size_t>())
        //.def(py::init<const tensor<T, 1, cuda_backend> &, size_t, size_t>())

        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("backend", [](const ttype &)
             { return linalg::traits<backend>::label(); });

    // expose the ttn node class.  This is our core tensor network object.
}


#endif // PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP
