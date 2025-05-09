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

#ifndef PYTHON_BINDING_LINALG_ORTHOGONAL_VECTOR_GEN_HPP
#define PYTHON_BINDING_LINALG_ORTHOGONAL_VECTOR_GEN_HPP

#include <linalg/linalg.hpp>

#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

template <typename T, typename backend>
void init_orthogonal_vector(py::module &m, const std::string &label)
{
    using namespace linalg;
    using size_type = typename backend::size_type;
    using ttype = orthogonal_vector<T, backend>;
    using vectype = linalg::vector<T, backend>;
    using mattype = linalg::matrix<T, backend>;

    py::class_<ttype>(m, label.c_str())
        .def_static("pad_random", [](mattype &a, size_type i, random_engine<backend> &rng)
                    { ttype::pad_random(a, i, rng); }, R"mydelim(
                    Pad the columns of the matrix starting at row index i with random vectors that are set to be orthogonal to all
                    current vectors using Gram schmidt

                    :param a: Matrix to be padded
                    :type a: Matrix
                    :param i: Starting row to be filled with random vectors
                    :type i: int
                    :param rng: The random number generate used to fill the vector
                    :type rng: random_engine
                    )mydelim")
        .def_static("generate", [](const mattype &a, random_engine<backend> &rng)
                    {
                        vectype b(a.shape(1));
                        ttype::generate(a, b, rng);
                        return b; }, R"mydelim(
                    Generate a vector that is orthogonal to the rows of the matrix a

                    :param a: Matrix to be padded
                    :type a: Matrix
                    :param rng: The random number generate used to fill the vector
                    :type rng: random_engine

                    :returns: The newly generated matrix with random orthogonal rows that are also orthogonal to all rows of matrix a
                    :rtype: Vector
                    )mydelim")
        .def("backend", [](const ttype &)
             { return backend::label(); });
}

template <typename backend>
void init_random_engine(py::module &m, const std::string &label)
{
    using namespace linalg;
    using ttype = random_engine<backend>;

    py::class_<ttype>(m, label.c_str())
        .def(py::init(), R"mydelim(
                    Python wrapper of the C++ random number generator object
                    )mydelim")
        .def("backend", [](const ttype &)
             { return backend::label(); }, R"mydelim(
                    Returns the backend type of the C++ random number generator.
                    )mydelim");
}

template <typename real_type, typename backend>
void initialise_orthogonal_vector(py::module &m)
{
    using complex_type = linalg::complex<real_type>;

    init_random_engine<backend>(m, "random_engine");
    init_orthogonal_vector<real_type, backend>(m, "orthogonal_vector_real");
    init_orthogonal_vector<complex_type, backend>(m, "orthogonal_vector_complex");
}
#endif // PYTHON_BINDING_LINALG_CUDA_BACKEND_HPP