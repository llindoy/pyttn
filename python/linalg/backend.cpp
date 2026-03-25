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

#include "backend.hpp"
#include <linalg/linalg.hpp>
#include <sstream>

void initialise_blas_backend(py::module &m)
{
    using namespace linalg;

    using size_type = typename linalg::traits<blas_backend>::size_type;

    // expose the ttn node class.  This is our core tensor network object.
    py::class_<blas_backend>(m, "backend")
        .def_static("initialise", [](size_type nthreads, bool batch_par)
                    { blas_backend::initialise(nthreads, batch_par); }, py::arg("nthreads") = 1, py::arg("batch_par") = false, R"mydelim(
            Initialise blas backend passing user defined arguments

            :param nthreads: The number of threads to use for linear algebra operations (Default: 1)
            :type nthreads: int, optional
            :param batch_par: Whether or not to parallelise batched gemm operationrs (Default: false)
            :type bath_par: bool, optional
            )mydelim")
        .def_static("destroy", &blas_backend::destroy, R"mydelim(
            Clear the blas_backend object.   Free any resources allocated.
            )mydelim")
        .def_static("label", &blas_backend::label, R"mydelim(
            :returns: A string representing a blas backend
            :rtype: str
            )mydelim");
}

