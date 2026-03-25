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

#ifndef PYTHON_BINDING_TDVP_HPP
#define PYTHON_BINDING_TDVP_HPP

#include "../../utils.hpp"

#include <ttns_lib/ttn/ttn.hpp>
#include <ttns_lib/ttn/ms_ttn.hpp>
#include <ttns_lib/sweeping_algorithm/tdvp.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

void initialise_tdvp_onesite_ttn(py::module &m);
void initialise_tdvp_adaptive_ttn(py::module &m);

void initialise_tdvp_onesite_msttn(py::module &m);

#ifdef PYTTN_BUILD_CUDA
void initialise_tdvp_onesite_ttn_cuda(py::module &m);
void initialise_tdvp_adaptive_ttn_cuda(py::module &m);

void initialise_tdvp_onesite_msttn_cuda(py::module &m);
#endif

#endif
