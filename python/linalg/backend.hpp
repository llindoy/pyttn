#ifndef PYTHON_BINDING_LINALG_CUDA_BACKEND_HPP
#define PYTHON_BINDING_LINALG_CUDA_BACKEND_HPP

#include <linalg/linalg.hpp>

#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

void initialise_blas_backend(py::module& m);

#ifdef PYTTN_BUILD_CUDA
void initialise_cuda_backend(py::module& m);
#endif

#endif //PYTHON_BINDING_LINALG_CUDA_BACKEND_HPP