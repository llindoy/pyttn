#include "tensor.hpp"
#include <linalg/linalg.hpp>

template <> void initialise_tensors<linalg::blas_backend>(py::module& m);
