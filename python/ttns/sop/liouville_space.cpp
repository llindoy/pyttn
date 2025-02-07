#include "liouville_space.hpp"
#include "../../pyttn_typedef.hpp"

namespace py=pybind11;

template <> void initialise_liouville_space<pyttn_real_type, linalg::blas_backend>(py::module &m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_liouville_space<pyttn_real_type, linalg::cuda_backend>(py::module &m);
#endif
