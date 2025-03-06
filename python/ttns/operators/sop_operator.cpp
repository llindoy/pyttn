#include "sop_operator.hpp"
#include "../../pyttn_typedef.hpp"

template <> void initialise_sop_operator<pyttn_real_type, linalg::blas_backend>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_sop_operator<pyttn_real_type, linalg::cuda_backend>(py::module& m);
#endif