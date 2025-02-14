#include "tdvp.hpp"
#include "../../pyttn_typedef.hpp"

template <> void initialise_tdvp<pyttn_real_type, linalg::blas_backend>(py::module& m);
template <> void initialise_tdvp_adaptive<pyttn_real_type, linalg::blas_backend>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_tdvp<pyttn_real_type, linalg::cuda_backend>(py::module& m);
//template <> void initialise_tdvp_adaptive<pyttn_real_type, linalg::cuda_backend>(py::module& m);

#endif