#include "operator_dictionary.hpp"
#include "../../pyttn_typedef.hpp"

template <> void initialise_operator_dictionary<pyttn_real_type, linalg::blas_backend>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_operator_dictionary<pyttn_real_type, linalg::cuda_backend>(py::module& m);
#endif

