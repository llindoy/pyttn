#include "matrix_element.hpp"
#include "../../pyttn_typedef.hpp"

template <> void initialise_matrix_element<pyttn_real_type, linalg::blas_backend>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_matrix_element<pyttn_real_type, linalg::cuda_backend>(py::module& m);
#endif
