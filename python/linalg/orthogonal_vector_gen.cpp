#include "orthogonal_vector_gen.hpp"
#include <linalg/linalg.hpp>

#include "../pyttn_typedef.hpp"


template<> void initialise_orthogonal_vector<pyttn_real_type, linalg::blas_backend>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template<> void initialise_orthogonal_vector<pyttn_real_type, linalg::cuda_backend>(py::module& m);
#endif

