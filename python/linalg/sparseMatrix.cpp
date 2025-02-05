#include "sparseMatrix.hpp"
#include "../pyttn_typedef.hpp"

template <> void initialise_sparse_matrices<pyttn_real_type>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template <> void initialise_sparse_matrices_cuda<pyttn_real_type>(py::module& m);
#endif
