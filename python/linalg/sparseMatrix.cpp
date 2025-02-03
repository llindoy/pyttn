#include "sparseMatrix.hpp"

template <> void initialise_sparse_matrices<linalg::blas_backend>(py::module& m);
