#include "sparseMatrix.hpp"

template <> void initialise_sparse_matrices<double>(py::module& m);
template <> void initialise_sparse_matrices_cuda<double>(py::module& m);
