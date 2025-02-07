#include "sop_operator.hpp"

template <> void initialise_sop_operator<linalg::blas_backend>(py::module& m);
