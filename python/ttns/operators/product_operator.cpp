#include "product_operator.hpp"

template <> void initialise_product_operator<linalg::blas_backend>(py::module& m);
