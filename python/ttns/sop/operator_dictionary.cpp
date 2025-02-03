#include "operator_dictionary.hpp"

template <> void initialise_operator_dictionary<linalg::blas_backend>(py::module& m);
