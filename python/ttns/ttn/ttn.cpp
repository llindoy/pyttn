#include "ttn.hpp"

template <> void initialise_ttn<linalg::blas_backend>(py::module& m);
