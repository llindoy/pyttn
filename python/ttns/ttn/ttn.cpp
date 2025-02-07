#include "ttn.hpp"
#include "../../pyttn_typedef.hpp"

template <> void initialise_ttn<pyttn_real_type, linalg::blas_backend>(py::module& m);
