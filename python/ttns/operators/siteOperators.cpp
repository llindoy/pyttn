#include "siteOperators.hpp"

template <> void initialise_site_operators<linalg::blas_backend>(py::module& m);
