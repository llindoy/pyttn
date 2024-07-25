#include "models.hpp"

void initialise_models(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_models<real_type>(m, "real");
    init_models<complex_type>(m, "complex");
}
