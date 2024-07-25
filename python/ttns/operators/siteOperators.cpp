#include "siteOperators.hpp"

void initialise_site_operators(py::module& m)
{

    using real_type = double;
    using complex_type = linalg::complex<real_type>;
    init_site_operators<real_type>(m, "real");
    init_site_operators<complex_type>(m, "complex");
}
