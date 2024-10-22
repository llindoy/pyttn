#include "siteOperators.hpp"

void initialise_site_operators(py::module& m)
{

    using real_type = double;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_site_operators<real_type>(m, "real");
#endif
    init_site_operators<complex_type>(m, "complex");
}
