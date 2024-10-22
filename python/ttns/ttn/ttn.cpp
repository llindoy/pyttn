#include "ttn.hpp"

void initialise_ttn(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_ttn<real_type>(m, "real");
#endif
    init_ttn<complex_type>(m, "complex");
}
