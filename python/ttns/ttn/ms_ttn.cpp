#include "ms_ttn.hpp"

void initialise_msttn(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_msttn<real_type>(m, "real");
#endif
    init_msttn<complex_type>(m, "complex");
}
