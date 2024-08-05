#include "ms_ttn.hpp"

void initialise_msttn(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_msttn<real_type>(m, "real");
    init_msttn<complex_type>(m, "complex");
}
