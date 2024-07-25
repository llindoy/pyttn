#include "ttn.hpp"

void initialise_ttn(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_ttn<real_type>(m, "real");
    init_ttn<complex_type>(m, "complex");
}
