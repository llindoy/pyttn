#include "SOP.hpp"

void initialise_SOP(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
#ifdef BUILD_REAL_TTN
    init_SOP<real_type>(m, "SOP_real");
#endif
    init_SOP<complex_type>(m, "SOP_complex");
}
