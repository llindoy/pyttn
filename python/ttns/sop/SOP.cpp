#include "SOP.hpp"
#include "../../pyttn_typedef.hpp"

template <>
void initialise_SOP<pyttn_real_type>(py::module& m)
{
    using real_type = pyttn_real_type;
    using complex_type = linalg::complex<real_type>;
#ifdef BUILD_REAL_TTN
    init_SOP<real_type>(m, "SOP_real");
#endif
    init_SOP<complex_type>(m, "SOP_complex");
}
