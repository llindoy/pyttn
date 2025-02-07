#include "models.hpp"
#include "../../../pyttn_typedef.hpp"

template <>
void initialise_models<pyttn_real_type>(py::module& m)
{
    using real_type = pyttn_real_type;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_models<real_type>(m, "real");
#endif
    init_models<complex_type>(m, "complex");
}
