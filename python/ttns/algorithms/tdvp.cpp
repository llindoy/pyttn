#include "tdvp.hpp"

void initialise_tdvp(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
  
#ifdef BUILD_REAL_TTN
    //init_tdvp<real_type>(m, "real");
#endif
    init_tdvp<complex_type>(m, "complex");
}
