#include "tdvp.hpp"

void initialise_tdvp(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
  
    //init_tdvp<real_type>(m, "real");
    init_tdvp<complex_type>(m, "complex");
}
