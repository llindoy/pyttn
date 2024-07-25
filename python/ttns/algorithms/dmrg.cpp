#include "dmrg.hpp"

void initialise_dmrg(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
  
    //init_dmrg<real_type>(m, "real");
    init_dmrg<complex_type>(m, "complex");
}
