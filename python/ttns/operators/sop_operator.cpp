#include "sop_operator.hpp"

void initialise_sop_operator(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_sop_operator<real_type>(m, "real");
    init_sop_operator<complex_type>(m, "complex");
}
