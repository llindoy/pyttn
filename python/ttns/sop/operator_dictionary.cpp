#include "operator_dictionary.hpp"

void initialise_operator_dictionary(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
  
    init_operator_dictionary<real_type>(m, "real");
    init_operator_dictionary<complex_type>(m, "complex");
}
