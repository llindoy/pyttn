#include "matrix_element.hpp"

void initialise_matrix_element(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_matrix_element<real_type>(m, "real");
    init_matrix_element<complex_type>(m, "complex");
}
