#include "product_operator.hpp"

void initialise_product_operator(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;

    init_product_operator<real_type>(m, "real");
    init_product_operator<complex_type>(m, "complex");
}
