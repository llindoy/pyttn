#include "discretisation.hpp"

void initialise_discretisation(py::module& m)
{
    using real_type = double;

    init_discretisation<real_type>(m);
}
