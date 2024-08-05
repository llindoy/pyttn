#include "orthopol.hpp"

void initialise_orthopol(py::module& m)
{
    using real_type = double;

    init_orthopol<real_type>(m);
}
