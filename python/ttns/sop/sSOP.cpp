#include "sSOP.hpp"

void initialise_sSOP(py::module& m)
{
    using real_type = double;
    init_sSOP<real_type>(m);
}
