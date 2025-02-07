#include "discretisation.hpp"
#include "../pyttn_typedef.hpp"

template <> void initialise_discretisation<pyttn_real_type>(py::module& m)
{
    using real_type = double;

    init_discretisation<real_type>(m);
}
