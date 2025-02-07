#include "orthopol.hpp"
#include "../pyttn_typedef.hpp"

template <> void initialise_orthopol<pyttn_real_type>(py::module& m)
{
    init_orthopol<pyttn_real_type>(m);
}
