#include "sSOP.hpp"
#include "../../pyttn_typedef.hpp"


template <> void initialise_sSOP<pyttn_real_type>(py::module& m)
{
    init_sSOP<pyttn_real_type>(m);
}
