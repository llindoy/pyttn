#include "liouville_space.hpp"
#include "../../pyttn_typedef.hpp"

namespace py=pybind11;

template <> void initialise_liouville_space<pyttn_real_type>(py::module &m);

