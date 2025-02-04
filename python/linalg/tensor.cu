#include "tensor.hpp"
#include <linalg/linalg.hpp>

template<> void initialise_tensors<double>(py::module& m);
template<> void initialise_tensors_cuda<double>(py::module& m);

