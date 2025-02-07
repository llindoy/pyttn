#include "tensor.hpp"
#include <linalg/linalg.hpp>
#include "../pyttn_typedef.hpp"


template<> void initialise_tensors<pyttn_real_type>(py::module& m);

#ifdef PYTTN_BUILD_CUDA
template<> void initialise_tensors_cuda<pyttn_real_type>(py::module& m);
#endif

