#define TTNS_REGISTER_COMPLEX_DOUBLE_OPERATOR

#include <map>
#include <fstream>
#include <sstream>
#include <random>
#include <iostream>
#include <iomanip>
#include <cstdint>
#include <algorithm>
#include <regex>
#include <string>

#include <pybind11/pybind11.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

//#include "utils/orthopol.hpp"
//#include "utils/discretisation.hpp"

#include "linalg/tensor.hpp"
#include "linalg/sparseMatrix.hpp"
#include "linalg/cuda_backend.hpp"

/*
#include "ttns/ttn/ntree.hpp"
#include "ttns/ttn/ttn.hpp"
#include "ttns/ttn/ms_ttn.hpp"
#include "ttns/operators/siteOperators.hpp"
#include "ttns/operators/sop_operator.hpp"
#include "ttns/operators/product_operator.hpp"

#include "ttns/observables/matrix_element.hpp"

#include "ttns/sop/operator_dictionary.hpp"
#include "ttns/sop/sSOP.hpp"
#include "ttns/sop/system_information.hpp"
#include "ttns/sop/SOP.hpp"
#include "ttns/sop/liouville_space.hpp"

#include "ttns/sop/models/models.hpp"

#include "ttns/algorithms/dmrg.hpp"
#include "ttns/algorithms/tdvp.hpp"
*/
//using namespace ttns;
namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<int>);
//PYBIND11_MAKE_OPAQUE(std::vector<size_t>);

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(ttnpp, m)
{
    m.doc()=R"mydelimiter(
      Python wrapping of the TTNS_LIB library for performing calculations on tree tensor network states
      )mydelimiter";
    auto m_linalg = m.def_submodule("linalg",R"mydelimiter(
        Linear algebra submodule for TTNS library."
        )mydelimiter");
    auto m_orthopol = m.def_submodule("utils", R"mydelimiter(
        Orthogonal polynomials submodule for TTNS library.
        )mydelimiter");

    auto m_models = m.def_submodule("models", R"mydelimiter(
        Pre-defined models specifying system and Hamiltonian information.
        )mydelimiter");


    auto m_linalg_gpu = m.def_submodule("linalg_gpu",R"mydelimiter(
        Linear algebra submodule for TTNS library."
        )mydelimiter");
    auto m_orthopol_gpu = m.def_submodule("utils_gpu", R"mydelimiter(
        Orthogonal polynomials submodule for TTNS library.
        )mydelimiter");

    auto m_models_gpu = m.def_submodule("models_gpu", R"mydelimiter(
        Pre-defined models specifying system and Hamiltonian information.
        )mydelimiter");

    //
    //Wrap the required linear algebra types to enable python based instantiation of operators.
    //
    //The CPU implementations
    initialise_tensors<double>(m_linalg);
    initialise_sparse_matrices<double>(m_linalg);

    //the GPU implementations
    initialise_tensors_cuda<double>(m_linalg_gpu);
    initialise_sparse_matrices_cuda<double>(m_linalg);

    //initialise the cuda environment objects
    initialise_cuda_backend(m_linalg_gpu);
}


