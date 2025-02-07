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

#include "pyttn_typedef.hpp"

#include "utils/orthopol.hpp"
#include "utils/discretisation.hpp"

#include "linalg/backend.hpp"
#include "linalg/tensor.hpp"
#include "linalg/sparseMatrix.hpp"

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

    //
    //Wrap the required linear algebra types to enable python based instantiation of operators.
    //
    initialise_tensors<pyttn_real_type>(m_linalg);
    initialise_sparse_matrices<pyttn_real_type>(m_linalg);
    initialise_blas_backend(m_linalg);

    //
    //Wrap the required utils functions
    //
    initialise_orthopol<pyttn_real_type>(m_orthopol);
    initialise_discretisation<pyttn_real_type>(m_orthopol);
    
    //
    //Wrap the sOP functionality
    //
    initialise_sSOP<pyttn_real_type>(m);
    initialise_system_info(m);
    initialise_SOP<pyttn_real_type>(m);
    initialise_operator_dictionary<pyttn_real_type, linalg::blas_backend>(m);
    initialise_liouville_space<pyttn_real_type, linalg::blas_backend>(m);

    //
    //Wrap the models functionality included in SOP
    //
    initialise_models<pyttn_real_type>(m_models);
    
    //
    //Wrap core ttns functionality
    //
    initialise_ntree(m);
    initialise_ttn<pyttn_real_type,linalg::blas_backend>(m);
    initialise_msttn<pyttn_real_type, linalg::blas_backend>(m);

    initialise_matrix_element<linalg::blas_backend>(m);
    //
    //Wrap operator classes
    //
    auto m_ops = m.def_submodule("ops", "Operator submodule for TTNS library.");
    initialise_site_operators<linalg::blas_backend>(m_ops);
    initialise_product_operator<linalg::blas_backend>(m);
    initialise_sop_operator<linalg::blas_backend>(m);

    //
    //Wrap the core algorithms for operating on ttns
    //
    initialise_dmrg<linalg::blas_backend>(m);
    initialise_tdvp<linalg::blas_backend>(m);
}


