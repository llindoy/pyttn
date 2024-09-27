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

#include "utils/orthopol.hpp"
#include "utils/discretisation.hpp"

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

#include "ttns/sop/models/models.hpp"

#include "ttns/algorithms/dmrg.hpp"
#include "ttns/algorithms/tdvp.hpp"

//using namespace ttns;
namespace py = pybind11;

//PYBIND11_MAKE_OPAQUE(std::vector<int>);
//PYBIND11_MAKE_OPAQUE(std::vector<size_t>);

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MODULE(_pyttn, m)
{
    m.doc()=R"mydelimiter(
      Python wrapping of the TTNS_LIB library for performing calculations on tree tensor network states
      -----------------------------

      .. currentmodule:: _pyttn

      .. autosummary::
          :toctree: _generate
              :maxdepth: 2
              :caption: Contents:

              one_site_dmrg_real
              one_site_dmrg_complex

      )mydelimiter";
    auto m_linalg = m.def_submodule("linalg",R"mydelimiter(
        Linear algebra submodule for TTNS library."
        )mydelimiter");
    auto m_orthopol = m.def_submodule("utils", R"mydelimiter(
        Orthogonal polynomials submodule for TTNS library.
        )mydelimiter");

    //
    //Wrap the required linear algebra types to enable python based instantiation of operators.
    //
    initialise_tensors(m_linalg);
    initialise_sparse_matrices(m_linalg);

    //
    //Wrap the required utils functions
    //
    initialise_orthopol(m_orthopol);
    initialise_discretisation(m_orthopol);
    
    //
    //Wrap the sOP functionality
    //
    initialise_sSOP(m);
    initialise_system_info(m);
    initialise_SOP(m);
    initialise_models(m);
    initialise_operator_dictionary(m);
    
    //
    //Wrap core ttns functionality
    //
    initialise_ntree(m);
    initialise_ttn(m);
    initialise_msttn(m);

    initialise_matrix_element(m);
    //
    //Wrap operator classes
    //
    auto m_ops = m.def_submodule("ops", "Operator submodule for TTNS library.");
    initialise_site_operators(m_ops);
    initialise_product_operator(m);
    initialise_sop_operator(m);


    //
    //Wrap the core algorithms for operating on ttns
    //
    initialise_dmrg(m);
    initialise_tdvp(m);
}


