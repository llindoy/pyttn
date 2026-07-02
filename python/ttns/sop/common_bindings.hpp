/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2025 NPL Management Limited
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *     http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License
 */

#ifndef PYTHON_BINDING_SOP_COMMON_BINDINGS_HPP
#define PYTHON_BINDING_SOP_COMMON_BINDINGS_HPP

#include "../../common_bindings.hpp"

namespace python_bindings
{
    template <typename real_type, typename CLS>
    void bind_add_sub_sop(py::class_<CLS> &c)
    {
        using complex_type = std::complex<real_type>;
        using namespace ttns;
        using namespace literal;

        bind_all<bind_add, CLS, sOP, sPOP, sNBO<real_type>, sNBO<complex_type>, sSOP<real_type>, sSOP<complex_type>>(c, R"mydelim(
              Functions for adding two OPBase objects
              depending on the dtype used.  

              :Parameters:  - **a** (:class:`OPBase`) - The left term in the expression
                            - **b** (class:`OPBase`) - The right term in the expression

              :returns: The result of the sum
              :rtype: :class:`sSOP` 
          )mydelim");

        bind_all<bind_sub, CLS, sOP, sPOP, sNBO<real_type>, sNBO<complex_type>, sSOP<real_type>, sSOP<complex_type>>(c, R"mydelim(
              Functions for subtracting an OPBase object from another OPBase object 
              depending on the dtype used.  

              :Parameters:  - **a** (:class:`OPBase`) - The left term in the expression
                            - **b** (class:`OPBase`) - The right term in the expression

              :returns: The result of the sum
              :rtype: :class:`sSOP` 
          )mydelim");
    }

    template <typename real_type, typename CLS>
    void bind_mul_div_sop(py::class_<CLS> &c)
    {
        using complex_type = std::complex<real_type>;

        using namespace ttns;
        using namespace literal;

        bind_all<bind_div, CLS, real_type, complex_type>(c);
        c.def("__truediv__", [](const CLS &a, py::object b)
              { return a * (1.0 / linalg::extract_scalar<real_type, complex_type>(b)); }, R"mydelim(
                         Functions for dividing a OBPase by a scalar.
                         :Parameters:  - **a** (:class:`OBPase`) - The left term in the expression
                                        - **b** (float or complex ) - The scalar to divide the OPBase by
                         :returns: The result of a/b
                         :rtype: :class:`OPBase`
                         )mydelim");

        bind_all<bind_mul, CLS, real_type, complex_type, coeff<real_type>, coeff<complex_type>, sOP, sPOP, sNBO<real_type>, sNBO<complex_type>, sSOP<real_type>, sSOP<complex_type>>(c, R"mydelim(
                    Functions for multiplying an OPBase by an object.

                    :Parameters:  - **a** (:class:`OPBase`) - The left term in the expression
                                   - **b** (:class:`OPBase` or :class:`coeff` of float or complex) - The right term in the expression

                    :Returns: The result of a*b
                    :Return Type: :class:`OPBase` 

                    )mydelim");
        c.def("__mul__", [](const CLS &a, py::object b)
              { return a * linalg::extract_scalar<real_type, complex_type>(b); });

        bind_all<bind_rmul, CLS, real_type, complex_type, coeff<real_type>, coeff<complex_type>>(c, R"mydelim(
                    Functions for multiplying an OPBase by an OPBase.

                    :Parameters:  - **a** (:class:`OPBase`) - The left term in the expression
                                  - **b** (:class:`coeff` of float or complex) - The right term in the expression

                    :Returns: The result of a*b
                    :Return Type: :class:`OPBase` 

                    )mydelim");

        c.def("__rmul__", [](const CLS &a, py::object b)
              { return a * linalg::extract_scalar<real_type, complex_type>(b); });
    }
}
#endif //PYTHON_BINDING_SOP_COMMON_BINDINGS_HPP