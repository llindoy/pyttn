/**
 * This files is part of the pyTTN package.
 * (C) Copyright 2026 NPL Management Limited
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

#include "symbolic_transpose.hpp"
#include "../../pyttn_typedef.hpp"
#include "../../common_bindings.hpp"

namespace py = pybind11;


template <typename In> struct symbolic_transpose_output{using type = In;};
template <> struct symbolic_transpose_output<ttns::sOP>{using type = ttns::sNBO<std::complex<pyttn_real_type>>;};
template <> struct symbolic_transpose_output<ttns::sPOP>{using type = ttns::sNBO<std::complex<pyttn_real_type>>;};
template <typename In, typename value_type> struct symbolic_transpose_dict_output;
template <template <typename > class Op, typename v1, typename v2> struct symbolic_transpose_dict_output<Op<v1>, v2>{using type = Op<decltype(v1()*v2())>;};
template <typename value_type> struct symbolic_transpose_dict_output<ttns::sOP, value_type>{using type = ttns::sNBO<value_type>;};
template <typename value_type> struct symbolic_transpose_dict_output<ttns::sPOP, value_type>{using type = ttns::sNBO<value_type>;};

template <typename In>
struct bind_symbolic_transpose
{
    static inline void apply (py::class_<ttns::symbolic_transpose>& cls, const char * = nullptr)
    {
        using Out = typename symbolic_transpose_output<In>::type;
        cls.def_static("apply", [](const In& in, const ttns::system_modes& sys){Out out; ttns::symbolic_transpose::apply(in, sys, out); return out;} );
    }
};
template <typename T, typename In>
struct bind_symbolic_transpose_dict
{
    static inline void apply(py::class_<ttns::symbolic_transpose>& cls, const char * = nullptr)
    {
        using namespace ttns;
        using opdict = operator_dictionary<T, linalg::blas_backend>;
        using Out = typename symbolic_transpose_dict_output<In, T>::type;

        cls.def_static("apply", [](const In& in, const opdict& dictin, const ttns::system_modes& sys, opdict& dictout, const std::string& suffix){Out out; ttns::symbolic_transpose::apply(in, dictin, sys, out, dictout, suffix); return out;}, 
                                py::arg(), py::arg(), py::arg(), py::arg(), py::arg("suffix") = std::string("~"));
    }
};


void initialise_symbolic_transpose(py::module &m)
{
    using namespace ttns;
    using real_type = pyttn_real_type;
    using complex_type = std::complex<real_type>;

    using opdictr = operator_dictionary<real_type, linalg::blas_backend>;
    using opdictc = operator_dictionary<complex_type, linalg::blas_backend>;
    auto cls = py::class_<symbolic_transpose>(m, "symbolic_transpose");
    python_bindings::bind_all<bind_symbolic_transpose, symbolic_transpose, sOP, sPOP, sNBO<complex_type>, sSOP<complex_type>, SOP<complex_type>, sNBO<real_type>, sSOP<real_type>, SOP<real_type>>(cls);
    python_bindings::bind_all<bind_symbolic_transpose_dict, symbolic_transpose, real_type, sOP, sPOP, sNBO<complex_type>, sSOP<complex_type>, SOP<complex_type>, sNBO<real_type>, sSOP<real_type>, SOP<real_type>>(cls);
    python_bindings::bind_all<bind_symbolic_transpose_dict, symbolic_transpose, complex_type, sOP, sPOP, sNBO<complex_type>, sSOP<complex_type>, SOP<complex_type>, sNBO<real_type>, sSOP<real_type>, SOP<real_type>>(cls);
}
