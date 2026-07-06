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

#include "SOP.hpp"
#include "../../pyttn_typedef.hpp"
#include "common_bindings.hpp"


template <typename T>
void init_SOP(py::module &m, const std::string &label)
{
    using namespace ttns;


    using real_type = typename linalg::get_real_type<T>::type;
    using complex_type = std::complex<real_type>;
    using _SOP = SOP<T>;
    using _msSOP = multiset_SOP<T>;
    {
        // wrapper for the SOP type
        auto cls = py::class_<_SOP>(m, label.c_str());
        cls.def(py::init<size_t>())
            .def(py::init<size_t, const std::string &>())
            .def(py::init<const _SOP &>())
            .def("__iter__", [](_SOP &s){ return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
            .def("clear", &_SOP::clear)
            .def("resize", &_SOP::resize)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def_property("operator_dictionary", &_SOP::operator_dictionary, &_SOP::set_operator_dictionary)
            .def("set_operator_dictionary", &_SOP::set_operator_dictionary)
            .def("get_operator_dictionary", &_SOP::operator_dictionary)
            .def("insert", static_cast<void (_SOP::*)(const T &, const sPOP &)>(&_SOP::insert))
            .def("insert", static_cast<void (_SOP::*)(const sNBO<T> &)>(&_SOP::insert))
            .def("set_is_fermion_mode", &_SOP::set_is_fermionic_mode)
            .def("prune_zeros", &_SOP::prune_zeros, py::arg("tol") = 1e-15)
            .def("jordan_wigner", static_cast<_SOP &(_SOP::*)(const system_modes &, double)>(&_SOP::jordan_wigner), py::arg(), py::arg("tol") = 1e-15)
            .def("expand", &_SOP::expand)
            .def_property("label", static_cast<const std::string &(_SOP::*)() const>(&_SOP::label), [](_SOP &o, const std::string &i){ o.label() = i; })
            .def("__str__", [](const _SOP &o){std::ostringstream oss; oss << o; return oss.str(); })
            .doc() = R"mydelim(
                A class for storing a compact representation of a sum-of-product string operators.  This class requires
                knowledge of the total number of degrees of freedom.

                Construct arguments

                :param A: The Tree Tensor Network Object that will be optimised using the DMRG algorithm
                :type A: ttn_complex
                :param H: The Hamiltonian sop operator object
                :type H: sop_operator_complex
                :param krylov_dim: The krylov subspace dimension used for the eigensolver steps. (Default: 16)
                :type krylov_dim: int, optional
                :param numthreads: The number of openmp threads to be used by the solver. (Default: 1)
                :type numthreads: int, optional

                Callable arguments

                :param A: Tree Tensor Network that the DMRG algorithm will act on
                :type A: ttn_complex
                :param h: The Hamiltonian sop operator object
                :type h: sop_operator_complex
                :param update_env: Whether or not to force an update of all environment tensor at the start of the update scheme.  (Default: False)
                :type update_env: bool, optional
            )mydelim";

        using namespace python_bindings;
        bind_all<bind_imul, _SOP, real_type, T>(cls);
        bind_all<bind_idiv, _SOP, real_type, T>(cls);
        bind_all<bind_iadd, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_all<bind_isub, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_all<bind_add, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_all<bind_radd, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_all<bind_sub, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_all<bind_rsub, _SOP, real_type, T, sOP, sPOP, sNBO<real_type>, sNBO<T>, sSOP<real_type>, sSOP<T>, SOP<real_type>, _SOP>(cls);
        bind_todense<T>(cls);
        bind_dtype<T>(cls);
        bind_utils(cls);
        bind_copyable(cls);
        bind_pickleable(cls);
    }
            
    {
        // wrapper for the msSOP type
        auto cls = py::class_<_msSOP>(m, (std::string("multiset_") + label).c_str());
        cls.def(py::init())
            .def(py::init<size_t, size_t>())
            .def(py::init<size_t, size_t, const std::string &>())
            .def(py::init<const _msSOP &>())
            .def("clear", &_msSOP::clear)
            .def("resize", &_msSOP::resize)
            .def("nmodes", &_msSOP::nmodes)
            .def("nset", &_msSOP::nset)
            .def("nterms", &_msSOP::nterms)

            .def("set", static_cast<void (_msSOP::*)(size_t, size_t, const SOP<T> &)>(&_msSOP::set))
            .def("set_is_fermion_mode", &_msSOP::set_is_fermionic_mode)
            .def("prune_zeros", &_msSOP::prune_zeros, py::arg("tol") = 1e-15)
            .def("jordan_wigner", static_cast<_msSOP &(_msSOP::*)(const system_modes &, double)>(&_msSOP::jordan_wigner), py::arg(), py::arg("tol") = 1e-15)

            .def("__getitem__", [](_msSOP &i, std::pair<size_t, size_t> ind) -> _SOP & { return i(std::get<0>(ind), std::get<1>(ind)); }, py::return_value_policy::reference)
            .def("__setitem__", [](_msSOP &i, std::pair<size_t, size_t> ind, const _SOP &o){ i(std::get<0>(ind), std::get<1>(ind)) = o; })
            .def_property("label", static_cast<const std::string &(_msSOP::*)() const>(&_msSOP::label), [](_msSOP &o, const std::string &i){ o.label() = i; })
            .def("__str__", [](const _msSOP &o){std::ostringstream oss; oss << o; return oss.str(); });
            
        using namespace python_bindings;
        bind_dtype<T>(cls);
        bind_utils(cls);
        bind_copyable(cls);
        bind_pickleable(cls);
    }

    // SOP<T>& operator()(size_t i, size_t j)
    // const SOP<T>& operator()(size_t i, size_t j) const
}


template <typename T>
void init_prodOP(py::module &m, const std::string &label)
{
    using namespace ttns;

    using prod_type = prodOP;
    using elem_type = typename prod_type::elem_type;

    py::class_<prod_type>(m, label.c_str())
        .def("__len__", &prod_type::size)
        .def("size", &prod_type::size)
        .def("nmodes", &prod_type::nmodes)

        .def(
            "__iter__",
            [](const prod_type &p)
            {
                return py::make_iterator(p.begin(), p.end());
            },
            py::keep_alive<0, 1>())

        .def("__getitem__", [](const prod_type &p, size_t i)
             {
                 if (i >= p.size())
                     throw py::index_error("prodOP index out of range");
                 const auto &e = p[i];
                 return py::make_tuple(
                     std::get<0>(e),  // operator index
                     std::get<1>(e),  // mode
                     std::get<2>(e)   // fermionic flag
                 );
             })

        .def(
            "as_sPOP",
            [](const prod_type &p,
               const std::vector<std::vector<std::string>> &opdict)
            {
                return p.as_prod_op(opdict);
            },
            py::arg("operator_dictionary"))

        .def("__repr__", [](const prod_type &p)
             { return std::string(p); })


        .def("contains_jw", &prod_type::contains_jordan_wigner_string)
        .def("prepend_jw", &prod_type::prepend_jordan_wigner_string);
};

void initialise_SOP(py::module &m)
{
    using real_type = pyttn_real_type;
    using complex_type = std::complex<real_type>;
#ifdef BUILD_REAL_TTN
    init_SOP<real_type>(m, "SOP_real");
    init_prodOP<real_type>(m, "prodOP_real");

#endif
    init_SOP<complex_type>(m, "SOP_complex");
    init_prodOP<complex_type>(m, "prodOP_complex");

}
