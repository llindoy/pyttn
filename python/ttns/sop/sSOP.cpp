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

#include "sSOP.hpp"
#include "../../pyttn_typedef.hpp"
#include "../../common_bindings.hpp"
#include "common_bindings.hpp"
namespace python_bindings
{
    template <typename T>
    void bind_nbo_common(py::class_<ttns::sNBO<T>> &cls)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;
        using namespace ttns;
        using namespace literal;
        using NBO = sNBO<T>;
        cls.def(py::init())
            .def(py::init<const sOP &>())
            .def(py::init<const sPOP &>())
            .def(py::init<const coeff<T> &, const sPOP &>())
            .def(py::init<const coeff<T> &, const sOP &>())
            .def(py::init<const NBO &>())
            .def("clear", &NBO::clear)
            .def("insert_front", &NBO::prepend)
            .def("insert_back", &NBO::append)
            .def("nmodes", &NBO::nmodes)
            .def("__iter__", [](NBO &s){ return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
            .def("__str__", [](const NBO &o){ return static_cast<std::string>(o); });

        BIND_RW_PROPERTY(cls, NBO, coeff<T>, coeff, nullptr);
        BIND_RW_PROPERTY(cls, NBO, std::list<sOP>, ops, nullptr);
        BIND_RW_PROPERTY(cls, NBO, sPOP, pop, nullptr);

        bind_all<bind_imul, NBO, sOP, sPOP, NBO, real_type>(cls);
        bind_add_sub_sop<real_type>(cls);
        bind_mul_div_sop<real_type>(cls);
        bind_dtype<T>(cls);
        bind_utils(cls);
        bind_copyable(cls);
        bind_pickleable(cls);
    }

    template <typename T>
    void bind_coeff_common(py::class_<ttns::literal::coeff<T>> &cls)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;
        using other_type = typename std::conditional_t<std::is_same_v<T, real_type>, complex_type, real_type>;
        using func_type = std::function<T(real_type)>;

        using namespace ttns;
        using namespace literal;
        using coef = coeff<T>;
        cls.def(py::init())
            .def(py::init<const real_type &>())
            .def(py::init<const func_type &>())
            .def(py::init<const coef &>())
            .def("assign", [](coef &self, const real_type &o){ self = o; })
            .def("assign", [](coef &self, const func_type &o){ self = o; })
            .def("clear", &coef::clear)
            .def("is_zero", &coef::is_zero, py::arg("tol") = real_type(1e-14))
            .def("is_positive", &coef::is_positive)
            .def("is_time_dependent", &coef::is_time_dependent)
            .def("__call__", &coef::operator())
            .def("__str__", [](const coef &o){std::ostringstream oss; oss << o; return oss.str(); });

        bind_mul_div_sop<real_type>(cls);
        bind_all<bind_add, coef, coef, coeff<other_type>, real_type, complex_type>(cls);
        bind_all<bind_sub, coef, coef, coeff<other_type>, real_type, complex_type>(cls);
        bind_all<bind_mul, coef, coef, coeff<other_type>, real_type, complex_type>(cls);
        bind_all<bind_div, coef, real_type, complex_type>(cls);
        bind_all<bind_radd, coef, real_type, complex_type>(cls);
        bind_all<bind_rsub, coef, real_type, complex_type>(cls);
        bind_all<bind_iadd, coef, real_type, coef>(cls);
        bind_all<bind_isub, coef, real_type, coef>(cls);
        bind_all<bind_imul, coef, real_type, coef>(cls);
        bind_idiv<real_type>::apply(cls);
        bind_dtype<T>(cls);
        bind_copyable(cls);
        bind_pickleable(cls);
    }

    template <typename T>
    void bind_ssop_common(py::class_<ttns::sSOP<T>> &cls)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;

        using namespace ttns;
        using namespace literal;
        using _SOP = sSOP<T>;
        using container_type = typename _SOP::container_type;

        cls.def(py::init())
            .def(py::init<size_t>())
            .def(py::init<const std::string &>())
            .def(py::init<const sOP &>())
            .def(py::init<const sPOP &>())
            .def(py::init<const sNBO<T> &>())
            .def(py::init<const _SOP &>())
            .def("clear", &_SOP::clear)
            .def("reserve", &_SOP::reserve)
            .def("nmodes", &_SOP::nmodes)
            .def("nterms", &_SOP::nterms)
            .def("__len__", &_SOP::nterms)
            .def("__iter__", [](_SOP &s){ return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
            .def("__setitem__", [](_SOP &self, size_t i, const sNBO<T> &v){ self[i] = v; })
            .def("__getitem__", static_cast<sNBO<T> &(_SOP::*)(size_t)>(&_SOP::operator[]), py::return_value_policy::reference)
            .def("__str__", [](const _SOP &o){std::ostringstream oss; oss << o; return oss.str(); });
                
        BIND_RW_PROPERTY(cls, _SOP, std::string, label, nullptr);
        BIND_RW_PROPERTY(cls, _SOP, container_type, terms, nullptr);

        bind_all<bind_iadd, _SOP, sOP, sPOP, sNBO<T>, _SOP>(cls);
        bind_all<bind_isub, _SOP, sOP, sPOP, sNBO<T>, _SOP>(cls);
        bind_all<bind_imul, _SOP, real_type, sOP, sPOP, sNBO<T>, _SOP>(cls);
        bind_idiv<real_type>::apply(cls);
        bind_add_sub_sop<real_type>(cls);
        bind_mul_div_sop<real_type>(cls);
        bind_dtype<T>(cls);
        bind_utils(cls);
        bind_copyable(cls);
        bind_pickleable(cls);
    }
}

template <typename real_type>
void init_sSOP(py::module &m)
{
    using complex_type = std::complex<real_type>;
    using namespace ttns;
    using namespace literal;
    // wrapper for the sOP type

    {
        auto cls = py::class_<sOP>(m, "sOP");
        cls.def(py::init())
            .def(py::init<const std::string &, size_t>())
            .def(py::init<const std::string &, size_t, bool>())
            .def(py::init<const sOP &>())
            .def("clear", &sOP::clear, "Clear the sOPs mode and label information.")
            .def("__str__", [](const sOP &o){ return static_cast<std::string>(o); });

        BIND_RW_PROPERTY(cls, sOP, std::string, op, nullptr);
        BIND_RW_PROPERTY(cls, sOP, size_t, mode, nullptr);
        BIND_RW_PROPERTY(cls, sOP, bool, fermionic, nullptr);

        python_bindings::bind_add_sub_sop<real_type>(cls);
        python_bindings::bind_mul_div_sop<real_type>(cls);
        python_bindings::bind_todense<real_type>(cls);
        python_bindings::bind_utils(cls);
        python_bindings::bind_copyable(cls);
        python_bindings::bind_pickleable(cls);

        cls.doc() = R"mydelim(
                    The single site operator used for the string operator handling functionality of pyTTN.  This class allows for definition of 
                    a string label for an operator and the mode that the operator acts upon. In addition to allowing for arbitrary string labels
                    with the combination of user defined operator dictionaries.  This code supports several automatic dictionaries depending on
                    the type of mode considered.  These are

                    :Fermion Modes:
                    - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "f"}
                    - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
                    - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
                    - Vacancy operator :math:`1-\\hat{c}^\\dagger\\hat{c}` :  "v"

                    :Bosonic Modes:
                    - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "b"}
                    - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "bdag", "cd", "ad", "bd"}
                    - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "bdagb", "cdc", "ada", "bdb"}
                    - Position operator :math:`\\hat{q}` : {"q", "x"}
                    - Momentum opeartor :math:`\\hat{p}` : "p"

                    :Spin Modes for arbitrary spin S:
                    - :math:`\\hat{S}_x` : {"sx", "x"}
                    - :math:`\\hat{S}_y` : {"sy", "y"}
                    - :math:`\\hat{S}_z` : {"sz", "z"}
                    - :math:`\\hat{S}_+` : {"s+", "sp"}
                    - :math:`\\hat{S}_-` : {"s-", "sm"}

                    :Two Level System Modes:
                    - :math:`\\hat{\\sigma}_x` : {"sx", "x", "sigmax"}
                    - :math:`\\hat{\\sigma}_y` : {"sy", "y", "sigmay"}
                    - :math:`\\hat{\\sigma}_z` : {"sz", "z", "sigmaz"}
                    - :math:`\\hat{\\sigma}_+` : {"s+", "sp", "sigma+", "sigmap"}
                    - :math:`\\hat{\\sigma}_-` : {"s-", "sm", "sigma-", "sigmam"}
               )mydelim";
    }

    m.def("fermion_operator", &fermion_operator, R"mydelim(
      Create a new site operator string where the operator is a Fermionic operator.

      :param arg0: The operator label associated with this operator.   
      :type arg0: str
      :param arg1: The mode this operator acts upon.
      :type arg1: int

      For fermionic systems the following operator are supported:
        - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "f"}
        - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
        - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
        - Vacancy operator :math:`1-\\hat{c}^\\dagger\\hat{c}` :  {"v"}

      :returns: fermionic mode data object
      :rtype: mode_data
      )mydelim");

    m.def("fOP", &fermion_operator, R"mydelim(
      Create a new site operator string where the operator is a Fermionic operator.

      :param arg0: The operator label associated with this operator.   
      :type arg0: str
      :param arg1: The mode this operator acts upon.
      :type arg1: int

      For fermionic systems the following operator are supported:
        - Annihilation operator :math:`\\hat{c}` :  {"c", "a", "f"}
        - Creation operator :math:`\\hat{c}^\\dagger` :  {"cdag", "adag", "fdag", "cd", "ad", "fd"}
        - Number operator :math:`\\hat{c}^\\dagger\\hat{c}` :  {"n", "cdagc", "adaga", "fdagf", "cdc", "ada", "fdf"}
        - Vacancy operator :math:`1-\\hat{c}^\\dagger\\hat{c}` :  {"v"}

      :returns: fermionic mode data object
      :rtype: mode_data
      )mydelim");

    {
        // wrapper for the sPOP typ

        auto cls = py::class_<sPOP>(m, "sPOP");
        cls.def(py::init())
            .def(py::init<const sOP &>())
            .def(py::init<const std::list<sOP> &>())
            .def(py::init<const sPOP &>())
            .def("clear", &sPOP::clear, "empty the internal buffer storing the sOPs in the sPOP")
            .def("insert_front", &sPOP::append, R"mydelim(

                    :param o: Insert a sOP object at the front of this product.
                    :type o: sOP
               )mydelim")
            .def("insert_back", &sPOP::prepend, R"mydelim(

                    :param o: Insert a sOP object at the back of this product.
                    :type o: sOP
               )mydelim")
            .def("size", &sPOP::size, R"mydelim(

                    :returns: The number of individual sOP terms in this product.
                    :rtype: int
               )mydelim")
            .def("nmodes", &sPOP::nmodes, R"mydelim(

                    :returns: The number of modes that this sPOP acts on
                    :rtype: int
               )mydelim")
            .def("__iter__", [](sPOP &s){ return py::make_iterator(s.begin(), s.end()); }, py::keep_alive<0, 1>())
            .def("__str__", [](const sPOP &o){ return static_cast<std::string>(o); });;

        BIND_RW_PROPERTY(cls, sPOP, std::list<sOP>, ops, "A list of the individual sOP objects forming the sPOP.");


        python_bindings::bind_add_sub_sop<real_type>(cls);
        python_bindings::bind_mul_div_sop<real_type>(cls);
        python_bindings::bind_all<python_bindings::bind_imul, sPOP, sOP, sPOP>(cls);
        python_bindings::bind_todense<real_type>(cls);
        python_bindings::bind_utils(cls);
        python_bindings::bind_copyable(cls);
        python_bindings::bind_pickleable(cls);

        cls.doc() = R"mydelim(
               A product of sOP operators. This function handles the fact that in general sOP operators do not commute.
          )mydelim";
    }

    {
        using coef = coeff<real_type>;
        // using complex_func_type = std::function<complex_type(real_type)>;
        using func_type = std::function<real_type(real_type)>;
        auto cls = py::class_<coef>(m, "coeff_real");
        python_bindings::bind_coeff_common(cls);
    }
    {
        using coef = coeff<complex_type>;
        using real_func_type = std::function<real_type(real_type)>;
        auto cls = py::class_<coef>(m, "coeff_complex");
        python_bindings::bind_coeff_common(cls);

        cls.def(py::init([](const complex_type &o)
                         { return coef(complex_type(o)); }))
            .def(py::init<const real_func_type &>())
            .def(py::init<const coeff<real_type> &>())
            .def("assign", [](coef &self, const coeff<real_type> &o){ self = o; })
            .def("assign", [](coef &self, const complex_type &o){ self = complex_type(o); })
            .def("assign", [](coef &self, const real_func_type &o){ self = coeff<real_type>(o); });
            python_bindings::bind_all<python_bindings::bind_iadd, coef, complex_type, coeff<real_type>>(cls);
            python_bindings::bind_all<python_bindings::bind_isub, coef, complex_type, coeff<real_type>>(cls);
            python_bindings::bind_all<python_bindings::bind_imul, coef, complex_type, coeff<real_type>>(cls);
            python_bindings::bind_idiv<complex_type>::apply(cls);
    }

    {
        using NBO = sNBO<real_type>;
        // wrapper for the sPOP type
        auto cls = py::class_<NBO>(m, "sNBO_real");
        python_bindings::bind_nbo_common(cls);
        cls.def(py::init<const real_type &, const sPOP &>());
        cls.def(py::init<const real_type &, const sOP &>());
        python_bindings::bind_todense<real_type>(cls);
    }
    {
        using NBO = sNBO<complex_type>;
        // wrapper for the sPOP type
        auto cls = py::class_<NBO>(m, "sNBO_complex");
        python_bindings::bind_nbo_common(cls);

        cls.def(py::init([](const complex_type &coeff, const sPOP &o){ return NBO(complex_type(coeff), o); }));
        cls.def(py::init([](const complex_type &coeff, const sOP &o){ return NBO(complex_type(coeff), o); }));
        cls.def(py::init<const sNBO<real_type> &>());
        cls.def("assign", [](NBO &self, const sNBO<real_type> &o){ self = o; });
        python_bindings::bind_imul<complex_type>::apply(cls);
        python_bindings::bind_todense<complex_type>(cls);
    }

    {
        using _SOP = sSOP<real_type>;
        // wrapper for the sPOP type
        auto cls = py::class_<_SOP>(m, "sSOP_real");
        python_bindings::bind_ssop_common(cls);
        python_bindings::bind_todense<real_type>(cls);
    }
    {
        using _SOP = sSOP<complex_type>;
        // wrapper for the sPOP type
        auto cls = py::class_<_SOP>(m, "sSOP_complex");
        python_bindings::bind_ssop_common(cls);
        cls.def(py::init<const sNBO<real_type> &>())
            .def(py::init<const sSOP<real_type> &>())
            .def("assign", [](_SOP &self, const sSOP<real_type> &o){ self = o; });

        python_bindings::bind_all<python_bindings::bind_iadd, _SOP, sNBO<real_type>, sSOP<real_type>>(cls);
        python_bindings::bind_all<python_bindings::bind_isub, _SOP, sNBO<real_type>, sSOP<real_type>>(cls);
        python_bindings::bind_idiv<complex_type>::apply(cls);
        python_bindings::bind_all<python_bindings::bind_imul, _SOP, complex_type, sNBO<real_type>, sSOP<real_type>>(cls);
        python_bindings::bind_todense<complex_type>(cls);
    }
}

void initialise_sSOP(py::module &m)
{
    init_sSOP<pyttn_real_type>(m);
}
