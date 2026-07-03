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

#ifndef PYTHON_BINDING_HELPERS_HPP_
#define PYTHON_BINDING_HELPERS_HPP_

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>

#include "utils.hpp"

namespace python_bindings
{


    struct add_op
    {
        static constexpr const char* name = "__add__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return a + b;
        }
    };
    struct radd_op
    {
        static constexpr const char* name = "__radd__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return b+a;
        }
    };
    struct iadd_op
    {
        static constexpr const char* name = "__iadd__";
        template<typename A, typename B>
        static auto apply(A& a, const B& b)
        {
            return a+=b;
        }
    };
    struct sub_op
    {
        static constexpr const char* name = "__sub__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return a-b;
        }
    };
    struct rsub_op
    {
        static constexpr const char* name = "__rsub__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return b-a;
        }
    };
    struct isub_op
    {
        static constexpr const char* name = "__isub__";
        template<typename A, typename B>
        static auto apply(A& a, const B& b)
        {
            return a-=b;
        }
    };
    struct mul_op
    {
        static constexpr const char* name = "__mul__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return a*b;
        }
    };
    struct rmul_op
    {
        static constexpr const char* name = "__rmul__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return b*a;
        }
    };
    struct imul_op
    {
        static constexpr const char* name = "__imul__";
        template<typename A, typename B>
        static auto apply(A& a, const B& b)
        {
            return a*=b;
        }
    };
    struct div_op
    {
        static constexpr const char* name = "__div__";
        template<typename A, typename B>
        static auto apply(const A& a, const B& b)
        {
            return a*(1.0/b);
        }
    };
    struct idiv_op
    {
        static constexpr const char* name = "__idiv__";
        template<typename A, typename B>
        static auto apply(A& a, const B& b)
        {
            return a*=(1.0/b);
        }
    };

    template <typename op, typename right_type>
    struct bind_binary
    {
        template <typename CLS>
        static inline void apply(py::class_<CLS> &c, const char *doc = nullptr)
        {
            if (doc)
            {
                c.def(op::name, [](CLS &a, const right_type &b)
                      { return op::apply(a, b); }, doc);
            }
            else
            {
                c.def(op::name, [](CLS &a, const right_type &b)
                      { return op::apply(a, b); });
            }
        }
    };

    template <typename right_type> using bind_add = bind_binary<add_op, right_type>;
    template <typename right_type> using bind_radd = bind_binary<radd_op, right_type>;
    template <typename right_type> using bind_iadd = bind_binary<iadd_op, right_type>;
    template <typename right_type> using bind_sub = bind_binary<sub_op, right_type>;
    template <typename right_type> using bind_rsub = bind_binary<rsub_op, right_type>;
    template <typename right_type> using bind_isub = bind_binary<isub_op, right_type>;
    template <typename right_type> using bind_mul = bind_binary<mul_op, right_type>;
    template <typename right_type> using bind_rmul = bind_binary<rmul_op, right_type>;
    template <typename right_type> using bind_imul = bind_binary<imul_op, right_type>;
    template <typename right_type> using bind_div = bind_binary<div_op, right_type>;
    template <typename right_type> using bind_idiv = bind_binary<idiv_op, right_type>;

    template <template <typename> class Func, typename... RHS>
    void bind_all(py::module &m, const char *doc = nullptr)
    {
        constexpr std::size_t N = sizeof...(RHS);
        size_t i = 0;
        (Func<RHS>::apply(m, (++i == N ? doc : nullptr)), ...);
    }

    template <template <typename, typename> class Func, typename Fixed, typename... RHS>
    void bind_all(py::module &m, const char *doc = nullptr)
    {
        constexpr std::size_t N = sizeof...(RHS);
        size_t i = 0;
        (Func<Fixed, RHS>::apply(m, (++i == N ? doc : nullptr)), ...);
    }

    template <template <typename> class Func, typename CLS, typename... RHS>
    void bind_all(py::class_<CLS> &c, const char *doc = nullptr)
    {
        constexpr std::size_t N = sizeof...(RHS);
        size_t i = 0;

        (Func<RHS>::apply(c, (++i == N ? doc : nullptr)), ...);
    }

    template <template <typename, typename> class Func, typename CLS, typename Fixed, typename... RHS>
    void bind_all(py::class_<CLS> &c, const char *doc = nullptr)
    {
        constexpr std::size_t N = sizeof...(RHS);
        size_t i = 0;

        (Func<Fixed, RHS>::apply(c, (++i == N ? doc : nullptr)), ...);
    }

    template <typename T, typename CLS>
    typename std::enable_if<std::is_same_v<typename linalg::get_real_type<T>::type, T>,void>::type bind_todense(py::class_<CLS> &c)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;
        using namespace ttns;
        using namespace literal;
        c.def("_todense",
              [](const CLS &op, const system_modes &sys)
              {
                  linalg::matrix<complex_type> mat;
                  CALL_AND_HANDLE(convert_to_dense(op, sys, mat), "Failed to convert OPBase to dense matrix.");
                  return mat;
              });
        c.def("_todense",
              [](const CLS &op, const system_modes &sys, const operator_dictionary<real_type, linalg::blas_backend> &dict)
              {
                  linalg::matrix<real_type> mat;
                  CALL_AND_HANDLE(convert_to_dense(op, sys, dict, mat), "Failed to convert OPBase to dense matrix.");
                  return mat;
              });
        c.def("_todense",
              [](const CLS &op, const system_modes &sys, const operator_dictionary<complex_type, linalg::blas_backend> &dict)
              {
                  linalg::matrix<complex_type> mat;
                  CALL_AND_HANDLE(convert_to_dense(op, sys, dict, mat), "Failed to convert OPBase to dense matrix.");
                  return mat;
              });
    }

    template <typename T, typename CLS>
    typename std::enable_if<!std::is_same_v<typename linalg::get_real_type<T>::type, T>,void>::type bind_todense(py::class_<CLS> &c)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;
        using namespace ttns;
        using namespace literal;
        c.def("_todense",
              [](const CLS &op, const system_modes &sys)
              {
                  linalg::matrix<complex_type> mat;
                  CALL_AND_HANDLE(convert_to_dense(op, sys, mat), "Failed to convert OPBase to dense matrix.");
                  return mat;
              });
        c.def("_todense",
              [](const CLS &op, const system_modes &sys, const operator_dictionary<complex_type, linalg::blas_backend> &dict)
              {
                  linalg::matrix<complex_type> mat;
                  CALL_AND_HANDLE(convert_to_dense(op, sys, dict, mat), "Failed to convert OPBase to dense matrix.");
                  return mat;
              });
    }

    template <typename CLS>
    void bind_utils(py::class_<CLS> &c)
    {
        c.def_property_readonly("__array_priority__", [](const CLS &)
                                { return 10000.0; });
    }

    template <typename CLS>
    void bind_copyable(py::class_<CLS> &c)
    {
        c.def("assign", [](CLS &self, const CLS &o)
              { self = o; });
        c.def("__copy__", [](const CLS &o)
              { return CLS(o); });
        c.def("__deepcopy__", [](const CLS &o, py::dict)
              { return CLS(o); }, py::arg("memo"));
    }

    template <typename CLS>
    void bind_pickleable(py::class_<CLS> &c)
    {
#ifdef CEREAL_LIBRARY_FOUND
        c.def("save", [](const CLS &a, const std::string &ofname, bool as_binary)
              { serialisation_utilities::save_obj(a, ofname, as_binary); }, py::arg(), py::arg("as_binary") = true);
        c.def("load", [](CLS &a, const std::string &ifname, bool as_binary)
              { serialisation_utilities::load_obj(a, ifname, as_binary); }, py::arg(), py::arg("as_binary") = true);
        c.def(py::pickle([](const CLS &a)
                         { return serialisation_utilities::__getstate__(a); }, [](py::tuple t)
                         { return serialisation_utilities::__setstate__<CLS>(t); }));
#endif
    }

    template <typename T, typename CLS>
    void bind_dtype(py::class_<CLS>& cls)
    {
        using real_type = typename linalg::get_real_type<T>::type;
        using complex_type = std::complex<real_type>;
        cls.def_property_readonly("dtype", [](const CLS &){if constexpr (std::is_same<T, real_type>::value){return py::dtype::of<real_type>();}else{return py::dtype::of<T>();} });
        cls.def("complex_dtype", [](const CLS &){ return !std::is_same<T, real_type>::value; });
    } 

    template<typename T, typename CLS, typename Getter, typename Setter>
    void bind_rw_property(py::class_<CLS>& cls, const char* name, Getter get, Setter set, const char* doc = nullptr)
    {
        if(doc){cls.def_property(name, static_cast<const T&(CLS::*)() const>(get), [set](CLS& o, const T& i){(o.*set)() = i;}, doc);}
        else{cls.def_property(name, static_cast<const T&(CLS::*)() const>(get),[set](CLS& o, const T& i){(o.*set)() = i;});}
    }
}


#define RW_PROP(CLASS, TYPE, MEMBER) \
    static_cast<const TYPE&(CLASS::*)() const>(&CLASS::MEMBER), \
    static_cast<TYPE&(CLASS::*)()>(&CLASS::MEMBER)

#define BIND_RW_PROPERTY(cls, CLS, TYPE, MEMBER, DOC)          \
    python_bindings::bind_rw_property<TYPE>(                   \
        cls,                                                   \
        #MEMBER,                                               \
        static_cast<const TYPE&(CLS::*)() const>(&CLS::MEMBER),\
        static_cast<TYPE&(CLS::*)()>(&CLS::MEMBER),            \
        DOC)

#endif // PYTHON_BINDING_HELPERS_HPP_