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

#ifndef PYTHON_BINDING_TTNS_OP_HPP
#define PYTHON_BINDING_TTNS_OP_HPP

#include "../../utils.hpp"
#include <ttns_lib/op.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py = pybind11;

template <typename T, typename backend>
void init_Op(py::module &m, const std::string &label)
{
    using namespace ttns;

    using Optype = Op<T, backend>;
    using real_type = typename linalg::get_real_type<T>::type;
    using conv = linalg::pybuffer_converter<backend>;    
    using mat = linalg::matrix<T, backend>;

#ifdef PYTTN_BUILD_CUDA_NO
     using otherbackend = typename other_backend<backend>::type;
#endif

    // the base primitive operator type
    py::class_<Optype>(m, (std::string("Op_") + label).c_str())
        .def(py::init())
        .def(py::init<const Optype &>())
        //.def(py::init<const Op<real_type, backend> &>())
 #ifdef PYTTN_BUILD_CUDA_NO
        .def(py::init<const Op<T, otherbackend> &>())
        //.def(py::init<const Op<real_type, otherbackend> &>())
#endif       
        .def(py::init<const mat &, const std::vector<size_t>&, const std::vector<size_t>&>())
        .def(py::init([](py::buffer& b, const std::vector<size_t>& inds, const std::vector<size_t>& dim)
                        {
                            mat _m;
                            conv::copy_to_tensor(b, _m);
                            return Optype(_m, inds, dim); 
                        }
                    )   
                )
        .def("assign", &Optype::template operator= <T, backend>)
        //.def("assign", &Optype::template operator= <real_type, backend>)
 #ifdef PYTTN_BUILD_CUDA_NO
        .def("assign", &Optype::template operator= <T, otherbackend>)
        //.def("assign", &Optype::template operator= <real_type, otherbackend>)
#endif       
        .def("clear", &Optype::clear)
        .def("complex_dtype", [](const Optype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const Optype &o)
             {std::ostringstream oss; oss << o; return oss.str(); })
        .def("backend", [](const Optype &)
             { return linalg::traits<backend>::label(); })
        .def("set_operator", 
            [](Optype& op, const mat& _m)
            {
                if(_m.shape(0) != _m.shape(1) || _m.shape(0) != op.size())
                {
                    RAISE_EXCEPTION("Failed to set operator value.  Matrix is not compatible with input buffer.");
                }
                op.matrix() = _m;
            }
        )
        .def("set_operator", 
            [](Optype& op, py::buffer &b)
            {
                mat _m;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, _m));
                if(_m.shape(0) != _m.shape(1) || _m.shape(0) != op.size())
                {
                    RAISE_EXCEPTION("Failed to set operator value.  Matrix is not compatible with input buffer.");
                }
                op.matrix() = _m;
            }
        )
        .def_property("operator",
            [](const Optype& op){return op.matrix();},
            [](Optype& op, const mat& _m){op.matrix() = _m;}
        )
        .def_property("indices",
            [](const Optype& op){return op.indices();},
            [](Optype& op, const std::vector<size_t>& _m){op.indices() = _m;}
        )
        .def_property("dims",
            [](const Optype& op){return op.dims();},
            [](Optype& op, const std::vector<size_t>& _m){op.dims() = _m;}
        )
        .def("nmodes", &Optype::nmodes)
        .def("ndim", &Optype::ndim)
        .def("size", &Optype::size)
        .def("as_mpo", &Optype::as_mpo, py::arg("nbmax")=-1, py::arg("tol")=-1.0);
}

template <typename real_type, typename backend>
void initialise_Op(py::module &m)
{
    using complex_type = std::complex<real_type>;

#ifdef BUILD_REAL_TTN
    init_Op<real_type, backend>(m, "real");
#endif
    init_Op<complex_type, backend>(m, "complex");
}

#endif
