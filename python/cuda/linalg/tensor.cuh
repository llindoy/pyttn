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

#ifndef PYTHON_BINDING_LINALG_TENSOR_CUH
#define PYTHON_BINDING_LINALG_TENSOR_CUH

#include <linalg/linalg.cuh>
#include <linalg/linalg.hpp>

#include "../../linalg/tensor.hpp"
#include "../../linalg/tensor.tpp"

namespace py = pybind11;

template <typename T>
void init_matrix_gpu(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = linalg::cuda_backend;
    using ttype = tensor<T, 2, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    // expose the ttn node class.  This is our core tensor network object.
    py::class_<ttype>(m, (label).c_str())
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens; }),
             R"mydelim(
            Construct a cuda linear algebra tensor object from a python buffer object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :param in: The Input numpy array buffer
            :type in: np.ndarray            
            )mydelim")
        .def(py::init<const ttype &>(),
             R"mydelim(
            Construct an empty cuda linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.
            )mydelim")
        .def(py::init<const tensor<T, 2, linalg::blas_backend> &>(),
             R"mydelim(
            Construct a cuda linear algebra tensor object from a linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :param in: The Input linear algebra array buffer
            )mydelim")
        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__matmul__",
             [](const ttype &a, ttype &b)
             {
                 ttype ret;
                 ret = a * b;
                 return ret;
             })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("shape", [](const ttype& o, size_t i){return o.shape(i);})
        .def("ndim", [](const ttype&){return 2;})
        .def("set_subblock", [](ttype &o, py::buffer &b)
             {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                o.set_subblock(tens); })
        .def("transpose",
             [](const ttype &o, const std::vector<int> &inds)
             {
                 ttype b = linalg::transpose(o, inds);
                 return b;
             })
        .def("transpose",
             [](const ttype &o)
             {
                 ttype b = linalg::trans(o);
                 return b;
             })
#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const ttype & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](ttype & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const ttype& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<ttype>(t);}
         ))
#endif

        .def("backend", [](const ttype &)
             { return linalg::traits<backend>::label(); })
        .def("clear", &ttype::clear);
}

template <typename T, size_t D>
void init_tensor_gpu(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = linalg::cuda_backend;
    using ttype = tensor<T, D, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    // expose the ttn node class.  This is our core tensor network object.
    py::class_<ttype>(m, (label).c_str())
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens; }),
             R"mydelim(
            Construct a cuda linear algebra tensor object from a python buffer object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :param in: The Input numpy array buffer
            :type in: np.ndarray            
            )mydelim")
        .def(py::init<const ttype &>(),
             R"mydelim(
            Construct an empty cuda linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.
            )mydelim")
        .def(py::init<const tensor<T, D, linalg::blas_backend> &>(),
             R"mydelim(
            Construct a cuda linear algebra tensor object from a linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :param in: The Input linalg array buffer
            )mydelim")
        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("ndim", [](const ttype&){return D;})

        .def("shape", [](const ttype& o, size_t i){return o.shape(i);})
        .def("transpose",
             [](const ttype &o, const std::vector<int> &inds)
             {
                 ttype b = linalg::transpose(o, inds);
                 return b;
             })
#ifdef CEREAL_LIBRARY_FOUND
         .def("save", 
            [](const ttype & a, const std::string& ofname, bool as_binary){serialisation_utilities::save_obj(a, ofname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
        .def("load", 
            [](ttype & a, const std::string& ifname, bool as_binary){serialisation_utilities::load_obj(a, ifname, as_binary);},
            py::arg(), py::arg("as_binary")=true)
         .def(py::pickle(
            [](const ttype& a){return serialisation_utilities::__getstate__(a);},
            [](py::tuple t){return serialisation_utilities::__setstate__<ttype>(t);}
         ))
#endif

        .def("backend", [](const ttype &)
             { return linalg::traits<backend>::label(); })
        .def("clear", &ttype::clear);
}

#endif // PYTHON_BINDING_LINALG_TENSOR_HPP
