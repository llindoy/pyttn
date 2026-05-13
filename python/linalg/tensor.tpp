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

#ifndef PYTHON_BINDING_LINALG_TENSOR_TPP
#define PYTHON_BINDING_LINALG_TENSOR_TPP

#include "tensor.hpp"

namespace py = pybind11;


template <typename T>
void init_matrix_cpu(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = linalg::blas_backend;
    using ttype = tensor<T, 2, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using _T = typename numpy_converter<T>::type;

    using conv = pybuffer_converter<backend>;

    py::class_<ttype>(m, (label).c_str(), py::buffer_protocol())
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens; }),
             R"mydelim(
            Construct a linear algebra tensor object from a python buffer object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :param in: The Input numpy array buffer
            :type in: np.ndarray
            )mydelim")
        .def(py::init<const ttype &>(),
             R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
        .def(py::init<const linalg::tensor<real_type, 2, backend> &>(),
             R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<const tensor<T, 2, linalg::cuda_backend> &>(),
             R"mydelim(
            Construct a linear algebra tensor object from a cuda linear algebra object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :param in: The Input cuda array buffer
            )mydelim")
#endif
        .def_buffer([](ttype &mi) -> py::buffer_info
                    {
                        std::vector<py::ssize_t> stride_arr(2); 
                        std::vector<py::ssize_t> shape_arr(2);
                        for(size_t i = 0; i < 2; ++i)
                        {
                            shape_arr[i] = mi.shape(i);
                            stride_arr[i] = mi.stride(i)*sizeof(T);
                        }
                        return py::buffer_info
                        (
                            mi.buffer(),                             //pointer to buffer
                            sizeof(T),                              //size of one scalar
                            py::format_descriptor<_T>::format(),     //Python struct-style format descriptor
                            2,                                      //Number of dimensions D
                            shape_arr,                              //shape of the array
                            stride_arr                              //strides of the array
                        ); })
        .def("set_subblock", [](ttype &o, py::buffer &b)
             {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                o.set_subblock(tens); })
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
        .def("clear", &ttype::clear)
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
             { return linalg::traits<backend>::label(); });
}

template <typename T, size_t D>
void init_tensor_cpu(py::module &m, const std::string &label)
{
    using namespace linalg;
    using backend = linalg::blas_backend;
    using ttype = tensor<T, D, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using _T = typename numpy_converter<T>::type;

    using conv = pybuffer_converter<backend>;

    py::class_<ttype>(m, (label).c_str(), py::buffer_protocol())
        .def(py::init([](py::buffer &b)
                      {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens; }),
             R"mydelim(
            Construct a linear algebra tensor object from a python buffer object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :param in: The Input numpy array buffer
            :type in: np.ndarray            
            )mydelim")
        .def(py::init<const ttype &>(),
             R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
        .def(py::init<const linalg::tensor<real_type, D, backend> &>(),
             R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<const tensor<T, D, linalg::cuda_backend> &>(),
             R"mydelim(
            Construct a linear algebra tensor object from a cuda linear algebra object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :param in: The Input cuda array buffer
            )mydelim")
#endif
        .def_buffer([](ttype &mi) -> py::buffer_info
                    {
                        std::vector<py::ssize_t> stride_arr(D); 
                        std::vector<py::ssize_t> shape_arr(D);
                        for(size_t i = 0; i < D; ++i)
                        {
                            shape_arr[i] = mi.shape(i);
                            stride_arr[i] = mi.stride(i)*sizeof(T);
                        }
                        return py::buffer_info
                        (
                            mi.buffer(),                             //pointer to buffer
                            sizeof(T),                              //size of one scalar
                            py::format_descriptor<_T>::format(),     //Python struct-style format descriptor
                            D,                                      //Number of dimensions D
                            shape_arr,                              //shape of the array
                            stride_arr                              //strides of the array
                        ); })
        .def("complex_dtype", [](const ttype &)
             { return !std::is_same<T, real_type>::value; })
        .def("__str__", [](const ttype &o)
             {std::stringstream oss;   oss << o; return oss.str(); })
        .def("ndim", [](const ttype&){return D;})
        .def("clear", &ttype::clear)
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
             { return linalg::traits<backend>::label(); });
}

#endif //PYTHON_BINDING_LINALG_TENSOR_TPP
