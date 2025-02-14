#ifndef PYTHON_BINDING_LINALG_TENSOR_HPP
#define PYTHON_BINDING_LINALG_TENSOR_HPP

#include <linalg/linalg.hpp>

#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_matrix_cpu(py::module &m, const std::string& label)
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
                return tens;
            }
        ),          
            R"mydelim(
            Construct a linear algebra tensor object from a python buffer object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :Parameters:    - **in** (:class:`np.ndarray`) - The Input numpy array buffer
            )mydelim"
        )
        .def(py::init<const ttype&>(),          
            R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<const tensor<T, 2, linalg::cuda_backend>&>(),          
            R"mydelim(
            Construct a linear algebra tensor object from a cuda linear algebra object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :Parameters:    - **in** - The Input cuda linear algebra type
            )mydelim")
#endif
        .def_buffer([](ttype& mi) -> py::buffer_info 
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
                        );  
                    }
                   )
        .def("set_subblock", [](ttype& o, py::buffer &b)
            {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                o.set_subblock(tens);
            })
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("clear", &ttype::clear)
        .def("transpose", 
            [](const ttype& o, const std::vector<int>& inds)
            {
                ttype b = linalg::transpose(o, inds);
                return b;
            })
        .def("backend", [](const ttype&){return backend::label();});
}

template <typename T, size_t D>
void init_tensor_cpu(py::module &m, const std::string& label)
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
                return tens;
            }
        ),          
            R"mydelim(
            Construct a linear algebra tensor object from a python buffer object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :Parameters:    - **in** (:class:`np.ndarray`) - The Input numpy array buffer
            )mydelim"
        )
        .def(py::init<const ttype&>(),          
            R"mydelim(
            Construct an empty linear algebra tensor object.  This is the internal type used for linear
            algebra operations by the pyTTN package.
            )mydelim")
#ifdef PYTTN_BUILD_CUDA
        .def(py::init<const tensor<T, D, linalg::cuda_backend>&>(),          
            R"mydelim(
            Construct a linear algebra tensor object from a cuda linear algebra object.  This is the internal type used for linear
            algebra operations by the pyTTN package.

            :Parameters:    - **in** - The Input cuda linear algebra type
            )mydelim")
#endif
        .def_buffer([](ttype& mi) -> py::buffer_info 
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
                        );  
                    }
                   )
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("clear", &ttype::clear)
        .def("transpose", 
            [](const ttype& o, const std::vector<int>& inds)
            {
                ttype b = linalg::transpose(o, inds);
                return b;
            })
        .def("backend", [](const ttype&){return backend::label();});

}

template <typename real_type> void initialise_tensors(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
    init_tensor_cpu<real_type, 1>(m, "vector_real");     
    init_matrix_cpu<real_type>(m, "matrix_real");     
    init_tensor_cpu<real_type, 3>(m, "tensor_3_real");     
    init_tensor_cpu<real_type, 4>(m, "tensor_4_real");     

    init_tensor_cpu<complex_type, 1>(m, "vector_complex");     
    init_matrix_cpu<complex_type>(m, "matrix_complex");     
    init_tensor_cpu<complex_type, 3>(m, "tensor_3_complex");     
    init_tensor_cpu<complex_type, 4>(m, "tensor_4_complex");     
}

#ifdef PYTTN_BUILD_CUDA

template <typename T>
void init_matrix_gpu(py::module &m, const std::string& label)
{
    using namespace linalg;
    using backend = linalg::cuda_backend;
    using ttype = tensor<T, 2, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    //expose the ttn node class.  This is our core tensor network object.
    py::class_<ttype>(m, (label).c_str())
        .def(py::init([](py::buffer &b)
            {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens;
            }
        ),          
            R"mydelim(
            Construct a cuda linear algebra tensor object from a python buffer object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :Parameters:    - **in** (:class:`np.ndarray`) - The Input numpy array buffer
            )mydelim"
        )
        .def(py::init<const ttype&>(),          
            R"mydelim(
            Construct an empty cuda linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.
            )mydelim"
        )
        .def(py::init<const tensor<T, 2, linalg::blas_backend>&>(),
            R"mydelim(
            Construct a cuda linear algebra tensor object from a linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :Parameters:    - **in**  - The input linear algebra tensor object
            )mydelim"
        )
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("set_subblock", [](ttype& o, py::buffer &b)
            {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                o.set_subblock(tens);
            })
        .def("transpose", 
            [](const ttype& o, const std::vector<int>& inds)
            {
                ttype b = linalg::transpose(o, inds);
                return b;
            })
        .def("backend", [](const ttype&){return backend::label();});
         //.def("clear", &ttype::clear);
}

template <typename T, size_t D>
void init_tensor_gpu(py::module &m, const std::string& label)
{
    using namespace linalg;
    using backend = linalg::cuda_backend;
    using ttype = tensor<T, D, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    //expose the ttn node class.  This is our core tensor network object.
    py::class_<ttype>(m, (label).c_str())
        .def(py::init([](py::buffer &b)
            {
                ttype tens;
                CALL_AND_RETHROW(conv::copy_to_tensor(b, tens));
                return tens;
            }
        ),          
            R"mydelim(
            Construct a cuda linear algebra tensor object from a python buffer object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :Parameters:    - **in** (:class:`np.ndarray`) - The Input numpy array buffer
            )mydelim"
        )
        .def(py::init<const ttype&>(),          
            R"mydelim(
            Construct an empty cuda linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.
            )mydelim"
        )
        .def(py::init<const tensor<T, D, linalg::blas_backend>&>(),
            R"mydelim(
            Construct a cuda linear algebra tensor object from a linear algebra tensor object.  This is the internal type used for 
            cuda accelerated linear algebra operations by the pyTTN package.

            :Parameters:    - **in**  - The input linear algebra tensor object
            )mydelim"
        )
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();})
        .def("transpose", 
            [](const ttype& o, const std::vector<int>& inds)
            {
                ttype b = linalg::transpose(o, inds);
                return b;
            })
        .def("backend", [](const ttype&){return backend::label();});
         //.def("clear", &ttype::clear);
}

template <typename real_type> void initialise_tensors_cuda(py::module& m)
{
    using complex_type = linalg::complex<real_type>;
    init_tensor_gpu<real_type, 1>(m, "vector_real");     
    init_matrix_gpu<real_type>(m, "matrix_real");     
    init_tensor_gpu<real_type, 3>(m, "tensor_3_real");     
    init_tensor_gpu<real_type, 4>(m, "tensor_4_real");     

    init_tensor_gpu<complex_type, 1>(m, "vector_complex");     
    init_matrix_gpu<complex_type>(m, "matrix_complex");     
    init_tensor_gpu<complex_type, 3>(m, "tensor_3_complex");     
    init_tensor_gpu<complex_type, 4>(m, "tensor_4_complex");     
}
#endif



#endif  //PYTHON_BINDING_LINALG_TENSOR_HPP


