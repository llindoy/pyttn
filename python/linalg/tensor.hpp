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

template <typename T, size_t D>
void init_tensor(py::module &m, const std::string& label)
{
    using namespace linalg;
    using ttype = tensor<T, D, linalg::blas_backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    //expose the ttn node class.  This is our core tensor network object.
    py::class_<ttype>(m, (label).c_str(), py::buffer_protocol())
        .def(py::init([](py::buffer &b)
            {
                ttype tens;
                CALL_AND_RETHROW(copy_pybuffer_to_tensor(b, tens));
                return tens;
            }
        ))
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
                            py::format_descriptor<T>::format(),     //Python struct-style format descriptor
                            D,                                      //Number of dimensions D
                            shape_arr,                              //shape of the array
                            stride_arr                              //strides of the array
                        );  
                    }
                   )
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();});
        //.def("clear", &ttype::clear);
}


void initialise_tensors(py::module& m);


#endif  //PYTHON_BINDING_LINALG_TENSOR_HPP


