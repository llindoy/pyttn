#ifndef PYTHON_BINDING_UTILS_HPP
#define PYTHON_BINDING_UTILS_HPP

#include <linalg/linalg.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

namespace linalg
{
template<typename T, size_t D>
void convert_pybuffer_to_tensor(py::buffer& b, linalg::tensor<T, D>& t)
{
    py::buffer_info info = b.request();
    if(info.format != py::format_descriptor<T>::format())
    {
        RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect scalar type.");
    }
    if(info.ndim != D)
    {
        RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect dimension.");
    }
    using ttype = linalg::tensor<T, D>;
    using size_type = typename ttype::size_type;
    
    typename ttype::shape_type shape;
    typename ttype::shape_type stride;
    
    for(size_type i = 0; i < D; ++i)
    {
        shape[i] = info.shape[i];
        stride[i] = static_cast<size_type>(info.strides[i])/sizeof(T);
    }
    t.resize(shape);
    t.set_buffer(static_cast<T*>(info.ptr), shape, stride);
}

template<typename T, size_t D>
void copy_pybuffer_to_tensor(py::buffer& b, linalg::tensor<T, D>& t)
{
    py::buffer_info info = b.request();

    using real_type = typename get_real_type<T>::type;

    if(info.format == py::format_descriptor<T>::format())
    {
        convert_pybuffer_to_tensor(b, t);
    }
    else if(info.format == py::format_descriptor<real_type>::format())
    {
        linalg::tensor<real_type, D> temp;
        convert_pybuffer_to_tensor(b, temp);
        t = temp;
    }
    else
    {
        RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect scalar type.");
    }
}


template<typename T>
void convert_pybuffer_to_diagonal_matrix(py::buffer& b, diagonal_matrix<T>& t)
{
    py::buffer_info info = b.request();
    if(info.ndim != 1)
    {
        RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect dimension.");
    }
    t.resize(info.shape[0], info.shape[0]);
    t.set_buffer(static_cast<T*>(info.ptr), info.shape[0]);
}
template<typename T>
void copy_pybuffer_to_diagonal_matrix(py::buffer& b, diagonal_matrix<T>& t)
{
    py::buffer_info info = b.request();

    using real_type = typename get_real_type<T>::type;
    if(info.format == py::format_descriptor<T>::format())
    {
        convert_pybuffer_to_diagonal_matrix(b, t);
    }
    else if(info.format == py::format_descriptor<real_type>::format())
    {
        diagonal_matrix<real_type> temp;
        convert_pybuffer_to_diagonal_matrix(b, temp);
        t=temp;
    }
    else
    {
        RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect dimension.");
    }
}

}


#endif
