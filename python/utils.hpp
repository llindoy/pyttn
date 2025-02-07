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

template <typename T>
class numpy_converter
{
    public:
    using type = T;
};

template <>
class numpy_converter<complex<float>>
{
public:
    using type = std::complex<float>;
};

template <>
class numpy_converter<complex<double>>
{
public:
    using type = std::complex<double>;
};

template <typename T> 
void pybuffer_to_vector(const py::buffer& b, std::vector<T>& res)
{
    using _T = typename numpy_converter<T>::type;
    using real_type = typename get_real_type<T>::type;


    py::buffer_info info = b.request();
    if(info.format == py::format_descriptor<_T>::format())
    {
        T* ptr = static_cast<T*>(info.ptr);
        res = std::vector<T>(ptr, ptr+info.size);
    }
    else if(info.format == py::format_descriptor<real_type>::format())
    {
        res.resize(info.size);
        real_type* ptr = static_cast<real_type*>(info.ptr);

        for(size_t i = 0; i < static_cast<size_t>(info.size); ++i)
        {
            res[i] = ptr[i];
        }
    }
    else
    {
        RAISE_EXCEPTION("Failed to convert pybuffer to vector.")
    }
}


template <typename backend>
class pybuffer_converter
{
public:
    template<typename T, size_t D>
    static void convert_to_tensor(const py::buffer& b, linalg::tensor<T, D, backend>& t)
    {
        using _T = typename numpy_converter<T>::type;

        py::buffer_info info = b.request();
        if(info.format != py::format_descriptor<_T>::format())
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
    static void copy_to_tensor(const py::buffer& b, linalg::tensor<T, D, backend>& t)
    {
        py::buffer_info info = b.request();
        using _T = typename numpy_converter<T>::type;

        using real_type = typename get_real_type<T>::type;

        if(info.format == py::format_descriptor<_T>::format())
        {
            convert_to_tensor(b, t);
        }
        else if(info.format == py::format_descriptor<real_type>::format())
        {
            linalg::tensor<real_type, D, backend> temp;
            convert_to_tensor(b, temp);
            t = temp;
        }
        else
        {
            RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect scalar type.");
        }
    }

    template<typename T>
    static void convert_to_diagonal_matrix(const py::buffer& b, diagonal_matrix<T, backend>& t)
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
    static void copy_to_diagonal_matrix(const py::buffer& b, diagonal_matrix<T, backend>& t)
    {
        py::buffer_info info = b.request();
        using _T = typename numpy_converter<T>::type;

        using real_type = typename get_real_type<T>::type;
        if(info.format == py::format_descriptor<_T>::format())
        {
            convert_to_diagonal_matrix(b, t);
        }
        else if(info.format == py::format_descriptor<real_type>::format())
        {
            diagonal_matrix<real_type, backend> temp;
            convert_to_diagonal_matrix(b, temp);
            t=temp;
        }
        else
        {
            RAISE_EXCEPTION("Incompatible format for arrays assignment. Incorrect dimension.");
        }
    }
};


}

#ifdef PYTTN_BUILD_CUDA
template <typename backend> class other_backend;
template <> class other_backend<linalg::blas_backend>
{
public:
    using type = linalg::cuda_backend;
};

template <> class other_backend<linalg::cuda_backend>
{
public:
    using type = linalg::blas_backend;
};
#endif

#endif
