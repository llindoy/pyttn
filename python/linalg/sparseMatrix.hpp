#ifndef PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP
#define PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP

#include <linalg/linalg.hpp>
#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T, typename backend>
void init_csr_matrix(py::module &m, const std::string& label)
{
    using namespace linalg;
    using ttype = csr_matrix<T, backend>;
    using index_type = typename ttype::index_type;
    using coo_type = std::vector<std::tuple<index_type, index_type, T>>;
    using real_type = typename linalg::get_real_type<T>::type;

    //to do figure out a way of exposing the c++ buffers to python
    py::class_<ttype>(m, (label).c_str())
        .def(py::init<const std::vector<T>&, const std::vector<index_type>&, const std::vector<index_type>&, size_t>(), py::arg(), py::arg(), py::arg(), py::arg("ncols")=0)
        .def(py::init<const coo_type&, size_t, size_t>(), py::arg(), py::arg("nrows")=0, py::arg("ncols")=0)
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();});
}

template <typename T, typename backend>
void init_diagonal_matrix(py::module& m, const std::string& label)
{
    using namespace linalg;
    using ttype = diagonal_matrix<T, backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    using conv = pybuffer_converter<backend>;
    py::class_<ttype>(m, (label).c_str(), py::buffer_protocol())
        .def(py::init([](py::buffer &b)
            {
                ttype tens;
                conv::copy_to_diagonal_matrix(b, tens);
                return tens;
            }
        ))
        .def(py::init<const std::vector<T>&>())
        .def(py::init<const std::vector<T>&, size_t>())
        .def(py::init<const std::vector<T>&, size_t, size_t>())
        .def(py::init<const tensor<T, 1>&>())
        .def(py::init<const tensor<T, 1>&, size_t>())
        .def(py::init<const tensor<T, 1>&, size_t, size_t>())
        .def_buffer([](ttype& mi) -> py::buffer_info 
                    {
                        return py::buffer_info
                        (
                            mi.buffer(),                            //pointer to buffer
                            sizeof(T),                              //size of one scalar
                            py::format_descriptor<T>::format(),     //Python struct-style format descriptor
                            1,                                      //Number of dimensions D
                            std::vector<size_t>{mi.nrows()},        //shape of the array
                            std::vector<size_t>{sizeof(T)}          //strides of the array
                        );  
                    }
                   )
        .def("complex_dtype", [](const ttype&){return !std::is_same<T, real_type>::value;})
        .def("__str__", [](const ttype& o){std::stringstream oss;   oss << o; return oss.str();});

    //expose the ttn node class.  This is our core tensor network object.
}


template <typename backend>
void initialise_sparse_matrices(py::module& m)
{
    using real_type = double;
    using complex_type = linalg::complex<real_type>;
    init_csr_matrix<real_type, backend>(m, "csr_matrix_real");
    init_csr_matrix<complex_type, backend>(m, "csr_matrix_complex");
    init_diagonal_matrix<real_type, backend>(m, "diagonal_matrix_real");
    init_diagonal_matrix<complex_type, backend>(m, "diagonal_matrix_complex");
}

#endif  //PYTHON_BINDING_LINALG_SPARSE_MATRIX_HPP


