#ifndef PYTHON_BINDING_LINALG_ORTHOGONAL_VECTOR_GEN_HPP
#define PYTHON_BINDING_LINALG_ORTHOGONAL_VECTOR_GEN_HPP

#include <linalg/linalg.hpp>

#include "../utils.hpp"

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

template <typename T, typename backend>
void init_orthogonal_vector(py::module& m, const std::string& label)
{
    using namespace linalg;
    using size_type = typename backend::size_type;
    using ttype = orthogonal_vector<T, backend>;
    using vectype = linalg::vector<T, backend>;
    using mattype = linalg::matrix<T, backend>;

    py::class_<ttype>(m, label.c_str(), py::module_local())
        .def_static("pad_random", 
            [](mattype& a, size_type i, random_engine<backend>& rng)
            {
                ttype::pad_random(a, i, rng);
            }
        )        
        .def_static("generate", 
            [](const mattype& a, random_engine<backend>& rng)
            {
                vectype b(a.shape(1));
                ttype::generate(a, b, rng);
                return b;
            }
        );

}

template <typename backend>
void init_random_engine(py::module& m, const std::string& label)
{
    using namespace linalg;
    using ttype = random_engine<backend>;

    py::class_<ttype>(m, label.c_str())
        .def(py::init());
}

template <typename real_type, typename backend>
void initialise_orthogonal_vector(py::module& m)
{
    using complex_type = linalg::complex<real_type>;

    init_random_engine<backend>(m, "random_engine");
    init_orthogonal_vector<real_type, backend>(m, "orthogonal_vector_real");
    init_orthogonal_vector<complex_type, backend>(m, "orthogonal_vector_complex");
}
#endif //PYTHON_BINDING_LINALG_CUDA_BACKEND_HPP