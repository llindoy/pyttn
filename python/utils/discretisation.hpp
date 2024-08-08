#ifndef PYTHON_BINDING_DISCRETISATION_HPP
#define PYTHON_BINDING_DISCRETISATION_HPP

#include <utils/bath/density_discretisation.hpp>
#include <utils/bath/orthopol_discretisation.hpp>
#include <linalg/linalg.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

#include "../utils.hpp"

namespace py=pybind11;

template <typename T> 
void init_discretisation(py::module &m)
{
    using namespace utils;

    //wrapper for the orthopol manager class 
    py::class_<density_discretisation>(m, "density_discretisation")
        .def_static
            (
                "discretise", 
                [](const std::function<T(const T&)>& J, const std::function<T(const T&)>& rho, T wmin, T wmax, size_t N, T atol, T rtol, size_t nquad, T wtol, T ftol, size_t niter)
                {
                    std::pair<std::vector<T>, std::vector<T>> wg;
                    density_discretisation::discretise(J, rho, wmin, wmax, N, std::get<0>(wg), std::get<1>(wg), atol, rtol, nquad, wtol, ftol, niter);
                    return wg;
                }, 
                py::arg(), py::arg(),  py::arg(), py::arg(), py::arg(), py::arg("atol") = 1e-10, py::arg("rtol") = 1e-12, py::arg("nquad") = 100, py::arg("wtol") =  1e-12, py::arg("ftol") = 1e-12, py::arg("niters") = 100
            );

    py::class_<orthopol_discretisation>(m, "orthopol_discretisation")
        .def_static
            (
                "discretise", 
                [](const std::function<T(const T&)>& J, orthopol<T>& poly, T wrange, T moment_scaling, T atol, T rtol, size_t nquad)
                {
                    std::pair<std::vector<T>, std::vector<T>> wg;
                    orthopol_discretisation::discretise(J, poly, std::get<0>(wg), std::get<1>(wg), wrange, moment_scaling, atol, rtol, nquad);
                    return wg;
                }, 
                py::arg(), py::arg(),  py::arg("wrange") = 1.0, py::arg("moment_scaling") = 1.0, py::arg("atol") = 1e-10, py::arg("rtol") = 1e-12, py::arg("nquad") = 100
            )
        .def_static
            (
                "discretise", 
                [](const std::function<T(const T&)>& J, T wmin, T wmax, size_t N, T moment_scaling, T alpha, T beta, T atol, T rtol, size_t nquad)
                {
                    std::pair<std::vector<T>, std::vector<T>> wg;
                    orthopol_discretisation::discretise(J, wmin, wmax, N, std::get<0>(wg), std::get<1>(wg), moment_scaling, alpha, beta, atol, rtol, nquad);
                    return wg;
                }, 
                py::arg(), py::arg(), py::arg(), py::arg(), py::arg("moment_scaling") = 1.0, py::arg("alpha") = 1.0, py::arg("beta") = 0.0, py::arg("atol") = 1e-10, py::arg("rtol") = 1e-12, py::arg("nquad") = 100
            );
}

void initialise_discretisation(py::module& m);

#endif
