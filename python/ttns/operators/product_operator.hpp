#ifndef PYTHON_BINDING_TTNS_PRODUCT_OPERATOR_HPP
#define PYTHON_BINDING_TTNS_PRODUCT_OPERATOR_HPP

#include <ttns_lib/operators/product_operator.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename T>
void init_product_operator(py::module &m, const std::string& label)
{
    using namespace ttns;

    using opdict = operator_dictionary<T, linalg::blas_backend>;
    using pop = product_operator<T, linalg::blas_backend>;
    using real_type = typename linalg::get_real_type<T>::type;

    //the base primitive operator type
    py::class_<pop>(m, (std::string("product_operator_")+label).c_str())
        .def(py::init())
        .def(py::init<const pop&>())
        .def(py::init<sNBO<real_type>&,  const system_modes&, bool>(), 
              py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<real_type>&, const system_modes&, const opdict&, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<T>&,  const system_modes&, bool>(), 
              py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sNBO<T>&, const system_modes&, const opdict&, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sPOP&,  const system_modes&, bool>(), 
              py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sPOP&, const system_modes&, const opdict&, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sOP&,  const system_modes&, bool>(), 
              py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def(py::init<sOP&, const system_modes&, const opdict&, bool>(), 
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true)
        .def("assign", [](pop& self, const pop& o){self=o;})
        .def("__copy__",[](const pop& o){return pop(o);})
        .def("__deepcopy__", [](const pop& o, py::dict){return pop(o);}, py::arg("memo"))
        .def(
              "initialise", 
              [](pop& o, sNBO<T>& sop, const system_modes& sys, bool use_sparse)
              {
                  o.initialise(sop, sys, use_sparse);
              },
              py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sNBO<T>& sop, const system_modes& sys, const opdict& opd, bool use_sparse)
              {
                  o.initialise(sop, sys, opd, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sNBO<real_type>& sop, const system_modes& sys, bool use_sparse)
              {
                  o.initialise(sop, sys, use_sparse);
              },
              py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sNBO<real_type>& sop, const system_modes& sys, const opdict& opd, bool use_sparse)
              {
                  o.initialise(sop, sys, opd, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sPOP& sop, const system_modes& sys, bool use_sparse)
              {
                  o.initialise(sop, sys, use_sparse);
              },
              py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sPOP& sop, const system_modes& sys, const opdict& opd, bool use_sparse)
              {
                  o.initialise(sop, sys, opd, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sOP& sop, const system_modes& sys, bool use_sparse)
              {
                  o.initialise(sop, sys, use_sparse);
              },
              py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def(
              "initialise", 
              [](pop& o, sOP& sop, const system_modes& sys, const opdict& opd, bool use_sparse)
              {
                  o.initialise(sop, sys, opd, use_sparse);
              },
              py::arg(), py::arg(), py::arg(), py::arg("use_sparse") = true
            )
        .def("clear", &pop::clear)
        .def("nmodes", &pop::nmodes)
        .def("__str__", [](const pop& o){std::ostringstream oss; oss << o; return oss.str();});
}

void initialise_product_operator(py::module& m);


#endif

