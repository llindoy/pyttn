#ifndef PYTHON_BINDING_LIOUVILLE_SPACE_SUPEROPERATORS_HPP
#define PYTHON_BINDING_LIOUVILLE_SPACE_SUPEROPERATORS_HPP

#include <ttns_lib/sop/sSOP.hpp>
#include <ttns_lib/sop/SOP.hpp>
#include <ttns_lib/sop/liouville_space.hpp>
#include <ttns_lib/sop/operator_dictionaries/default_operator_dictionaries.hpp>

#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <pybind11/cast.h>
#include <pybind11/stl_bind.h>
#include <pybind11/pytypes.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>

namespace py=pybind11;

template <typename real_type, typename backend>
void initialise_liouville_space(py::module& m)
{
    using namespace ttns;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    using opdictr = operator_dictionary<real_type, backend>;
#endif
    using opdictc = operator_dictionary<complex_type, backend>;

    py::class_<liouville_space>(m, "liouville_space")
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, SOP<complex_type>&, complex_type)>(&liouville_space::left_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, SOP<complex_type>&, complex_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, SOP<complex_type>&, complex_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, SOP<complex_type>&, complex_type)>(&liouville_space::anticommutator_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )        
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, const opdictc&, SOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::left_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, const opdictc&, SOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, const opdictc&, SOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const SOP<complex_type>&, const system_modes&, const opdictc&, SOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::anticommutator_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, sSOP<complex_type>&, complex_type)>(&liouville_space::left_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, sSOP<complex_type>&, complex_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, sSOP<complex_type>&, complex_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, sSOP<complex_type>&, complex_type)>(&liouville_space::anticommutator_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )        
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, const opdictc&, sSOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::left_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, const opdictc&, sSOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, const opdictc&, sSOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const sSOP<complex_type>&, const system_modes&, const opdictc&, sSOP<complex_type>&, opdictc&, complex_type)>(&liouville_space::anticommutator_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=complex_type(1)
            )
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, sSOP<real_type>&, real_type)>(&liouville_space::left_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, sSOP<real_type>&, real_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, sSOP<real_type>&, real_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, sSOP<real_type>&, real_type)>(&liouville_space::anticommutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )

//Functions for handling real valued SOPs.  These should only be allowed if the user has compiled with the option BUILD_REAL_TTN 
#ifdef BUILD_REAL_TTN
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, SOP<real_type>&, real_type)>(&liouville_space::left_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, SOP<real_type>&, real_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, SOP<real_type>&, real_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, SOP<real_type>&, real_type)>(&liouville_space::anticommutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, const opdictr&, SOP<real_type>&, opdictr&, real_type)>(&liouville_space::left_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, const opdictr&, SOP<real_type>&, opdictr&, real_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, const opdictr&, SOP<real_type>&, opdictr&, real_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const SOP<real_type>&, const system_modes&, const opdictr&, SOP<real_type>&, opdictr&, real_type)>(&liouville_space::anticommutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )        
        .def_static(
              "left_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, const opdictr&, sSOP<real_type>&, opdictr&, real_type)>(&liouville_space::left_superoperator), 
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "right_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, const opdictr&, sSOP<real_type>&, opdictr&, real_type)>(&liouville_space::right_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "commutator_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, const opdictr&, sSOP<real_type>&, opdictr&, real_type)>(&liouville_space::commutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
        .def_static(
              "anticommutator_superoperator", 
              static_cast<void (*)(const sSOP<real_type>&, const system_modes&, const opdictr&, sSOP<real_type>&, opdictr&, real_type)>(&liouville_space::anticommutator_superoperator),
              py::arg(), py::arg(), py::arg(), py::arg(), py::arg(), py::arg("coeff")=real_type(1)
            )
#endif
        ;
}
#endif
