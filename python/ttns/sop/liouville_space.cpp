#include "liouville_space.hpp"

namespace py=pybind11;

void initialise_liouville_space(py::module &m)
{
    using namespace ttns;

    using real_type = double;
    using complex_type = linalg::complex<real_type>;

#ifdef BUILD_REAL_TTN
    using opdictr = operator_dictionary<real_type, linalg::blas_backend>;
#endif
    using opdictc = operator_dictionary<complex_type, linalg::blas_backend>;

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
#endif
        ;
}
